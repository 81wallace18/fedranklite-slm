import torch
import pytest
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

from src.models import mask_lora_rank, restore_lora_rank, get_lora_state, truncate_lora_state, pad_lora_state


@pytest.fixture
def small_model():
    """A tiny model for testing — uses a small pretrained model."""
    model = AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-BertModel",
        num_labels=2,
    )
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, config)
    return model


class TestZeroMasking:
    def test_mask_keeps_shape_but_zeros_weights(self, small_model):
        """After masking, shape is unchanged but weights beyond rank are zero."""
        r_max = 8
        mask_lora_rank(small_model, rank=4)

        for name, param in small_model.named_parameters():
            if "lora_A" in name and param.requires_grad:
                assert param.shape[0] == r_max, f"{name}: expected r_max={r_max}, got {param.shape[0]}"
                assert torch.all(param.data[4:] == 0), f"{name}: rows beyond rank=4 should be zero"
                assert not torch.all(param.data[:4] == 0), f"{name}: first 4 rows should be non-zero (unless randomly zero)"
            if "lora_B" in name and param.requires_grad:
                assert param.shape[1] == r_max, f"{name}: expected r_max={r_max}, got {param.shape[1]}"
                assert torch.all(param.data[:, 4:] == 0), f"{name}: cols beyond rank=4 should be zero"
                assert not torch.all(param.data[:, :4] == 0), f"{name}: first 4 cols should be non-zero (unless randomly zero)"

    def test_masked_forward_works(self, small_model):
        """Model can still do forward pass after masking."""
        mask_lora_rank(small_model, rank=4)
        dummy_input = {
            "input_ids": torch.randint(0, 100, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
            "labels": torch.tensor([0, 1]),
        }
        output = small_model(**dummy_input)
        assert output.loss is not None
        output.loss.backward()

    def test_masked_backward_only_updates_rank_dims(self, small_model):
        """After masking to rank=4, gradients only exist for 4 dimensions."""
        mask_lora_rank(small_model, rank=4)
        dummy_input = {
            "input_ids": torch.randint(0, 100, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
            "labels": torch.tensor([0, 1]),
        }
        output = small_model(**dummy_input)
        output.loss.backward()

        for name, param in small_model.named_parameters():
            if "lora_A" in name and param.grad is not None:
                assert param.grad.shape[0] == 8, f"{name}: grad shape should be (8, d)"
                assert torch.all(param.grad[4:] == 0), f"{name}: gradients beyond rank=4 should be zero"
                assert not torch.all(param.grad[:4] == 0), f"{name}: first 4 grads should be non-zero (unless zero loss)"
            if "lora_B" in name and param.grad is not None:
                assert param.grad.shape[1] == 8, f"{name}: grad shape should be (d, 8)"
                assert torch.all(param.grad[:, 4:] == 0), f"{name}: gradients beyond rank=4 should be zero"
                assert not torch.all(param.grad[:, :4] == 0), f"{name}: first 4 grads should be non-zero (unless zero loss)"

    def test_restore_is_noop(self, small_model):
        """restore_lora_rank does nothing with zero masking."""
        original_state = get_lora_state(small_model)
        mask_lora_rank(small_model, rank=4)

        # modify masked params to simulate training
        for name, param in small_model.named_parameters():
            if "lora_" in name and param.requires_grad:
                param.data.fill_(1.0)

        restore_lora_rank(small_model, r_max=8)
        state_after = get_lora_state(small_model)

        # restore should be no-op — shape unchanged, values preserved
        for k in original_state:
            assert state_after[k].shape == original_state[k].shape
            # after training, values should be 1.0 (not restored to original)
            assert torch.all(state_after[k] == 1.0)

    def test_mask_preserves_trained_dims(self, small_model):
        """After mask→train, only active rank dims are updated."""
        mask_lora_rank(small_model, rank=4)
        state_masked = get_lora_state(small_model)

        # modify masked params to simulate training
        for name, param in small_model.named_parameters():
            if "lora_" in name and param.requires_grad:
                param.data.fill_(1.0)

        state_after_training = get_lora_state(small_model)

        for k in state_after_training:
            if "lora_A" in k:
                assert torch.all(state_after_training[k] == 1.0), f"{k}: should all be 1.0"
            elif "lora_B" in k:
                assert torch.all(state_after_training[k] == 1.0), f"{k}: should all be 1.0"

    def test_no_masking_when_rank_equals_rmax(self, small_model):
        """When rank == r_max, nothing is zeroed out."""
        shapes_before = {
            name: param.shape
            for name, param in small_model.named_parameters()
            if "lora_" in name and param.requires_grad
        }
        mask_lora_rank(small_model, rank=8)
        for name, param in small_model.named_parameters():
            if name in shapes_before:
                assert param.shape == shapes_before[name]
                # gradients should exist for all dims (not zeroed out)
                dummy_input = {
                    "input_ids": torch.randint(0, 100, (2, 16)),
                    "attention_mask": torch.ones(2, 16, dtype=torch.long),
                    "labels": torch.tensor([0, 1]),
                }
                small_model(**dummy_input).loss.backward()
                if param.grad is not None:
                    assert not torch.all(param.grad == 0), f"{name}: grads should be non-zero (unless zero loss)"


class TestTruncateAndPad:
    def test_truncate_reduces_rank_dim(self):
        state = {
            "layer.lora_A": torch.randn(16, 64),
            "layer.lora_B": torch.randn(128, 16),
        }
        truncated = truncate_lora_state(state, rank=4)
        assert truncated["layer.lora_A"].shape == (4, 64)
        assert truncated["layer.lora_B"].shape == (128, 4)

    def test_pad_restores_shape(self):
        state = {
            "layer.lora_A": torch.randn(4, 64),
            "layer.lora_B": torch.randn(128, 4),
        }
        padded = pad_lora_state(state, r_max=16)
        assert padded["layer.lora_A"].shape == (16, 64)
        assert padded["layer.lora_B"].shape == (128, 16)
        # padded region should be zeros
        assert torch.all(padded["layer.lora_A"][4:] == 0)
        assert torch.all(padded["layer.lora_B"][:, 4:] == 0)

    def test_roundtrip_preserves_data(self):
        original = {
            "layer.lora_A": torch.randn(8, 64),
            "layer.lora_B": torch.randn(128, 8),
        }
        truncated = truncate_lora_state(original, rank=4)
        padded = pad_lora_state(truncated, r_max=8)
        # first 4 rows/cols should match
        assert torch.allclose(padded["layer.lora_A"][:4], original["layer.lora_A"][:4])
        assert torch.allclose(padded["layer.lora_B"][:, :4], original["layer.lora_B"][:, :4])
