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


class TestTruncation:
    def test_truncate_reduces_actual_shape(self, small_model):
        """After truncation, lora_A has shape (rank, d) and lora_B has shape (d, rank)."""
        mask_lora_rank(small_model, rank=4)
        for name, param in small_model.named_parameters():
            if "lora_A" in name and param.requires_grad:
                assert param.shape[0] == 4, f"{name} should have 4 rows, got {param.shape[0]}"
            if "lora_B" in name and param.requires_grad:
                assert param.shape[1] == 4, f"{name} should have 4 cols, got {param.shape[1]}"

    def test_truncated_forward_works(self, small_model):
        """Model can still do forward pass after truncation."""
        mask_lora_rank(small_model, rank=4)
        dummy_input = {
            "input_ids": torch.randint(0, 100, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
            "labels": torch.tensor([0, 1]),
        }
        output = small_model(**dummy_input)
        assert output.loss is not None
        output.loss.backward()

    def test_truncated_backward_only_updates_rank_dims(self, small_model):
        """After truncation to rank=4, gradients only exist for 4 dimensions."""
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
                assert param.grad.shape[0] == 4
            if "lora_B" in name and param.grad is not None:
                assert param.grad.shape[1] == 4

    def test_restore_after_truncate(self, small_model):
        """restore_lora_rank pads back to r_max."""
        original_shapes = {}
        for name, param in small_model.named_parameters():
            if "lora_" in name and param.requires_grad:
                original_shapes[name] = param.shape

        mask_lora_rank(small_model, rank=4)
        restore_lora_rank(small_model, r_max=8)

        for name, param in small_model.named_parameters():
            if name in original_shapes:
                assert param.shape == original_shapes[name], (
                    f"{name}: expected {original_shapes[name]}, got {param.shape}"
                )

    def test_restore_preserves_trained_dims(self, small_model):
        """After truncate→train→restore, the first rank dims are preserved."""
        state_before = get_lora_state(small_model)
        mask_lora_rank(small_model, rank=4)

        # modify the truncated params to simulate training
        for name, param in small_model.named_parameters():
            if "lora_" in name and param.requires_grad:
                param.data.fill_(1.0)

        restore_lora_rank(small_model, r_max=8)
        state_after = get_lora_state(small_model)

        for k in state_after:
            if "lora_A" in k:
                assert torch.all(state_after[k][:4] == 1.0)
                assert torch.all(state_after[k][4:] == 0.0)
            elif "lora_B" in k:
                assert torch.all(state_after[k][:, :4] == 1.0)
                assert torch.all(state_after[k][:, 4:] == 0.0)

    def test_no_truncation_when_rank_equals_rmax(self, small_model):
        """When rank == r_max, nothing changes."""
        shapes_before = {
            name: param.shape
            for name, param in small_model.named_parameters()
            if "lora_" in name and param.requires_grad
        }
        mask_lora_rank(small_model, rank=8)
        for name, param in small_model.named_parameters():
            if name in shapes_before:
                assert param.shape == shapes_before[name]


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
