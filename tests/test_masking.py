import torch
import pytest
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

from src.models import mask_lora_rank, get_lora_state, truncate_lora_state, pad_lora_state


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


class TestMasking:
    def test_mask_zeros_beyond_rank(self, small_model):
        mask_lora_rank(small_model, rank=4)
        state = get_lora_state(small_model)
        for k, v in state.items():
            if "lora_A" in k:
                assert torch.all(v[4:] == 0), f"{k} rows beyond rank 4 should be zero"
            if "lora_B" in k:
                assert torch.all(v[:, 4:] == 0), f"{k} cols beyond rank 4 should be zero"

    def test_gradient_zeroed_beyond_rank(self, small_model):
        mask_lora_rank(small_model, rank=4)
        # do a fake forward/backward
        dummy_input = {
            "input_ids": torch.randint(0, 100, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
            "labels": torch.tensor([0, 1]),
        }
        output = small_model(**dummy_input)
        output.loss.backward()

        for name, param in small_model.named_parameters():
            if "lora_A" in name and param.grad is not None:
                assert torch.all(param.grad[4:] == 0), f"Grad of {name} beyond rank 4 should be zero"
            if "lora_B" in name and param.grad is not None:
                assert torch.all(param.grad[:, 4:] == 0), f"Grad of {name} beyond rank 4 should be zero"


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
