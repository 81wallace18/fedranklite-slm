from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


def load_model(cfg: dict, num_labels: int = 2):
    model_name = cfg["model"]["name"]
    quant = cfg["model"]["quantization"]
    lora_cfg = cfg["model"]["lora"]

    bnb_config = None
    if quant == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
        device_map="auto",
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_cfg["r_max"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def get_lora_state(model) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu().clone()
        for k, v in model.named_parameters()
        if "lora_" in k and v.requires_grad
    }


def set_lora_state(model, state: dict[str, torch.Tensor]):
    own = dict(model.named_parameters())
    for k, v in state.items():
        if k in own:
            own[k].data.copy_(v.to(own[k].device))


def mask_lora_rank(model, rank: int):
    for name, param in model.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        if "lora_A" in name:
            # A shape: (r_max, in_features) — freeze rows beyond rank
            if param.shape[0] > rank:
                param.data[rank:] = 0.0
                param.register_hook(_zero_grad_hook(rank, dim=0))
        elif "lora_B" in name:
            # B shape: (out_features, r_max) — freeze cols beyond rank
            if param.shape[1] > rank:
                param.data[:, rank:] = 0.0
                param.register_hook(_zero_grad_hook(rank, dim=1))


def _zero_grad_hook(rank: int, dim: int):
    def hook(grad):
        if dim == 0:
            grad[rank:] = 0.0
        else:
            grad[:, rank:] = 0.0
        return grad
    return hook


def truncate_lora_state(state: dict[str, torch.Tensor], rank: int) -> dict[str, torch.Tensor]:
    truncated = {}
    for k, v in state.items():
        if "lora_A" in k:
            truncated[k] = v[:rank]
        elif "lora_B" in k:
            truncated[k] = v[:, :rank]
        else:
            truncated[k] = v.clone()
    return truncated


def pad_lora_state(state: dict[str, torch.Tensor], r_max: int) -> dict[str, torch.Tensor]:
    padded = {}
    for k, v in state.items():
        if "lora_A" in k:
            current_r = v.shape[0]
            if current_r < r_max:
                pad = torch.zeros(r_max - current_r, v.shape[1], dtype=v.dtype)
                v = torch.cat([v, pad], dim=0)
            padded[k] = v
        elif "lora_B" in k:
            current_r = v.shape[1]
            if current_r < r_max:
                pad = torch.zeros(v.shape[0], r_max - current_r, dtype=v.dtype)
                v = torch.cat([v, pad], dim=1)
            padded[k] = v
        else:
            padded[k] = v.clone()
    return padded
