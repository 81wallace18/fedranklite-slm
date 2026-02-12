from __future__ import annotations

import time
from dataclasses import dataclass, field

import psutil
import torch
from torch.utils.data import DataLoader

from .models import get_lora_state, mask_lora_rank, set_lora_state, truncate_lora_state


@dataclass
class ClientResult:
    client_id: int
    rank_used: int
    lora_delta: dict[str, torch.Tensor]
    n_samples: int
    loss_before: float
    loss_after: float
    train_time: float
    peak_memory_mb: float
    bytes_sent: int


def train_client(
    client_id: int,
    model,
    dataloader: DataLoader,
    global_lora: dict[str, torch.Tensor],
    assigned_rank: int,
    cfg: dict,
) -> ClientResult:
    set_lora_state(model, global_lora)
    mask_lora_rank(model, assigned_rank)

    device = next(model.parameters()).device
    is_cuda = device.type == "cuda"

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # evaluate loss before training
    loss_before = _eval_loss(model, dataloader, device)

    # setup optimizer
    train_cfg = cfg["training"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=train_cfg["lr"])

    model.train()
    t0 = time.time()
    n_samples = 0

    for epoch in range(train_cfg["local_epochs"]):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, train_cfg["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()
            n_samples += batch["input_ids"].shape[0]

    train_time = time.time() - t0

    # loss after
    loss_after = _eval_loss(model, dataloader, device)

    # memory
    if is_cuda:
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_mem = psutil.Process().memory_info().rss / (1024 ** 2)

    # extract and truncate LoRA delta
    updated_lora = get_lora_state(model)
    delta = {
        k: updated_lora[k] - global_lora[k].to(updated_lora[k].device)
        for k in updated_lora
    }
    delta = truncate_lora_state(delta, assigned_rank)

    bytes_sent = sum(v.numel() * v.element_size() for v in delta.values())

    return ClientResult(
        client_id=client_id,
        rank_used=assigned_rank,
        lora_delta=delta,
        n_samples=n_samples,
        loss_before=loss_before,
        loss_after=loss_after,
        train_time=train_time,
        peak_memory_mb=peak_mem,
        bytes_sent=bytes_sent,
    )


@torch.no_grad()
def _eval_loss(model, dataloader: DataLoader, device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item() * batch["input_ids"].shape[0]
        count += batch["input_ids"].shape[0]
    model.train()
    return total_loss / max(count, 1)
