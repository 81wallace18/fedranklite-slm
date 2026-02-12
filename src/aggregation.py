from __future__ import annotations

import torch

from .client import ClientResult
from .models import pad_lora_state


def aggregate_lora(
    global_lora: dict[str, torch.Tensor],
    results: list[ClientResult],
    r_max: int,
    method: str = "fedavg",
    weighting: str = "num_samples",
) -> dict[str, torch.Tensor]:
    if not results:
        return global_lora

    if method == "fedavg":
        return _fedavg(global_lora, results, r_max, weighting)
    elif method == "fedavg_blockwise":
        return _fedavg_blockwise(global_lora, results, r_max, weighting)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def _fedavg(
    global_lora: dict[str, torch.Tensor],
    results: list[ClientResult],
    r_max: int,
    weighting: str,
) -> dict[str, torch.Tensor]:
    weights = _compute_weights(results, weighting)
    agg = {}

    for key in global_lora:
        padded_deltas = []
        for res in results:
            delta = res.lora_delta[key] if key in res.lora_delta else torch.zeros_like(global_lora[key])
            delta = _pad_tensor(delta, global_lora[key].shape, key)
            padded_deltas.append(delta)

        weighted_sum = sum(w * d for w, d in zip(weights, padded_deltas))
        agg[key] = global_lora[key] + weighted_sum

    return agg


def _fedavg_blockwise(
    global_lora: dict[str, torch.Tensor],
    results: list[ClientResult],
    r_max: int,
    weighting: str,
) -> dict[str, torch.Tensor]:
    """Aggregate by blocks: for rank dimension r, only clients that trained
    with rank >= r contribute to that slice."""
    agg = {}

    for key in global_lora:
        shape = global_lora[key].shape
        accumulated = torch.zeros_like(global_lora[key])

        if "lora_A" in key:
            rank_dim = 0
        elif "lora_B" in key:
            rank_dim = 1
        else:
            # non-rank parameter: simple weighted average
            ws = _compute_weights(results, weighting)
            accumulated = sum(
                w * (res.lora_delta.get(key, torch.zeros_like(global_lora[key])))
                for w, res in zip(ws, results)
            )
            agg[key] = global_lora[key] + accumulated
            continue

        for r in range(shape[rank_dim]):
            contributors = [res for res in results if res.rank_used > r]
            if not contributors:
                break
            ws = _compute_weights(contributors, weighting)
            for w, res in zip(ws, contributors):
                delta = res.lora_delta.get(key, torch.zeros_like(global_lora[key]))
                delta = _pad_tensor(delta, shape, key)
                if rank_dim == 0:
                    accumulated[r] += w * delta[r]
                else:
                    accumulated[:, r] += w * delta[:, r]

        agg[key] = global_lora[key] + accumulated

    return agg


def _compute_weights(results: list[ClientResult], weighting: str) -> list[float]:
    if weighting == "uniform":
        n = len(results)
        return [1.0 / n] * n
    # num_samples
    total = sum(r.n_samples for r in results)
    if total == 0:
        n = len(results)
        return [1.0 / n] * n
    return [r.n_samples / total for r in results]


def _pad_tensor(tensor: torch.Tensor, target_shape: tuple, key: str) -> torch.Tensor:
    if tensor.shape == target_shape:
        return tensor
    padded = torch.zeros(target_shape, dtype=tensor.dtype)
    if "lora_A" in key:
        padded[: tensor.shape[0]] = tensor
    elif "lora_B" in key:
        padded[:, : tensor.shape[1]] = tensor
    else:
        padded = tensor
    return padded
