import torch
import pytest
from src.aggregation import aggregate_lora
from src.client import ClientResult


def _make_result(client_id, rank, n_samples=100, loss_b=1.0, loss_a=0.5):
    """Helper: create a ClientResult with synthetic LoRA deltas."""
    delta = {
        "layer.lora_A": torch.randn(rank, 64),
        "layer.lora_B": torch.randn(128, rank),
    }
    return ClientResult(
        client_id=client_id,
        rank_used=rank,
        lora_delta=delta,
        n_samples=n_samples,
        loss_before=loss_b,
        loss_after=loss_a,
        train_time=1.0,
        peak_memory_mb=100.0,
        bytes_sent=1024,
    )


def _make_global(r_max=16):
    return {
        "layer.lora_A": torch.zeros(r_max, 64),
        "layer.lora_B": torch.zeros(128, r_max),
    }


class TestFedAvg:
    def test_same_rank(self):
        global_lora = _make_global(8)
        r1 = _make_result(0, rank=8)
        r2 = _make_result(1, rank=8)
        agg = aggregate_lora(global_lora, [r1, r2], r_max=8, method="fedavg")
        assert agg["layer.lora_A"].shape == (8, 64)
        assert agg["layer.lora_B"].shape == (128, 8)

    def test_different_ranks_pads_correctly(self):
        r_max = 16
        global_lora = _make_global(r_max)
        r1 = _make_result(0, rank=4)
        r2 = _make_result(1, rank=16)
        agg = aggregate_lora(global_lora, [r1, r2], r_max=r_max, method="fedavg")
        assert agg["layer.lora_A"].shape == (r_max, 64)
        assert agg["layer.lora_B"].shape == (128, r_max)

    def test_uniform_weighting(self):
        global_lora = _make_global(8)
        r1 = _make_result(0, rank=8, n_samples=10)
        r2 = _make_result(1, rank=8, n_samples=1000)
        agg_uniform = aggregate_lora(global_lora, [r1, r2], r_max=8, weighting="uniform")
        agg_samples = aggregate_lora(global_lora, [r1, r2], r_max=8, weighting="num_samples")
        # they should differ because weighting is different
        assert not torch.allclose(agg_uniform["layer.lora_A"], agg_samples["layer.lora_A"])

    def test_empty_results_returns_global(self):
        global_lora = _make_global(8)
        agg = aggregate_lora(global_lora, [], r_max=8)
        for k in global_lora:
            assert torch.equal(agg[k], global_lora[k])


class TestBlockwise:
    def test_only_high_rank_contributes_to_upper_dims(self):
        r_max = 8
        global_lora = _make_global(r_max)
        r1 = _make_result(0, rank=2, n_samples=100)
        r2 = _make_result(1, rank=8, n_samples=100)
        agg = aggregate_lora(global_lora, [r1, r2], r_max=r_max, method="fedavg_blockwise")
        # dims 0-1 of lora_A: both contributed
        # dims 2-7 of lora_A: only r2 contributed
        assert agg["layer.lora_A"].shape == (r_max, 64)
