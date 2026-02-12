import pytest
from src.schedulers.lite import LiteScheduler
from src.schedulers.fixed import FixedScheduler


class TestFixedScheduler:
    def test_all_same_rank(self):
        s = FixedScheduler(r_min=2, r_max=16, rank=8)
        result = s.allocate(0, [0, 1, 2], [])
        for cid in [0, 1, 2]:
            assert result[cid]["rank"] == 8

    def test_clamps_to_bounds(self):
        s = FixedScheduler(r_min=2, r_max=16, rank=32)
        result = s.allocate(0, [0], [])
        assert result[0]["rank"] == 16


class TestLiteScheduler:
    def _make_telemetry(self, client_id, train_time, loss_before, loss_after, n_samples=100):
        return {
            "client_id": client_id,
            "train_time": train_time,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "n_samples": n_samples,
            "rank_used": 8,
            "exceeded_deadline": False,
        }

    def test_round_zero_uniform(self):
        s = LiteScheduler(r_min=2, r_max=16, r_bar=8)
        result = s.allocate(0, [0, 1, 2], [])
        for cid in [0, 1, 2]:
            assert result[cid]["rank"] == 8

    def test_respects_bounds(self):
        s = LiteScheduler(r_min=2, r_max=16, r_bar=8)
        tel = [
            self._make_telemetry(0, 1.0, 2.0, 0.1, 100),
            self._make_telemetry(1, 100.0, 2.0, 1.9, 100),
        ]
        result = s.allocate(1, [0, 1], tel)
        for cid in [0, 1]:
            assert s.r_min <= result[cid]["rank"] <= s.r_max

    def test_budget_constraint(self):
        s = LiteScheduler(r_min=2, r_max=16, r_bar=8)
        n = 6
        tel = [
            self._make_telemetry(i, float(i + 1), 2.0, 1.0, 100)
            for i in range(n)
        ]
        result = s.allocate(1, list(range(n)), tel)
        total = sum(r["rank"] for r in result.values())
        assert total == s.r_bar * n

    def test_deadline_penalty_reduces_rank(self):
        s = LiteScheduler(r_min=2, r_max=16, r_bar=8, deadline_penalty=0.1)
        tel_normal = [
            self._make_telemetry(0, 1.0, 2.0, 1.0, 100),
            self._make_telemetry(1, 1.0, 2.0, 1.0, 100),
        ]
        tel_exceeded = [
            {**self._make_telemetry(0, 1.0, 2.0, 1.0, 100), "exceeded_deadline": True},
            self._make_telemetry(1, 1.0, 2.0, 1.0, 100),
        ]
        r_normal = s.allocate(1, [0, 1], tel_normal)
        # reset EMA
        s._ema_throughput.clear()
        s._ema_gain.clear()
        r_penalty = s.allocate(1, [0, 1], tel_exceeded)
        # client 0 should get lower rank when it exceeded deadline
        assert r_penalty[0]["rank"] <= r_normal[0]["rank"]
