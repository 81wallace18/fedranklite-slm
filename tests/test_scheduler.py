import pytest
from src.schedulers.lite import LiteScheduler
from src.schedulers.fixed import FixedScheduler

try:
    from src.schedulers.cvx import CVXScheduler
    HAS_CVX = True
except ImportError:
    HAS_CVX = False


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
    def _make_telemetry(self, client_id, train_time, loss_before, loss_after, n_samples=100, rank_used=8):
        return {
            "client_id": client_id,
            "train_time": train_time,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "n_samples": n_samples,
            "rank_used": rank_used,
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
        s._ema_time_per_rank.clear()
        r_penalty = s.allocate(1, [0, 1], tel_exceeded)
        # client 0 should get lower rank when it exceeded deadline
        assert r_penalty[0]["rank"] <= r_normal[0]["rank"]

    def test_deadline_viability_caps_rank_for_slow_clients(self):
        """Slow clients (low compute_factor) should get lower ranks to meet deadline."""
        s = LiteScheduler(r_min=2, r_max=16, r_bar=8)

        # Simulate round 0 to initialize EMA
        s.allocate(0, [0, 1], [])

        # telemetry: client 0 fast, client 1 slow (high train_time from compute_factor)
        tel = [
            self._make_telemetry(0, 5.0, 2.0, 1.0, 100, rank_used=8),   # 5s for rank 8 → 0.625s/rank
            self._make_telemetry(1, 80.0, 2.0, 1.0, 100, rank_used=8),  # 80s for rank 8 → 10s/rank
        ]

        client_info = {
            0: {"compute_factor": 1.0},
            1: {"compute_factor": 0.15},  # slow client
            "_deadline_seconds": 120,
        }

        result = s.allocate(1, [0, 1], tel, client_info=client_info)

        # Client 1 is very slow: tpr=10, cf=0.15 → est_time = 10*r/0.15
        # For r=2: 10*2/0.15 = 133s > 120 → still exceeds, but r_min reached
        # The key is that client 1 should get r_min (2)
        assert result[1]["rank"] == 2

    def test_no_viability_without_client_info(self):
        """Without client_info, deadline viability check is skipped."""
        s = LiteScheduler(r_min=2, r_max=16, r_bar=8)
        tel = [
            self._make_telemetry(0, 5.0, 2.0, 1.0, 100),
            self._make_telemetry(1, 5.0, 2.0, 1.0, 100),
        ]
        result = s.allocate(1, [0, 1], tel)
        # Both should get r_bar without viability check
        total = sum(r["rank"] for r in result.values())
        assert total == 16  # r_bar * 2


@pytest.mark.skipif(not HAS_CVX, reason="cvxpy not installed")
class TestCVXScheduler:
    def _make_telemetry(self, client_id, train_time, rank_used=8):
        return {
            "client_id": client_id,
            "train_time": train_time,
            "rank_used": rank_used,
            "loss_before": 2.0,
            "loss_after": 1.0,
            "n_samples": 100,
            "exceeded_deadline": False,
        }

    def test_round_zero_gives_mid_rank(self):
        s = CVXScheduler(r_min=2, r_max=16)
        result = s.allocate(0, [0, 1, 2], [])
        for cid in [0, 1, 2]:
            assert result[cid]["rank"] == 9  # (2+16)//2

    def test_respects_bounds(self):
        s = CVXScheduler(r_min=2, r_max=16)
        tel = [
            self._make_telemetry(0, 1.0),
            self._make_telemetry(1, 100.0),
        ]
        result = s.allocate(1, [0, 1], tel)
        for cid in [0, 1]:
            assert 2 <= result[cid]["rank"] <= 16

    def test_slower_clients_get_lower_rank(self):
        s = CVXScheduler(r_min=2, r_max=16)
        tel = [
            self._make_telemetry(0, 1.0),   # fast
            self._make_telemetry(1, 100.0),  # slow
        ]
        result = s.allocate(1, [0, 1], tel)
        assert result[1]["rank"] <= result[0]["rank"]

    def test_deadline_constraint_caps_slow_client(self):
        """With deadline + client_info, slow clients get feasible ranks."""
        s = CVXScheduler(r_min=2, r_max=16, r_bar=8)
        tel = [
            self._make_telemetry(0, 8.0, rank_used=8),   # 1s/rank
            self._make_telemetry(1, 80.0, rank_used=8),   # 10s/rank
        ]
        client_info = {
            0: {"compute_factor": 1.0},
            1: {"compute_factor": 0.15},
            "_deadline_seconds": 120,
        }
        result = s.allocate(1, [0, 1], tel, client_info=client_info)
        # Client 1: tpr=10, cf=0.15 → effective cost = 10/0.15 = 66.7s/rank
        # deadline=120 → max rank = 120/66.7 ≈ 1.8 → r_min=2
        assert result[1]["rank"] == 2
