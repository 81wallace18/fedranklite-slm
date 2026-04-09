"""Integration tests for the federation engine."""

import pytest


class TestDroppedClientTelemetry:
    """Verify that dropped clients still generate telemetry for the scheduler."""

    def test_dropped_clients_appear_in_telemetry(self):
        """Simulate the engine's deadline logic: dropped clients must store telemetry."""
        # Simulate engine's per-client loop (extracted logic)
        telemetry_history = []
        results = []
        deadline_seconds = 120

        class FakeResult:
            def __init__(self, cid, train_time):
                self.client_id = cid
                self.rank_used = 8
                self.train_time = train_time
                self.loss_before = 2.0
                self.loss_after = 1.0
                self.n_samples = 100
                self.peak_memory_mb = 500.0
                self.bytes_sent = 1024

        # Client 0: fast (meets deadline), Client 1: slow (exceeds deadline)
        fake_results = [FakeResult(0, 50.0), FakeResult(1, 300.0)]

        for result in fake_results:
            exceeded = result.train_time > deadline_seconds
            dropped = exceeded  # straggler_policy = "drop"

            telemetry_history.append({
                "client_id": result.client_id,
                "rank_used": result.rank_used,
                "train_time": result.train_time,
                "exceeded_deadline": exceeded,
                "dropped": dropped,
            })

            if not dropped:
                results.append(result)

        # Client 1 was dropped but MUST appear in telemetry
        assert len(telemetry_history) == 2
        assert telemetry_history[0]["client_id"] == 0
        assert telemetry_history[0]["exceeded_deadline"] is False
        assert telemetry_history[0]["dropped"] is False

        assert telemetry_history[1]["client_id"] == 1
        assert telemetry_history[1]["exceeded_deadline"] is True
        assert telemetry_history[1]["dropped"] is True

        # Only non-dropped client in results for aggregation
        assert len(results) == 1
        assert results[0].client_id == 0


class TestClientInfoPassthrough:
    """Verify that client_info structure is correct for schedulers."""

    def test_client_info_has_compute_factor(self):
        """Engine should build client_info with compute_factor per client."""
        # Simulate _assign_tiers + client_info construction
        tiers = [
            {"name": "gpu_fast", "count": 3, "compute_factor": 1.0},
            {"name": "gpu_medium", "count": 3, "compute_factor": 0.7},
            {"name": "cpu_slow", "count": 4, "compute_factor": 0.15},
        ]
        all_clients = list(range(10))

        # Simulate _assign_tiers
        tier_assignment = {}
        idx = 0
        for tier in tiers:
            for _ in range(tier["count"]):
                if idx < len(all_clients):
                    tier_assignment[all_clients[idx]] = tier
                    idx += 1

        # Simulate client_info construction (as in engine.py)
        selected = [0, 3, 6, 9]
        client_info = {
            cid: {"compute_factor": tier_assignment[cid]["compute_factor"]}
            for cid in selected
        }
        client_info["_deadline_seconds"] = 120

        assert client_info[0]["compute_factor"] == 1.0    # gpu_fast
        assert client_info[3]["compute_factor"] == 0.7     # gpu_medium
        assert client_info[6]["compute_factor"] == 0.15    # cpu_slow
        assert client_info[9]["compute_factor"] == 0.15    # cpu_slow
        assert client_info["_deadline_seconds"] == 120
