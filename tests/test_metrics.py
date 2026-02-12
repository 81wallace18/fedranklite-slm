import pytest
from src.metrics import MetricsTracker, RoundTelemetry


@pytest.fixture
def cfg():
    return {
        "evaluation": {"target_score": 0.90, "metric": "accuracy", "eval_every": 5},
        "metrics": {
            "track_fairness": True,
            "percentiles": [10, 50, 90],
            "track_comm_bytes": True,
            "track_peak_memory": True,
        },
        "logging": {"output_dir": "/tmp/fedranklite_test", "save_every": 10, "log_telemetry": True},
    }


class TestTimeToTarget:
    def test_target_reached(self, cfg):
        tracker = MetricsTracker(cfg)
        t1 = RoundTelemetry(round_id=0, global_eval_score=0.80)
        t2 = RoundTelemetry(round_id=5, global_eval_score=0.92)
        tracker.log_round(t1)
        tracker.log_round(t2)
        assert tracker.target_round == 5

    def test_target_not_reached(self, cfg):
        tracker = MetricsTracker(cfg)
        t1 = RoundTelemetry(round_id=0, global_eval_score=0.50)
        tracker.log_round(t1)
        assert tracker.target_round is None


class TestDeadlineCompliance:
    def test_all_met(self, cfg):
        tracker = MetricsTracker(cfg)
        for i in range(10):
            tracker.log_round(RoundTelemetry(round_id=i, deadline_met=True))
        assert tracker.deadline_compliance() == 1.0

    def test_half_met(self, cfg):
        tracker = MetricsTracker(cfg)
        for i in range(10):
            tracker.log_round(RoundTelemetry(round_id=i, deadline_met=(i % 2 == 0)))
        assert tracker.deadline_compliance() == 0.5


class TestFairness:
    def test_computes_percentiles(self, cfg):
        tracker = MetricsTracker(cfg)
        clients = [{"loss_after": float(i)} for i in range(10)]
        t = RoundTelemetry(round_id=0, clients=clients)
        fairness = tracker.compute_fairness(t)
        assert fairness is not None
        assert fairness.loss_p10 < fairness.loss_p50 < fairness.loss_p90
        assert fairness.score_spread == fairness.loss_p90 - fairness.loss_p10

    def test_skips_when_disabled(self, cfg):
        cfg["metrics"]["track_fairness"] = False
        tracker = MetricsTracker(cfg)
        t = RoundTelemetry(round_id=0, clients=[{"loss_after": 1.0}])
        assert tracker.compute_fairness(t) is None
