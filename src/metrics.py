from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class RoundTelemetry:
    round_id: int
    clients: list[dict] = field(default_factory=list)
    global_eval_score: float | None = None
    global_eval_loss: float | None = None
    round_time: float = 0.0
    total_bytes: int = 0
    deadline_met: bool = True


@dataclass
class FairnessStats:
    loss_p10: float
    loss_p50: float
    loss_p90: float
    score_spread: float  # p90 - p10 of loss (higher = more unfair)


class MetricsTracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.history: list[RoundTelemetry] = []
        self.target_score = cfg["evaluation"]["target_score"]
        self.target_round: int | None = None
        self.output_dir = Path(cfg["logging"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_round(self, telemetry: RoundTelemetry):
        self.history.append(telemetry)

        # check time-to-target
        if (
            self.target_round is None
            and telemetry.global_eval_score is not None
            and telemetry.global_eval_score >= self.target_score
        ):
            self.target_round = telemetry.round_id
            logger.info(f"Target score {self.target_score} reached at round {self.target_round}")

        # fairness
        fairness = self.compute_fairness(telemetry)

        eval_str = f"eval={telemetry.global_eval_score:.4f} " if telemetry.global_eval_score else ""
        logger.info(
            f"Round {telemetry.round_id}: "
            f"{eval_str}"
            f"time={telemetry.round_time:.1f}s "
            f"bytes={telemetry.total_bytes} "
            f"deadline_met={telemetry.deadline_met}"
        )

        if self.cfg["metrics"]["track_fairness"] and fairness:
            logger.info(
                f"  fairness: loss_p10={fairness.loss_p10:.4f} "
                f"p50={fairness.loss_p50:.4f} p90={fairness.loss_p90:.4f}"
            )

    def compute_fairness(self, telemetry: RoundTelemetry) -> FairnessStats | None:
        if not self.cfg["metrics"]["track_fairness"] or not telemetry.clients:
            return None
        losses = [c["loss_after"] for c in telemetry.clients if "loss_after" in c]
        if len(losses) < 2:
            return None
        percentiles = self.cfg["metrics"]["percentiles"]
        vals = np.percentile(losses, percentiles)
        return FairnessStats(
            loss_p10=vals[0],
            loss_p50=vals[1],
            loss_p90=vals[2],
            score_spread=vals[2] - vals[0],
        )

    def deadline_compliance(self) -> float:
        if not self.history:
            return 1.0
        met = sum(1 for r in self.history if r.deadline_met)
        return met / len(self.history)

    def save(self):
        out = {
            "target_score": self.target_score,
            "target_round": self.target_round,
            "deadline_compliance": self.deadline_compliance(),
            "total_rounds": len(self.history),
            "rounds": [_telemetry_to_dict(t) for t in self.history],
        }
        path = self.output_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Metrics saved to {path}")


@torch.no_grad()
def evaluate_global(model, eval_dataloader: DataLoader, metric_name: str = "accuracy") -> dict:
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    all_labels = []
    total_loss = 0.0
    count = 0

    is_regression = metric_name == "pearson"

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item() * batch["input_ids"].shape[0]
        count += batch["input_ids"].shape[0]
        if is_regression:
            preds = outputs.logits.squeeze(-1)
        else:
            preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

    avg_loss = total_loss / max(count, 1)

    if metric_name == "accuracy":
        score = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_preds), 1)
    elif metric_name == "f1":
        score = _f1_binary(all_preds, all_labels)
    elif metric_name == "pearson":
        score = pearsonr(all_preds, all_labels).statistic if len(all_preds) > 1 else 0.0
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    return {"score": score, "loss": avg_loss}


def _f1_binary(preds: list, labels: list) -> float:
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _telemetry_to_dict(t: RoundTelemetry) -> dict:
    return {
        "round_id": t.round_id,
        "global_eval_score": t.global_eval_score,
        "global_eval_loss": t.global_eval_loss,
        "round_time": t.round_time,
        "total_bytes": t.total_bytes,
        "deadline_met": t.deadline_met,
        "clients": t.clients,
    }
