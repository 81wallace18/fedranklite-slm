from __future__ import annotations

import numpy as np

from .base import Scheduler


class LiteScheduler(Scheduler):
    """Lightweight online scheduler — O(N) per round, no solver needed.

    Score per client = gain_weight * normalized_gain + throughput_weight * normalized_throughput
    Rank allocated proportional to score, constrained to [r_min, r_max] and budget r_bar * N.
    EMA smoothing on throughput and gain to avoid oscillation.
    Deadline penalty: if a client exceeded deadline last round, its rank is reduced.
    Deadline viability: after allocation, ranks are capped so estimated time fits the deadline.
    """

    def __init__(
        self,
        r_min: int,
        r_max: int,
        r_bar: int = 8,
        ema_alpha: float = 0.3,
        gain_weight: float = 0.6,
        throughput_weight: float = 0.4,
        deadline_penalty: float = 0.5,
        **kwargs,
    ):
        super().__init__(r_min, r_max)
        self.r_bar = r_bar
        self.ema_alpha = ema_alpha
        self.gain_weight = gain_weight
        self.throughput_weight = throughput_weight
        self.deadline_penalty = deadline_penalty

        # EMA state per client
        self._ema_throughput: dict[int, float] = {}
        self._ema_gain: dict[int, float] = {}
        self._ema_time_per_rank: dict[int, float] = {}

    def allocate(self, round_id, client_ids, telemetry, client_info=None):
        n = len(client_ids)
        budget = self.r_bar * n

        # round 0 or no telemetry: uniform r_bar
        if round_id == 0 or not telemetry:
            return {cid: {"rank": self.r_bar} for cid in client_ids}

        # build lookup: keep most recent entry per client
        tel_map: dict[int, dict] = {}
        for t in telemetry:
            if t["client_id"] in client_ids:
                tel_map[t["client_id"]] = t

        # update EMA and compute scores
        scores = {}
        for cid in client_ids:
            t = tel_map.get(cid)
            if t is None:
                scores[cid] = 1.0
                continue

            throughput = t.get("n_samples", 0) / max(t.get("train_time", 1.0), 1e-6)
            gain = max(t.get("loss_before", 0) - t.get("loss_after", 0), 0.0)
            tpr = t.get("train_time", 1.0) / max(t.get("rank_used", 1), 1)

            # EMA update
            if cid in self._ema_throughput:
                self._ema_throughput[cid] = (
                    self.ema_alpha * throughput + (1 - self.ema_alpha) * self._ema_throughput[cid]
                )
                self._ema_gain[cid] = (
                    self.ema_alpha * gain + (1 - self.ema_alpha) * self._ema_gain[cid]
                )
                self._ema_time_per_rank[cid] = (
                    self.ema_alpha * tpr + (1 - self.ema_alpha) * self._ema_time_per_rank[cid]
                )
            else:
                self._ema_throughput[cid] = throughput
                self._ema_gain[cid] = gain
                self._ema_time_per_rank[cid] = tpr

            score = (
                self.gain_weight * self._ema_gain[cid]
                + self.throughput_weight * self._ema_throughput[cid]
            )

            # deadline penalty
            if t.get("exceeded_deadline", False):
                score *= self.deadline_penalty

            scores[cid] = max(score, 1e-8)

        # normalize scores and distribute budget
        total_score = sum(scores.values())
        raw_ranks = {cid: (scores[cid] / total_score) * budget for cid in client_ids}

        # clamp and round
        assignments = {}
        for cid in client_ids:
            r = int(np.clip(round(raw_ranks[cid]), self.r_min, self.r_max))
            assignments[cid] = {"rank": r}

        # adjust to meet budget exactly (greedy correction)
        current_total = sum(a["rank"] for a in assignments.values())
        diff = budget - current_total
        sorted_clients = sorted(client_ids, key=lambda c: scores[c], reverse=(diff > 0))

        for cid in sorted_clients:
            if diff == 0:
                break
            r = assignments[cid]["rank"]
            if diff > 0 and r < self.r_max:
                step = min(diff, self.r_max - r)
                assignments[cid]["rank"] = r + step
                diff -= step
            elif diff < 0 and r > self.r_min:
                step = min(-diff, r - self.r_min)
                assignments[cid]["rank"] = r - step
                diff += step

        # --- Deadline viability: cap ranks so estimated time fits deadline ---
        if client_info:
            deadline = client_info.get("_deadline_seconds")
            if deadline and deadline > 0:
                for cid in client_ids:
                    cf = client_info.get(cid, {}).get("compute_factor", 1.0)
                    tpr = self._ema_time_per_rank.get(cid)
                    if tpr is None or tpr <= 0:
                        continue
                    r = assignments[cid]["rank"]
                    est_time = tpr * r / cf
                    while est_time > deadline and r > self.r_min:
                        r -= 1
                        est_time = tpr * r / cf
                    assignments[cid]["rank"] = r

        return assignments
