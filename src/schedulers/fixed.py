from __future__ import annotations

from .base import Scheduler


class FixedScheduler(Scheduler):
    def __init__(self, r_min: int, r_max: int, rank: int = 8, **kwargs):
        super().__init__(r_min, r_max)
        self.rank = max(r_min, min(rank, r_max))

    def allocate(self, round_id, client_ids, telemetry):
        return {cid: {"rank": self.rank} for cid in client_ids}
