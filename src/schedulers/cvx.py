from __future__ import annotations

import numpy as np

from .base import Scheduler

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


class CVXScheduler(Scheduler):
    """Baseline scheduler approximating FAH-QLoRA's P1 optimization.

    Minimizes max round time across clients by choosing ranks.
    Falls back to greedy if cvxpy is not installed.
    """

    def __init__(
        self,
        r_min: int,
        r_max: int,
        solver: str = "ECOS",
        objective: str = "min_round_time",
        **kwargs,
    ):
        super().__init__(r_min, r_max)
        self.solver = solver
        self.objective = objective

    def allocate(self, round_id, client_ids, telemetry):
        n = len(client_ids)

        if round_id == 0 or not telemetry:
            mid = (self.r_min + self.r_max) // 2
            return {cid: {"rank": mid} for cid in client_ids}

        tel_map = {t["client_id"]: t for t in telemetry if t["client_id"] in client_ids}

        # estimate time per rank unit for each client
        time_per_rank = {}
        for cid in client_ids:
            t = tel_map.get(cid)
            if t and t.get("rank_used", 1) > 0 and t.get("train_time", 0) > 0:
                time_per_rank[cid] = t["train_time"] / t["rank_used"]
            else:
                time_per_rank[cid] = 1.0

        if HAS_CVXPY:
            return self._solve_cvx(client_ids, time_per_rank)
        else:
            return self._solve_greedy(client_ids, time_per_rank)

    def _solve_cvx(self, client_ids, time_per_rank):
        n = len(client_ids)
        r = cp.Variable(n, integer=True)
        costs = np.array([time_per_rank[cid] for cid in client_ids])

        objective = cp.Minimize(cp.max(cp.multiply(costs, r)))
        constraints = [
            r >= self.r_min,
            r <= self.r_max,
        ]
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=self.solver)
            if r.value is not None:
                ranks = np.clip(np.round(r.value).astype(int), self.r_min, self.r_max)
                return {cid: {"rank": int(ranks[i])} for i, cid in enumerate(client_ids)}
        except cp.SolverError:
            pass

        return self._solve_greedy(client_ids, time_per_rank)

    def _solve_greedy(self, client_ids, time_per_rank):
        """Greedy fallback: give smaller ranks to slower clients."""
        sorted_clients = sorted(client_ids, key=lambda c: time_per_rank[c], reverse=True)
        n = len(client_ids)
        ranks = np.linspace(self.r_min, self.r_max, n).astype(int)
        return {cid: {"rank": int(ranks[i])} for i, cid in enumerate(sorted_clients)}
