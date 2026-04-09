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

    Minimizes max round time across clients by choosing ranks,
    subject to deadline feasibility and budget constraints.
    Falls back to greedy if cvxpy is not installed.
    """

    def __init__(
        self,
        r_min: int,
        r_max: int,
        r_bar: int = 8,
        solver: str = "ECOS",
        objective: str = "min_round_time",
        **kwargs,
    ):
        super().__init__(r_min, r_max)
        self.r_bar = r_bar
        self.solver = solver
        self.objective = objective

    def allocate(self, round_id, client_ids, telemetry, client_info=None):
        n = len(client_ids)

        if round_id == 0 or not telemetry:
            mid = (self.r_min + self.r_max) // 2
            return {cid: {"rank": mid} for cid in client_ids}

        # keep most recent entry per client
        tel_map: dict[int, dict] = {}
        for t in telemetry:
            if t["client_id"] in client_ids:
                tel_map[t["client_id"]] = t

        # estimate time per rank unit for each client
        time_per_rank = {}
        for cid in client_ids:
            t = tel_map.get(cid)
            if t and t.get("rank_used", 1) > 0 and t.get("train_time", 0) > 0:
                time_per_rank[cid] = t["train_time"] / t["rank_used"]
            else:
                time_per_rank[cid] = 1.0

        deadline = None
        if client_info:
            deadline = client_info.get("_deadline_seconds")

        if HAS_CVXPY:
            return self._solve_cvx(client_ids, time_per_rank, client_info, deadline)
        else:
            return self._solve_greedy(client_ids, time_per_rank, client_info, deadline)

    def _solve_cvx(self, client_ids, time_per_rank, client_info=None, deadline=None):
        n = len(client_ids)
        r = cp.Variable(n, integer=True)

        # cost vector: time_per_rank / compute_factor for each client
        costs = np.array([
            time_per_rank[cid] / (client_info.get(cid, {}).get("compute_factor", 1.0) if client_info else 1.0)
            for cid in client_ids
        ])

        objective = cp.Minimize(cp.max(cp.multiply(costs, r)))
        constraints = [
            r >= self.r_min,
            r <= self.r_max,
            cp.sum(r) <= len(client_ids) * self.r_bar,
        ]

        # deadline feasibility per client
        if deadline and deadline > 0:
            for i in range(n):
                constraints.append(costs[i] * r[i] <= deadline)

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=self.solver)
            if r.value is not None:
                ranks = np.clip(np.round(r.value).astype(int), self.r_min, self.r_max)
                return {cid: {"rank": int(ranks[i])} for i, cid in enumerate(client_ids)}
        except cp.SolverError:
            pass

        return self._solve_greedy(client_ids, time_per_rank, client_info, deadline)

    def _solve_greedy(self, client_ids, time_per_rank, client_info=None, deadline=None):
        """Greedy fallback: give smaller ranks to slower clients, respecting deadline."""
        sorted_clients = sorted(client_ids, key=lambda c: time_per_rank[c], reverse=True)
        n = len(client_ids)
        ranks = np.linspace(self.r_min, self.r_max, n).astype(int)
        assignments = {cid: {"rank": int(ranks[i])} for i, cid in enumerate(sorted_clients)}

        # enforce deadline viability
        if deadline and deadline > 0 and client_info:
            for cid in client_ids:
                cf = client_info.get(cid, {}).get("compute_factor", 1.0)
                tpr = time_per_rank[cid]
                r = assignments[cid]["rank"]
                est_time = tpr * r / cf
                while est_time > deadline and r > self.r_min:
                    r -= 1
                    est_time = tpr * r / cf
                assignments[cid]["rank"] = r

        return assignments
