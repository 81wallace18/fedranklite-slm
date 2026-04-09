from __future__ import annotations

from abc import ABC, abstractmethod


class Scheduler(ABC):
    def __init__(self, r_min: int, r_max: int, **kwargs):
        self.r_min = r_min
        self.r_max = r_max

    @abstractmethod
    def allocate(
        self,
        round_id: int,
        client_ids: list[int],
        telemetry: list[dict],
        client_info: dict[int, dict] | None = None,
    ) -> dict[int, dict]:
        """Return {client_id: {"rank": int}} for each client in the round.

        Parameters
        ----------
        client_info : optional mapping ``{client_id: {"compute_factor": float, ...}}``
            Hardware profile for each client so the scheduler can estimate
            whether a given rank is feasible within the deadline.
        """
        ...
