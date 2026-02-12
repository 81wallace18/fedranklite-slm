from .base import Scheduler
from .fixed import FixedScheduler
from .lite import LiteScheduler

SCHEDULERS = {
    "fixed": FixedScheduler,
    "lite": LiteScheduler,
}

try:
    from .cvx import CVXScheduler
    SCHEDULERS["cvx"] = CVXScheduler
except ImportError:
    pass


def build_scheduler(cfg: dict) -> Scheduler:
    stype = cfg["federation"]["scheduler"]["type"]
    if stype not in SCHEDULERS:
        raise ValueError(f"Scheduler '{stype}' not found. Available: {list(SCHEDULERS)}")
    params = cfg["federation"]["scheduler"].get(stype, {})
    return SCHEDULERS[stype](
        r_min=cfg["model"]["lora"]["r_min"],
        r_max=cfg["model"]["lora"]["r_max"],
        **params,
    )
