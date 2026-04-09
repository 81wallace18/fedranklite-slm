#!/usr/bin/env python3
"""Entry point: python scripts/run.py --config configs/experiments/lite_noniid_deadline.yaml"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*use_return_dict.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.engine import run


def main():
    parser = argparse.ArgumentParser(description="FedRankLite-SLM")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Create output directory if it doesn't exist
    out_dir = Path(cfg["logging"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "run.log"),
        ],
    )

    tracker = run(cfg)

    print(f"\nExperiment: {cfg['experiment_name']}")
    print(f"Target round: {tracker.target_round}")
    print(f"Deadline compliance: {tracker.deadline_compliance():.2%}")
    print(f"Results saved to: {cfg['logging']['output_dir']}")


if __name__ == "__main__":
    main()
