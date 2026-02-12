#!/usr/bin/env python3
"""Entry point: python scripts/run.py --config configs/experiments/lite_noniid_deadline.yaml"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.engine import run


def main():
    parser = argparse.ArgumentParser(description="FedRankLite-SLM")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(cfg["logging"]["output_dir"]) / "run.log"),
        ],
    )

    tracker = run(cfg)

    print(f"\nExperiment: {cfg['experiment_name']}")
    print(f"Target round: {tracker.target_round}")
    print(f"Deadline compliance: {tracker.deadline_compliance():.2%}")
    print(f"Results saved to: {cfg['logging']['output_dir']}")


if __name__ == "__main__":
    main()
