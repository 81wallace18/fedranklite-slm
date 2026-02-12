# FedRankLite-SLM

Federated fine-tuning of Small Language Models with lightweight, adaptive LoRA rank scheduling.

## Overview

This project implements a config-driven experiment harness for federated learning with parameter-efficient fine-tuning (PEFT). The core contribution is a **lightweight online scheduler** that allocates heterogeneous LoRA ranks per client per round, replacing solver-based approaches (CVX) with an O(N) heuristic — making federated SLM training practical under hardware heterogeneity, non-IID data, and deadline constraints.

### Key ideas

- **Pluggable schedulers**: swap between CVX baseline, fixed rank (ablation), and our Lite scheduler via a single YAML field
- **Rank masking**: train with `r_max` LoRA, activate only `r_i` dimensions per client — no module reconstruction needed
- **Heterogeneous aggregation**: aggregate LoRA updates from clients with different ranks using zero-padding or blockwise strategies
- **Tier simulation**: model GPU/CPU speed differences with configurable `compute_factor` per hardware tier
- **Deadline enforcement**: drop, wait, or use partial updates from stragglers

## Quick start

```bash
# install
pip install -e ".[dev]"

# run an experiment
python scripts/run.py --config configs/experiments/lite_noniid_deadline.yaml

# run tests
pytest tests/ -v
```

## Project structure

```
configs/
  base.yaml                          # all defaults (model, training, tiers, metrics)
  experiments/
    glue/                             # GLUE benchmark: 5 tasks x 2 schedulers
      sst2_cvx.yaml, sst2_lite.yaml
      qnli_cvx.yaml, qnli_lite.yaml
      mrpc_cvx.yaml, mrpc_lite.yaml
      stsb_cvx.yaml, stsb_lite.yaml
      rte_cvx.yaml,  rte_lite.yaml
    baseline_cvx.yaml                 # baseline scheduler
    baseline_fixed.yaml               # fixed rank (ablation)
    lite_iid.yaml                     # Lite + IID
    lite_noniid_deadline.yaml         # Lite + non-IID + deadline
src/
  config.py          # YAML loader with inheritance (_base_) and deep merge
  models.py          # backbone + quantization + LoRA + rank mask/truncate/pad
  data.py            # HuggingFace datasets + Dirichlet/IID/label_skew partitioning
  client.py          # local training with telemetry (loss, time, memory, bytes)
  aggregation.py     # FedAvg and blockwise aggregation for heterogeneous ranks
  engine.py          # federated loop with tier simulation and deadline handling
  metrics.py         # time-to-target, deadline compliance, fairness (p10/p50/p90)
  schedulers/
    base.py          # abstract interface
    fixed.py         # constant rank for all clients
    cvx.py           # CVX solver / greedy fallback (baseline)
    lite.py          # our lightweight scheduler
scripts/
  run.py             # CLI entry point
  run_slurm.sh       # Slurm wrapper
tests/               # unit tests for aggregation, schedulers, masking, metrics, data
```

## Configuration

Everything is controlled by YAML. Experiment configs inherit from `base.yaml` via `_base_` and override only what changes.

```yaml
# configs/experiments/lite_noniid_deadline.yaml
_base_: "../base.yaml"
experiment_name: "lite_noniid_hard"
data:
  partition:
    method: "dirichlet"
    alpha: 0.1
federation:
  scheduler:
    type: "lite"
deadline:
  seconds: 90
```

### Main config sections

| Section | What it controls |
|---|---|
| `model` | HF model name, quantization (4bit/8bit/none), LoRA params (r_max, r_min, alpha, target_modules) |
| `federation` | num_rounds, clients_per_round, aggregation method, scheduler type and params |
| `training` | local_epochs, batch_size, lr, optimizer, grad clipping |
| `data` | dataset, task_type, num_labels, text_columns, partition method and alpha |
| `tiers` | hardware profiles with compute_factor and memory |
| `deadline` | enabled, seconds, straggler_policy (drop/wait/use_partial) |
| `evaluation` | target_score, metric (accuracy/f1/pearson), eval frequency |
| `metrics` | fairness tracking, percentiles, comm bytes, peak memory |

## Schedulers

| Scheduler | Config | Description |
|---|---|---|
| `fixed` | `type: "fixed"` | Same rank for all clients. Ablation baseline. |
| `cvx` | `type: "cvx"` | Minimizes max round time via CVX or greedy fallback. Paper baseline. |
| `lite` | `type: "lite"` | **Ours.** Scores clients by EMA of loss drop and throughput, distributes rank budget proportionally, penalizes deadline violations. O(N) per round. |

### Lite scheduler formula

```
score_i = gain_weight * EMA(loss_drop_i) + throughput_weight * EMA(throughput_i)
if exceeded_deadline: score_i *= deadline_penalty

rank_i = clamp(round((score_i / total_score) * budget), r_min, r_max)
```

Budget = `r_bar * num_selected_clients`. Greedy correction ensures exact budget match.

## GLUE benchmark

Reproduces the baseline paper's experimental setup with 5 GLUE tasks:

| Task | Clients | Metric | Text format |
|---|---|---|---|
| SST-2 | 12 | Accuracy | Single sentence |
| QNLI | 12 | Accuracy | Sentence pair |
| MRPC | 3 | F1 | Sentence pair |
| STS-B | 3 | Pearson | Sentence pair (regression) |
| RTE | 3 | Accuracy | Sentence pair |

```bash
# run all GLUE tasks (example)
for cfg in configs/experiments/glue/*_lite.yaml; do
  python scripts/run.py --config "$cfg"
done
```

## Hardware tiers

Default configuration simulates three hardware profiles:

| Tier | Real hardware | Speed factor | Clients |
|---|---|---|---|
| `gpu_fast` | GPU 24 GB | 1.0x | 3 |
| `gpu_medium` | GPU 32 GB | 0.7x | 3 |
| `cpu_slow` | CPU 256 cores | 0.15x | 4 |

The engine divides measured training time by `compute_factor` to simulate speed differences.

## Metrics collected

- **Time-to-target**: round at which eval score reaches `target_score`
- **Deadline compliance**: percentage of rounds without straggler timeout
- **Fairness**: p10/p50/p90 of per-client loss distribution
- **Communication**: bytes transmitted per client per round
- **Peak memory**: GPU (CUDA) or CPU (psutil) per client

Results are saved to `results/<experiment_name>/metrics.json`.

## Running on Slurm

```bash
sbatch scripts/run_slurm.sh configs/experiments/glue/sst2_lite.yaml
```

## Dependencies

- Python >= 3.10
- PyTorch >= 2.1
- Transformers >= 4.40
- PEFT >= 0.10
- bitsandbytes >= 0.43
- datasets >= 2.19

Optional: `cvxpy` (for CVX scheduler), `wandb` (for experiment tracking).

## References

- **FAH-QLoRA**: baseline paper — federated adaptive heterogeneous QLoRA with CVX-based rank allocation
- **PEFT/LoRA**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- **QLoRA**: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)
