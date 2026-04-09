from __future__ import annotations

import logging
import time

import numpy as np
import torch

from .aggregation import aggregate_lora
from .client import ClientResult, train_client
from .data import load_and_partition, make_dataloader
from .metrics import MetricsTracker, RoundTelemetry, evaluate_global
from .models import get_lora_state, load_model, set_lora_state
from .schedulers import build_scheduler

logger = logging.getLogger(__name__)


def _format_time(seconds: float) -> str:
    """Formata segundos em formato legivel (HH:MM:SS)."""
    if seconds < 0:
        return "--:--:--"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes:02d}m {secs:02d}s"
    else:
        return f"{secs:02d}s"


def run(cfg: dict):
    seed = cfg["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    fed_cfg = cfg["federation"]
    r_max = cfg["model"]["lora"]["r_max"]
    total_rounds = fed_cfg["num_rounds"]

    # --- load model + tokenizer ---
    num_labels = cfg["data"]["num_labels"]
    model, tokenizer = load_model(cfg, num_labels=num_labels)

    # --- load & partition data ---
    client_datasets, eval_ds = load_and_partition(cfg, tokenizer)
    eval_loader = make_dataloader(eval_ds, cfg["training"]["batch_size"], shuffle=False) if eval_ds else None

    # --- init ---
    global_lora = get_lora_state(model)
    scheduler = build_scheduler(cfg)
    tracker = MetricsTracker(cfg)
    rng = np.random.default_rng(seed)

    all_client_ids = list(range(fed_cfg["total_clients"]))

    # assign tiers to clients
    tier_assignment = _assign_tiers(all_client_ids, cfg["tiers"])

    telemetry_history: list[dict] = []
    round_times: list[float] = []
    training_start = time.time()

    print(f"\n{'=' * 60}")
    print(f"  Starting federation: {total_rounds} rounds, {fed_cfg['total_clients']} clients")
    print(f"  Scheduler: {fed_cfg['scheduler']['type']} | Deadline: {cfg['deadline']['seconds']}s")
    print(f"  Model: {cfg['model']['name']} | LoRA r_max={r_max}")
    print(f"{'=' * 60}\n")

    for round_id in range(total_rounds):
        t_round_start = time.time()

        # ETA calculation
        if round_times:
            avg_round_time = sum(round_times) / len(round_times)
            eta_seconds = avg_round_time * (total_rounds - round_id)
            eta_str = _format_time(eta_seconds)
        else:
            eta_str = "calculating..."

        # select clients for this round
        selected = rng.choice(
            all_client_ids,
            size=min(fed_cfg["clients_per_round"], len(all_client_ids)),
            replace=False,
        ).tolist()

        print(f"\n{'=' * 60}")
        print(f"  Round {round_id + 1}/{total_rounds} | ETA: {eta_str}")
        print(f"  Selected clients: {selected}")
        print(f"{'=' * 60}")

        # build client_info with compute_factor + deadline for scheduler
        client_info = {
            cid: {"compute_factor": tier_assignment[cid]["compute_factor"]}
            for cid in selected
        }
        if cfg["deadline"]["enabled"]:
            client_info["_deadline_seconds"] = cfg["deadline"]["seconds"]

        # scheduler decides ranks
        assignments = scheduler.allocate(round_id, selected, telemetry_history, client_info)

        # train each client
        results: list[ClientResult] = []
        round_exceeded_deadline = False

        for i, cid in enumerate(selected):
            rank = assignments[cid]["rank"]
            tier = tier_assignment[cid]

            print(f"  Training client {cid} ({i+1}/{len(selected)}) | rank={rank}, tier={tier['name']}, cf={tier['compute_factor']}")

            loader = make_dataloader(client_datasets[cid], cfg["training"]["batch_size"])

            result = train_client(
                client_id=cid,
                model=model,
                dataloader=loader,
                global_lora=global_lora,
                assigned_rank=rank,
                cfg=cfg,
            )

            # simulate tier speed factor
            real_time = result.train_time
            result.train_time /= tier["compute_factor"]

            print(
                f"    -> Client {cid} done in {_format_time(real_time)} (simulated: {_format_time(result.train_time)}) | "
                f"loss: {result.loss_before:.4f} -> {result.loss_after:.4f} | mem: {result.peak_memory_mb:.0f}MB"
            )

            # check deadline
            exceeded = False
            dropped = False
            if cfg["deadline"]["enabled"] and result.train_time > cfg["deadline"]["seconds"]:
                exceeded = True
                round_exceeded_deadline = True
                if cfg["deadline"]["straggler_policy"] == "drop":
                    dropped = True
                    print(f"    ** DROPPED (deadline {cfg['deadline']['seconds']}s exceeded: {result.train_time:.1f}s)")

            # always store telemetry (even for dropped clients — scheduler needs feedback)
            telemetry_history.append({
                "client_id": cid,
                "round_id": round_id,
                "rank_used": result.rank_used,
                "n_samples": result.n_samples,
                "loss_before": result.loss_before,
                "loss_after": result.loss_after,
                "train_time": result.train_time,
                "peak_memory_mb": result.peak_memory_mb,
                "bytes_sent": result.bytes_sent,
                "exceeded_deadline": exceeded,
                "dropped": dropped,
            })

            if dropped:
                continue

            results.append(result)

        # aggregate
        print("  Aggregating client models...")
        if results:
            agg_cfg = fed_cfg["aggregation"]
            global_lora = aggregate_lora(
                global_lora, results, r_max,
                method=agg_cfg["method"],
                weighting=agg_cfg["weighting"],
            )
            set_lora_state(model, global_lora)
        else:
            print("  Warning: No successful client updates. Skipping aggregation.")

        round_time = time.time() - t_round_start
        round_times.append(round_time)
        total_elapsed = time.time() - training_start
        n_dropped = len(selected) - len(results)

        # evaluate
        eval_score = None
        eval_loss = None
        eval_str = ""
        if eval_loader and (round_id + 1) % cfg["evaluation"]["eval_every"] == 0:
            eval_result = evaluate_global(model, eval_loader, cfg["evaluation"]["metric"])
            eval_score = eval_result["score"]
            eval_loss = eval_result["loss"]
            eval_str = f" | Eval: {eval_score:.4f}"

        print(
            f"  Round {round_id + 1} completed in {_format_time(round_time)} | "
            f"Total: {_format_time(total_elapsed)} | "
            f"{len(results)} active, {n_dropped} dropped{eval_str}"
        )

        if eval_score is not None and tracker.target_round is None and eval_score >= cfg["evaluation"]["target_score"]:
            print(f"  >>> TARGET SCORE {cfg['evaluation']['target_score']} REACHED at round {round_id + 1}! (score={eval_score:.4f})")

        # log round
        client_dicts = [
            {
                "client_id": r.client_id,
                "rank_used": r.rank_used,
                "loss_before": r.loss_before,
                "loss_after": r.loss_after,
                "train_time": r.train_time,
                "peak_memory_mb": r.peak_memory_mb,
                "bytes_sent": r.bytes_sent,
            }
            for r in results
        ]
        telemetry = RoundTelemetry(
            round_id=round_id,
            clients=client_dicts,
            global_eval_score=eval_score,
            global_eval_loss=eval_loss,
            round_time=round_time,
            total_bytes=sum(r.bytes_sent for r in results),
            deadline_met=not round_exceeded_deadline,
        )
        tracker.log_round(telemetry)

        # checkpoint
        if (round_id + 1) % cfg["logging"]["save_every"] == 0:
            _save_checkpoint(global_lora, round_id, tracker.output_dir)

    # final save
    tracker.save()
    _save_checkpoint(global_lora, total_rounds - 1, tracker.output_dir)

    total_time = time.time() - training_start
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total time: {_format_time(total_time)}")
    print(f"  Avg per round: {_format_time(total_time / total_rounds)}")
    print(f"  Target reached at round: {tracker.target_round}")
    print(f"  Deadline compliance: {tracker.deadline_compliance():.2%}")
    print(f"{'=' * 60}")

    return tracker


def _assign_tiers(client_ids: list[int], tiers: list[dict]) -> dict[int, dict]:
    assignment = {}
    idx = 0
    for tier in tiers:
        for _ in range(tier["count"]):
            if idx < len(client_ids):
                assignment[client_ids[idx]] = tier
                idx += 1
    # remaining clients get last tier
    for i in range(idx, len(client_ids)):
        assignment[client_ids[i]] = tiers[-1]
    return assignment


def _save_checkpoint(lora_state: dict, round_id: int, output_dir):
    path = output_dir / f"lora_round_{round_id}.pt"
    torch.save(lora_state, path)
    print(f"  Checkpoint saved: {path}")
