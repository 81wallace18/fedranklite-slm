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


def run(cfg: dict):
    seed = cfg["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    fed_cfg = cfg["federation"]
    r_max = cfg["model"]["lora"]["r_max"]

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

    logger.info(f"Starting federation: {fed_cfg['num_rounds']} rounds, {fed_cfg['total_clients']} clients")

    for round_id in range(fed_cfg["num_rounds"]):
        t_round_start = time.time()

        # select clients for this round
        selected = rng.choice(
            all_client_ids,
            size=min(fed_cfg["clients_per_round"], len(all_client_ids)),
            replace=False,
        ).tolist()

        # scheduler decides ranks
        assignments = scheduler.allocate(round_id, selected, telemetry_history)

        # train each client
        results: list[ClientResult] = []
        round_exceeded_deadline = False

        for cid in selected:
            rank = assignments[cid]["rank"]
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
            tier = tier_assignment[cid]
            result.train_time /= tier["compute_factor"]

            # check deadline
            exceeded = False
            if cfg["deadline"]["enabled"] and result.train_time > cfg["deadline"]["seconds"]:
                exceeded = True
                round_exceeded_deadline = True
                if cfg["deadline"]["straggler_policy"] == "drop":
                    logger.info(f"  Client {cid} dropped (deadline exceeded: {result.train_time:.1f}s)")
                    continue

            results.append(result)

            # store telemetry
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
            })

        # aggregate
        if results:
            agg_cfg = fed_cfg["aggregation"]
            global_lora = aggregate_lora(
                global_lora, results, r_max,
                method=agg_cfg["method"],
                weighting=agg_cfg["weighting"],
            )
            set_lora_state(model, global_lora)

        round_time = time.time() - t_round_start

        # evaluate
        eval_score = None
        eval_loss = None
        if eval_loader and (round_id + 1) % cfg["evaluation"]["eval_every"] == 0:
            eval_result = evaluate_global(model, eval_loader, cfg["evaluation"]["metric"])
            eval_score = eval_result["score"]
            eval_loss = eval_result["loss"]

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
    _save_checkpoint(global_lora, fed_cfg["num_rounds"] - 1, tracker.output_dir)

    logger.info(
        f"Done. Target reached at round {tracker.target_round}. "
        f"Deadline compliance: {tracker.deadline_compliance():.2%}"
    )

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
    logger.info(f"Checkpoint saved: {path}")
