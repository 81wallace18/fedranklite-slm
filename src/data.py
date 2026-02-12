from __future__ import annotations

import numpy as np
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader


def load_and_partition(cfg: dict, tokenizer) -> tuple[list[Dataset], Dataset]:
    ds_path = cfg["data"]["dataset"]
    parts = ds_path.split("/")
    if len(parts) == 2:
        raw = load_dataset(parts[0], parts[1])
    else:
        raw = load_dataset(ds_path)

    train_ds = raw["train"]
    eval_ds = raw["validation"] if "validation" in raw else None

    text_cols = cfg["data"]["text_columns"]
    max_length = cfg["data"].get("max_length", 128)
    label_col = "label"

    def tokenize_fn(examples):
        if len(text_cols) == 2:
            return tokenizer(
                examples[text_cols[0]],
                examples[text_cols[1]],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        return tokenizer(
            examples[text_cols[0]],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    remove_cols = [c for c in text_cols if c in train_ds.column_names]
    # also remove idx column if present
    if "idx" in train_ds.column_names:
        remove_cols.append("idx")

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    train_ds = train_ds.rename_column(label_col, "labels")
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    if eval_ds is not None:
        remove_cols_eval = [c for c in text_cols if c in eval_ds.column_names]
        if "idx" in eval_ds.column_names:
            remove_cols_eval.append("idx")
        eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=remove_cols_eval)
        eval_ds = eval_ds.rename_column(label_col, "labels")
        eval_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # partition
    part_cfg = cfg["data"]["partition"]
    n_clients = cfg["federation"]["total_clients"]

    if cfg["data"]["task_type"] == "regression":
        # for regression, use IID-like split (can't do label-based)
        method = part_cfg["method"] if part_cfg["method"] == "iid" else part_cfg["method"]
        labels_np = np.zeros(len(train_ds))  # dummy for partition
    else:
        method = part_cfg["method"]
        labels_raw = train_ds["labels"]
        labels_np = labels_raw.numpy() if hasattr(labels_raw, "numpy") else np.array(labels_raw)

    indices = _partition_indices(
        labels=labels_np,
        n_clients=n_clients,
        method=method,
        alpha=part_cfg.get("alpha", 1.0),
        min_samples=part_cfg.get("min_samples", 10),
        seed=cfg["seed"],
    )
    client_datasets = [train_ds.select(idx) for idx in indices]

    return client_datasets, eval_ds


def _partition_indices(
    labels: np.ndarray,
    n_clients: int,
    method: str,
    alpha: float,
    min_samples: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)

    if method == "iid":
        indices = rng.permutation(len(labels))
        return np.array_split(indices, n_clients)

    if method == "dirichlet":
        unique_labels = np.unique(labels)
        client_indices = [[] for _ in range(n_clients)]

        for lbl in unique_labels:
            lbl_idx = np.where(labels == lbl)[0]
            rng.shuffle(lbl_idx)
            proportions = rng.dirichlet(np.repeat(alpha, n_clients))
            proportions = proportions / proportions.sum()
            splits = (np.cumsum(proportions) * len(lbl_idx)).astype(int)[:-1]
            for client_id, chunk in enumerate(np.split(lbl_idx, splits)):
                client_indices[client_id].extend(chunk.tolist())

        # enforce min_samples: redistribute from largest to smallest
        for i in range(n_clients):
            if len(client_indices[i]) < min_samples:
                largest = max(range(n_clients), key=lambda j: len(client_indices[j]))
                deficit = min_samples - len(client_indices[i])
                moved = client_indices[largest][-deficit:]
                client_indices[largest] = client_indices[largest][:-deficit]
                client_indices[i].extend(moved)

        return [np.array(idx) for idx in client_indices]

    if method == "label_skew":
        unique_labels = np.unique(labels)
        # assign 2 labels per client (with overlap)
        client_indices = [[] for _ in range(n_clients)]
        labels_per_client = max(2, len(unique_labels) // n_clients)
        for i in range(n_clients):
            chosen = rng.choice(unique_labels, size=labels_per_client, replace=False)
            for lbl in chosen:
                lbl_idx = np.where(labels == lbl)[0]
                chunk_size = len(lbl_idx) // n_clients
                start = i * chunk_size
                client_indices[i].extend(lbl_idx[start : start + chunk_size].tolist())
        return [np.array(idx) for idx in client_indices]

    raise ValueError(f"Unknown partition method: {method}")


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
