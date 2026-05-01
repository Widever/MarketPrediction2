"""
parallel_reduce_combs_dml.py
GPU-accelerated combination reducer for AMD Radeon on Windows (DirectML / PyTorch).

Strategy
--------
Instead of farming one combination per CPU worker, we:
  1. Load the CSV once and convert every boolean column to a GPU tensor.
  2. Build a binary selector matrix for a *batch* of combinations at once.
  3. Use matrix multiplication to count how many rows match each combination.
  4. Count peak_down & peak_up hits with a single vectorised pass per batch.

Requirements
------------
  py -3.12 -m pip install torch-directml pandas numpy
"""

import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch_directml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimize_main_dir")

# Number of combinations evaluated in one GPU batch.
# Increase if you have plenty of VRAM; decrease if you get OOM errors.
DEFAULT_BATCH_SIZE = 8_000

REPORT_EVERY = 10_000          # progress reporting cadence (combinations)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the DirectML GPU device, or CPU as fallback."""
    try:
        device = torch_directml.device()
        print(f"[GPU] Using DirectML device: {torch_directml.device_name(0)}")
    except Exception as e:
        print(f"[GPU] DirectML unavailable ({e}) — falling back to CPU.")
        device = torch.device("cpu")
    return device


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_train_tensors(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """
    Read the CSV and return pre-allocated GPU tensors.

    Returns
    -------
    data_bool   : (N, C)  float32 — all boolean feature columns
    peak_down   : (N,)    float32 — 'peak_down' column
    peak_up     : (N,)    float32 — 'peak_up'   column
    col_names   : list[str]       — column names matching axis-1 of data_bool
    """
    csv_path = os.path.join(data_dir, "marked_points_train.csv")
    t0 = time.perf_counter()
    df = pd.read_csv(csv_path)
    print(f"CSV loaded in {time.perf_counter() - t0:.2f}s  |  shape: {df.shape}")

    # Identify boolean / uint8 feature columns (exclude target columns)
    target_cols = {"peak_down", "peak_up"}
    feature_cols = [
        c for c in df.columns
        if c not in target_cols and df[c].dtype in (bool, np.bool_, np.uint8, np.int8)
    ]

    # DirectML requires float32 — no float16 matmul support
    t1 = time.perf_counter()
    data_bool = torch.tensor(df[feature_cols].values.astype(np.float32), dtype=torch.float32, device=device)
    peak_down  = torch.tensor(df["peak_down"].values.astype(np.float32), dtype=torch.float32, device=device)
    peak_up    = torch.tensor(df["peak_up"].values.astype(np.float32),   dtype=torch.float32, device=device)
    print(f"Tensors on {device} in {time.perf_counter() - t1:.2f}s  |  "
          f"data_bool: {tuple(data_bool.shape)}  |  features: {len(feature_cols)}")

    return data_bool, peak_down, peak_up, feature_cols


# ---------------------------------------------------------------------------
# Core GPU kernel
# ---------------------------------------------------------------------------

def evaluate_batch(
    data_bool:    torch.Tensor,       # (N, C)  float32
    peak_down:    torch.Tensor,       # (N,)    float32
    peak_up:      torch.Tensor,       # (N,)    float32
    col_index:    dict[str, int],     # column-name → index in data_bool
    batch:        list[tuple[str, ...]],
    min_profit_n: int,
) -> list[tuple[str, ...]]:
    """
    Evaluate a batch of combinations entirely on the GPU.

    For each combination we need:
      matched_rows  = rows where ALL columns in the combo are True
      profit        = |matched_rows ∩ (peak_down=1 ∧ peak_up=0)|
      keep          = profit > min_profit_n

    Vectorised approach
    -------------------
    selector : (B, C) float32 — 1 where the combo uses that column
    match    : (B, N) = (selector @ data_bool.T) == combo_length
               a row matches iff every selected column is True
    profit   : (B,)  = match @ (peak_down & ~peak_up)
    """
    B      = len(batch)
    C      = data_bool.shape[1]
    device = data_bool.device

    # Build selector matrix (B, C) and combo lengths (B,)
    selector = torch.zeros(B, C, dtype=torch.float32, device=device)
    lengths  = torch.zeros(B,    dtype=torch.float32, device=device)
    for i, comb in enumerate(batch):
        idxs = [col_index[c] for c in comb]
        selector[i, idxs] = 1.0
        lengths[i] = float(len(comb))

    # hit_counts[i, row] = number of selected columns that are True for that row
    hit_counts = torch.mm(selector, data_bool.T)            # (B, N)

    # A row matches combo i iff every selected column is True
    match = hit_counts == lengths.unsqueeze(1)              # (B, N)  bool

    # profit_signal[row] = 1 iff peak_down=1 AND peak_up=0
    profit_signal = peak_down * (1.0 - peak_up)            # (N,)  float32

    # profit[i] = number of matched rows that are profit signals
    # Note: torch.mv is not supported by DirectML, so we use torch.mm with a column vector
    profits = torch.mm(match.to(torch.float32), profit_signal.unsqueeze(1)).squeeze(1)  # (B,)

    # Filter — move result back to CPU for Python-level filtering
    keep_mask = (profits > min_profit_n).cpu().numpy()

    return [comb for comb, keep in zip(batch, keep_mask) if keep]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reduce_combs_gpu(
    combs:        list[tuple[str, ...]],
    min_profit_n: int,
    batch_size:   int = DEFAULT_BATCH_SIZE,
    device:       Optional[torch.device] = None,
) -> list[tuple[str, ...]]:
    """
    GPU replacement for reduce_combs_parallel().

    Parameters
    ----------
    combs         : list of column-name tuples to evaluate
    min_profit_n  : minimum profit count to keep a combination
    batch_size    : how many combinations to evaluate per GPU batch
    device        : torch.device (auto-detected if None)

    Returns
    -------
    Filtered list of combinations where profit > min_profit_n.
    """
    if device is None:
        device = get_device()

    data_bool, peak_down, peak_up, feature_cols = load_train_tensors(device)
    col_index = {name: i for i, name in enumerate(feature_cols)}

    total     = len(combs)
    new_combs = []
    processed = 0
    start     = time.time()

    print(f"[GPU] Evaluating {total:,} combinations  |  batch_size={batch_size:,}")

    for batch_start in range(0, total, batch_size):
        batch = combs[batch_start : batch_start + batch_size]

        kept = evaluate_batch(data_bool, peak_down, peak_up, col_index, batch, min_profit_n)
        new_combs.extend(kept)
        processed += len(batch)

        if processed % REPORT_EVERY < batch_size or processed >= total:
            elapsed   = time.time() - start
            rate      = processed / elapsed if elapsed else 0
            remaining = total - processed
            eta       = remaining / rate if rate else 0
            print(
                f"  [{processed:,}/{total:,}] "
                f"({processed / total:.1%}) | "
                f"{rate:,.0f} combos/s | "
                f"ETA ~ {eta / 60:.1f} min | "
                f"kept so far: {len(new_combs):,}"
            )

    elapsed = time.time() - start
    print(f"[GPU] Done in {elapsed:.1f}s — kept {len(new_combs):,} / {total:,} combinations.")
    return new_combs


# ---------------------------------------------------------------------------
# Backwards-compat shim
# ---------------------------------------------------------------------------

def reduce_combs_parallel_gpu(
    combs: list[tuple[str, ...]],
    min_profit_n: int,
) -> list[tuple[str, ...]]:
    """Drop-in replacement for the original CPU multiprocessing version."""
    return reduce_combs_gpu(combs, min_profit_n)