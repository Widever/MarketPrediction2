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

from mp.optimizer.comb import grade_comb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimize_main_dir")

# Number of combinations evaluated in one GPU batch.
# Increase if you have plenty of VRAM; decrease if you get OOM errors.
DEFAULT_BATCH_SIZE = 8_000

REPORT_EVERY = 10_000          # progress reporting cadence (combinations)

# ---------------------------------------------------------------------------
# Profit-signal mode
# ---------------------------------------------------------------------------
# Controls which condition defines a "profitable" row when scoring combinations.
#
#   MODE 1 — original:
#       peak_down == 1  AND  peak_up == 0
#
#   MODE 2 — recent up-peak, short remaining window:
#       (len_peak_to_peak - len_from_last_peak) < M  AND  last_peak_type == 'up'
#
#   MODE 3 — recent down-peak, long total window:
#       len_from_last_peak < K  AND  len_peak_to_peak > L  AND  last_peak_type == 'down'
#
PROFIT_SIGNAL_MODE: int = 2   # choose 1, 2, or 3

# Thresholds used by MODE 2 and MODE 3 (ignored in MODE 1)
M: int = 3   # MODE 2: max allowed (len_peak_to_peak - len_from_last_peak)
K: int = 5    # MODE 3: max allowed len_from_last_peak
L: int = 10   # MODE 3: min required len_peak_to_peak


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, list[str]]:
    """
    Read the CSV and return pre-allocated GPU tensors.

    Returns
    -------
    data_bool          : (N, C)  float32 — all boolean feature columns
    peak_down          : (N,)    float32 — 'peak_down' column          (MODE 1)
    peak_up            : (N,)    float32 — 'peak_up'   column          (MODE 1)
    len_peak_to_peak   : (N,)    float32 — 'len_peak_to_peak' column   (MODE 2/3)
    len_from_last_peak : (N,)    float32 — 'len_from_last_peak' column (MODE 2/3)
    last_peak_type     : (N,)    np.ndarray[str]  — 'last_peak_type'   (MODE 2/3)
    col_names          : list[str] — column names matching axis-1 of data_bool
    """
    csv_path = os.path.join(data_dir, "marked_points_train.csv")
    t0 = time.perf_counter()
    df = pd.read_csv(csv_path)
    print(f"CSV loaded in {time.perf_counter() - t0:.2f}s  |  shape: {df.shape}")

    # Identify boolean / uint8 feature columns (exclude target columns)
    target_cols = {"peak_down", "peak_up", "len_peak_to_peak", "len_from_last_peak", "last_peak_type"}
    feature_cols = [
        c for c in df.columns
        if c not in target_cols and df[c].dtype in (bool, np.bool_, np.uint8, np.int8)
    ]

    # DirectML requires float32 — no float16 matmul support
    t1 = time.perf_counter()
    data_bool          = torch.tensor(df[feature_cols].values.astype(np.float32),              dtype=torch.float32, device=device)
    peak_down          = torch.tensor(df["peak_down"].values.astype(np.float32),               dtype=torch.float32, device=device)
    peak_up            = torch.tensor(df["peak_up"].values.astype(np.float32),                 dtype=torch.float32, device=device)
    len_peak_to_peak   = torch.tensor(df["len_peak_to_peak"].values.astype(np.float32),        dtype=torch.float32, device=device)
    len_from_last_peak = torch.tensor(df["len_from_last_peak"].values.astype(np.float32),      dtype=torch.float32, device=device)
    last_peak_type     = df["last_peak_type"].values   # kept as NumPy str array; used for boolean mask on CPU

    print(f"Tensors on {device} in {time.perf_counter() - t1:.2f}s  |  "
          f"data_bool: {tuple(data_bool.shape)}  |  features: {len(feature_cols)}")

    return data_bool, peak_down, peak_up, len_peak_to_peak, len_from_last_peak, last_peak_type, feature_cols


# ---------------------------------------------------------------------------
# Core GPU kernel
# ---------------------------------------------------------------------------

def evaluate_batch(
    data_bool:          torch.Tensor,       # (N, C)  float32
    peak_down:          torch.Tensor,       # (N,)    float32
    peak_up:            torch.Tensor,       # (N,)    float32
    len_peak_to_peak:   torch.Tensor,       # (N,)    float32
    len_from_last_peak: torch.Tensor,       # (N,)    float32
    last_peak_type:     np.ndarray,         # (N,)    str  ('up' | 'down')
    col_index:          dict[str, int],     # column-name → index in data_bool
    batch:              list[tuple[str, ...]],
) -> list[tuple[tuple[str, ...], int, int]]:
    """
    Evaluate a batch of combinations entirely on the GPU.

    For each combination we compute:
      total   = number of rows where ALL columns in the combo are True
      profit  = |matched_rows ∩ profit_signal|

    The profit_signal vector is built from the module-level PROFIT_SIGNAL_MODE:
      MODE 1: peak_down == 1  AND  peak_up == 0
      MODE 2: (len_peak_to_peak - len_from_last_peak) < M  AND  last_peak_type == 'up'
      MODE 3: len_from_last_peak < K  AND  len_peak_to_peak > L  AND  last_peak_type == 'down'

    Vectorised approach
    -------------------
    selector : (B, C) float32 — 1 where the combo uses that column
    match    : (B, N) = (selector @ data_bool.T) == combo_length
               a row matches iff every selected column is True
    totals   : (B,)  = match.sum(dim=1)
    profits  : (B,)  = match @ profit_signal

    Returns
    -------
    List of (comb, profit, total) for every combination in the batch.
    All filtering is left to the caller.
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
    match   = hit_counts == lengths.unsqueeze(1)            # (B, N)  bool
    match_f = match.to(torch.float32)                       # (B, N)  float32

    # total[i] = number of rows matched by combo i
    totals = match_f.sum(dim=1)                             # (B,)

    # ------------------------------------------------------------------
    # Build profit_signal (N,) according to PROFIT_SIGNAL_MODE
    # ------------------------------------------------------------------
    if PROFIT_SIGNAL_MODE == 1:
        # Original: peak_down == 1 AND peak_up == 0
        profit_signal = peak_down * (1.0 - peak_up)                        # (N,)  float32

    elif PROFIT_SIGNAL_MODE == 2:
        # (len_peak_to_peak - len_from_last_peak) < M  AND  last_peak_type == 'up'
        is_up_mask  = torch.tensor(last_peak_type == "up", dtype=torch.float32, device=device)
        window_cond = ((len_peak_to_peak - len_from_last_peak) < M).to(torch.float32)
        profit_signal = window_cond * is_up_mask                            # (N,)  float32

    elif PROFIT_SIGNAL_MODE == 3:
        # len_from_last_peak < K  AND  len_peak_to_peak > L  AND  last_peak_type == 'down'
        is_down_mask  = torch.tensor(last_peak_type == "down", dtype=torch.float32, device=device)
        recent_cond   = (len_from_last_peak < K).to(torch.float32)
        long_cond     = (len_peak_to_peak   > L).to(torch.float32)
        profit_signal = recent_cond * long_cond * is_down_mask              # (N,)  float32

    else:
        raise ValueError(f"Unknown PROFIT_SIGNAL_MODE={PROFIT_SIGNAL_MODE!r}. Must be 1, 2, or 3.")

    # profit[i] = number of matched rows that are profit signals
    # Note: torch.mv is not supported by DirectML, so we use torch.mm with a column vector
    profits = torch.mm(match_f, profit_signal.unsqueeze(1)).squeeze(1)  # (B,)

    # Single host transfer: move both vectors together to minimise PCIe round-trips
    results_np = torch.stack([profits, totals], dim=0).cpu().numpy()    # (2, B)

    return [
        (comb, int(results_np[0, i]), int(results_np[1, i]))
        for i, comb in enumerate(batch)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade_combs_gpu(
    combs:      list[tuple[str, ...]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    device:     Optional[torch.device] = None,
) -> list[tuple[tuple[str, ...], int, int]]:
    """
    GPU replacement for reduce_combs_parallel().

    Parameters
    ----------
    combs       : list of column-name tuples to evaluate
    batch_size  : how many combinations to evaluate per GPU batch
    device      : torch.device (auto-detected if None)

    Returns
    -------
    List of (comb, profit, total) for every input combination.
    No filtering is applied — callers decide their own threshold.
    """
    if device is None:
        device = get_device()

    data_bool, peak_down, peak_up, len_peak_to_peak, len_from_last_peak, last_peak_type, feature_cols = load_train_tensors(device)
    col_index = {name: i for i, name in enumerate(feature_cols)}

    total     = len(combs)
    results: list[tuple[tuple[str, ...], int, int]] = []
    processed = 0
    start     = time.time()

    print(f"[GPU] Evaluating {total:,} combinations  |  batch_size={batch_size:,}  |  mode={PROFIT_SIGNAL_MODE}")

    for batch_start in range(0, total, batch_size):
        batch = combs[batch_start : batch_start + batch_size]

        results.extend(evaluate_batch(
            data_bool, peak_down, peak_up,
            len_peak_to_peak, len_from_last_peak, last_peak_type,
            col_index, batch,
        ))
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
                f"ETA ~ {eta / 60:.1f} min"
            )

    elapsed = time.time() - start
    print(f"[GPU] Done in {elapsed:.1f}s — evaluated {total:,} combinations.")
    return results


# ---------------------------------------------------------------------------
# Backwards-compat shim
# ---------------------------------------------------------------------------

def grade_combs_parallel_gpu(
    combs: list[tuple[str, ...]],
) -> list[tuple[tuple[str, ...], int, int]]:
    """
    Drop-in replacement for the original CPU multiprocessing version.
    Applies the profit threshold filter that the core function no longer enforces.
    """
    results = grade_combs_gpu(combs)
    return results
    # return [comb for comb, profit, _total in results if profit > min_profit_n]