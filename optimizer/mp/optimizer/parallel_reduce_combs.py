import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"optimize_main_dir")

_train_df = None


def get_select_combs_mask_ignore_scope(marked_points_df: pd.DataFrame, combs: list[tuple[str, ...]]) -> pd.Series:
    if not combs:
        return pd.Series(False, index=marked_points_df.index)

    combs_masks = [
        marked_points_df[list(comb)].all(axis=1)
        for comb in combs
    ]

    any_comb = np.logical_or.reduce(combs_masks)

    return pd.Series(any_comb, index=marked_points_df.index)

def profit_loss_in_df(comb_df):
    # Peak down
    profit = len(
        comb_df[
            (comb_df["peak_down"]) &
            (~comb_df["peak_up"])
        ]
    )
    loss = len(comb_df) - profit
    return profit, loss

def comb_grade_worker(comb):
    global _train_df

    if _train_df is None:
        start = time.perf_counter()
        _train_df = pd.read_csv(f"{data_dir}/marked_points_train.csv")
        end = time.perf_counter()
        print(f"Read _train_df in worker, elapsed time: {end-start}s.")

    select_mask = get_select_combs_mask_ignore_scope(_train_df, [comb])
    comb_df = _train_df[select_mask]
    profit, loss = profit_loss_in_df(comb_df)

    return comb, profit, profit+loss


def grade_combs_parallel_cpu(combs: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    global _train_df
    workers = cpu_count() - 2
    print(f"Parallel reduce combs {workers=}.")
    total = len(combs)

    report_every = 10_000
    processed = 0
    start = time.time()
    new_combs = []

    worker = comb_grade_worker

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for v in executor.map(worker, combs, chunksize=10_000):
            new_combs.append(v)
            processed += 1

            if processed % report_every == 0:
                elapsed = time.time() - start
                rate = processed / elapsed
                remaining = total - processed
                eta = remaining / rate if rate else 0

                print(
                    f"[{processed:,}/{total:,}] "
                    f"({processed / total:.1%}) | "
                    f"{rate:,.0f} items/s | "
                    f"ETA ~ {eta / 60:.1f} min"
                )

    _train_df = None

    return new_combs