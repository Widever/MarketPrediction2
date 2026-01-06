import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import pandas as pd

import mp.optimizer.comb as comb_engine

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"optimize_main_dir")

_train_df = None

def comb_grade_worker(comb):
    global _train_df

    if _train_df is None:
        _train_df = pd.read_csv(f"{data_dir}/marked_points_train.csv")

    select_mask = comb_engine.get_select_combs_mask(_train_df, [comb])
    comb_df = _train_df[select_mask]

    return comb_engine.grade_comb(comb_df, comb)

def grade_combs_parallel(all_combs):

    workers = cpu_count() - 2  # або cpu_count() - 1
    # workers = 1  # або cpu_count() - 1
    total = len(all_combs)

    report_every = 10_000
    processed = 0
    start = time.time()
    grades = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for v in executor.map(comb_grade_worker, all_combs, chunksize=10_000):
            grades.append(v)
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

    return grades