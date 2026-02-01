import os

import numpy as np
import pandas as pd

from mp.optimizer.comb import CombGrade, get_select_combs_mask

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"optimize_main_dir")


def super_benchmark():
    combs: list[CombGrade] = [
        CombGrade(comb=('#tag_btc_ada_amp_ratio_lt_0.3', '#tag_ada_trend_kind_gt_DOWN',
                        '#tag_btc_doge_log_return_ratio_gt_0.2', '#tag_doge_ada_trend_value_ratio_lt_-0.2',
                        '#tag_btc_ada_log_return_ratio_gt_0.25', '#tag_doge_ada_log_return_ratio_gt_0.8',
                        '#tag_btc_eth_amp_ratio_lt_0.39', '#tag_eth_ada_log_return_ratio_gt_0.8'), count_=58,
                  sl_count=7, uniformity=0.0, uniformity2=np.float64(0.2888291475061903), k=7.285714285714286,
                  verify_grade=None),

    ]

    marked_points_df = pd.read_csv(f"{data_dir}/marked_points_frozen.csv")

    select_mask = get_select_combs_mask(marked_points_df, [comb_.comb for comb_ in combs])

    selected_df = marked_points_df[select_mask].reset_index(drop=True)

    selected_df["interval"] = pd.cut(selected_df["index"], bins=10)

    intervals_stat = selected_df.groupby("interval", observed=False).agg(
        count_=("index", "size"),
        sl_count_=("sl", "sum"),
    ).reset_index()

    intervals_stat["k"] = (intervals_stat["count_"] - intervals_stat["sl_count_"]) / intervals_stat["sl_count_"]
    total_count = intervals_stat["count_"].sum()
    total_sl_count = intervals_stat["sl_count_"].sum()
    total_k = (total_count - total_sl_count) / total_sl_count if total_sl_count > 0 else 0

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print("Intervals stat:")
    print(intervals_stat)
    print()
    print(f"Total: {total_count=}, {total_sl_count=}, {total_k=}")