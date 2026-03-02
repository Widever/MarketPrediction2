import os

import numpy as np
import pandas as pd

from mp.optimizer.comb import CombGrade, get_select_combs_mask

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"optimize_main_dir")


def super_benchmark():
    combs: list[CombGrade] = [
        # CombGrade(comb=('#tag_btc_doge_trend_value_ratio_gt_0.5', '#tag_doge_xrp_trend_value_ratio_lt_-0.5',
        #                 '#tag_btc_eth_log_return_ratio_gt_0.0', '#tag_doge_xrp_amp_ratio_gt_0.5',
        #                 '#tag_btc_trend_len_gt_10.0', '#tag_eth_xrp_amp_ratio_gt_0.5', '#tag_btc_xrp_amp_ratio_gt_0.25',
        #                 '#tag_eth_trend_len_gt_5.0', '#tag_ada_trend_kind_eq_UP', '#tag_eth_ada_amp_ratio_gt_0.48'),
        #           count_=90, sl_count=11, uniformity=0.0, uniformity2=np.float64(0.18961443226034666),
        #           k=7.181818181818182, k2=68, verify_grade=None),
        CombGrade(comb=('#tag_eth_ada_trend_value_ratio_lt_-0.5', '#tag_doge_trend_value_gt_0.01',
                        '#tag_btc_eth_trend_value_ratio_gt_-1.0', '#tag_btc_ada_log_return_ratio_gt_0.0',
                        '#tag_btc_trend_value_gt_-0.025', '#tag_doge_trend_len_gt_5.0', '#tag_btc_trend_len_gt_10.0',
                        '#tag_eth_trend_len_gt_5.0', '#tag_sui_trend_len_gt_5.0', '#tag_ada_trend_len_lt_5.0'),
                  count_=57, sl_count=8, uniformity=0.0, uniformity2=np.float64(0.11708524938096923), k=6.125, k2=41,
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