import os

import numpy as np
import pandas as pd

from mp.optimizer.comb import CombGrade, get_select_combs_mask, profit_loss_in_df

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"optimize_main_dir")


def super_benchmark(combs: list[CombGrade] = None):
    # combs: list[CombGrade] = [
    #     CombGrade(comb=('#tag_ada_trend_kind_eq_falling', '#tag_btc_ada_ch_from_peak_ratio_gt_1.0',
    #                     '#tag_doge_sui_amp_ratio_gt_0.87', '#tag_doge_ada_log_return_ratio_lt_1.6',
    #                     '#tag_btc_doge_log_return_ratio_gt_-0.2', '#tag_eth_xrp_log_return_ratio_lt_1.6',
    #                     '#tag_btc_eth_log_return_ratio_gt_0.0', '#tag_doge_xrp_log_return_ratio_lt_1.6',
    #                     '#tag_eth_ada_amp_ratio_gt_0.48', '#tag_btc_doge_amp_ratio_gt_0.25',
    #                     '#tag_eth_ada_log_return_ratio_gt_0.4', '#tag_btc_eth_ch_from_peak_ratio_gt_-1.0',
    #                     '#tag_eth_doge_amp_ratio_gt_0.7', '#tag_btc_doge_ch_from_peak_ratio_gt_1.0',
    #                     '#tag_ada_trend_ch_from_peak_gt_-0.025'), score=0.3780487804878049,
    #               props={'loss': 52, 'profit': 12}, verify_grade=None),
    #
    # ]

    marked_points_df = pd.read_csv(f"{data_dir}/marked_points_frozen.csv")
    profit, loss = profit_loss_in_df(marked_points_df)
    print(f"{profit=}")
    print(f"{loss=}")

    if combs:
        select_mask = get_select_combs_mask(marked_points_df, [comb_.comb for comb_ in combs])

        selected_df = marked_points_df[select_mask].reset_index(drop=True)
    else:
        selected_df = marked_points_df

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
    return intervals_stat, total_count, total_sl_count, total_k

if __name__ == "__main__":
    super_benchmark([])