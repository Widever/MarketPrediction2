import os

import numpy as np
import pandas as pd

from mp.optimizer.comb import CombGrade, get_select_combs_mask, profit_loss_in_df

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"optimize_main_dir")


def super_benchmark(combs: list[CombGrade] = None):
    # combs: list[CombGrade] = [
    #     CombGrade(
    #         comb=('#tag_btc_doge_log_return_ratio_lt_-1.316', '#tag_ada_trend_kind_eq_falling',
    #               '#tag_sui_trend_ch_from_peak_lt_-0.011', '#tag_avg_log_return_ratio_lt_0.468',
    #               '#tag_btc_ada_log_return_ratio_lt_0.451', '#tag_avg_ch_from_peak_ratio_gt_-0.098',
    #               '#tag_eth_doge_log_return_ratio_lt_1.081', '#tag_doge_sui_ch_from_peak_ratio_lt_0.999',
    #               '#tag_eth_doge_ch_from_peak_ratio_gt_0.59', '#tag_eth_ada_ch_from_peak_ratio_lt_0.904'),
    #         score=0.35294117647058826, props={'loss': 60, 'profit': 10, 'total': 70}, verify_grade=None),
    #     CombGrade(comb=('#tag_ada_trend_ch_from_peak_gt_0.048', '#tag_avg_log_return_ratio_lt_0.468',
    #                     '#tag_xrp_trend_ch_from_peak_gt_0.052', '#tag_btc_doge_ch_from_peak_ratio_lt_0.828',
    #                     '#tag_sui_trend_ch_from_peak_gt_-0.042', '#tag_btc_ada_ch_from_peak_ratio_gt_-3.617',
    #                     '#tag_btc_xrp_ch_from_peak_ratio_gt_-3.48', '#tag_eth_ada_ch_from_peak_ratio_gt_-1.563',
    #                     '#tag_eth_xrp_ch_from_peak_ratio_gt_-2.18', '#tag_doge_ada_ch_from_peak_ratio_gt_-2.031'),
    #               score=0.8973063973063973, props={'loss': 11, 'profit': 483, 'total': 494}, verify_grade=None)
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