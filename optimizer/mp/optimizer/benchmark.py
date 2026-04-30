import os

import numpy as np
import pandas as pd

from mp.optimizer.comb import CombGrade, get_select_combs_mask, profit_loss_in_df

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"optimize_main_dir")


def super_benchmark(combs: list[CombGrade] = None):
    # combs: list[CombGrade] = [
    #     CombGrade(comb=('#tag_ada_ampl_gt_0.012', '#tag_avg_drop_from_high_gt_0.03',
    #                     '#tag_btc_doge_log_return_ratio_lt_0.513', '#tag_eth_ampl_gt_0.008',
    #                     '#tag_btc_trend_kind_eq_falling', '#tag_doge_xrp_drop_from_high_ratio_lt_1.555',
    #                     '#tag_btc_xrp_ampl_ratio_lt_0.672', '#tag_eth_ada_ampl_ratio_lt_1.227',
    #                     '#tag_btc_ada_drop_from_high_ratio_lt_1.57', '#tag_btc_rise_from_low_lt_0.011'),
    #               score=0.411522633744856, props={'loss': 93, 'profit': 50, 'total': 143}, verify_grade=None)
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