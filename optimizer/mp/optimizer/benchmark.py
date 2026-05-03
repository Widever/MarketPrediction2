import os

import numpy as np
import pandas as pd

from mp.optimizer.comb import get_select_combs_mask, profit_loss_in_df
from mp.optimizer.comb_v2 import CombGrade

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"optimize_main_dir")


def super_benchmark(combs: list[CombGrade] = None):
    # combs: list[CombGrade] = [
    #     CombGrade(comb=('#tag_doge_log_return_gt_0.005', '#tag_ada_ampl_gt_0.009', '#tag_doge_drop_from_high_lt_0.016',
    #                     '#tag_avg_rise_from_low_lt_0.023', '#tag_ada_rise_from_low_gt_0.014',
    #                     '#tag_btc_doge_log_return_ratio_lt_0.513', '#tag_doge_sui_log_return_ratio_gt_-0.623',
    #                     '#tag_doge_rise_from_low_lt_0.038', '#tag_avg_log_return_gt_0.004',
    #                     '#tag_doge_xrp_log_return_ratio_gt_0.566', '#tag_eth_doge_ampl_ratio_lt_1.595',
    #                     '#tag_eth_ada_drop_from_high_ratio_lt_3.126', '#tag_doge_xrp_drop_from_high_ratio_lt_1.555',
    #                     '#tag_btc_ada_log_return_ratio_gt_-0.533', '#tag_btc_xrp_log_return_ratio_gt_-0.29'),
    #               bayesian_winrate=0.7625, profit=51, total=60),
    #     # CombGrade(comb=('#tag_ada_ampl_gt_0.012', '#tag_btc_log_return_lt_-0.001', '#tag_ada_drop_from_high_gt_0.025',
    #     #                 '#tag_sui_rise_from_low_gt_0.02', '#tag_btc_rise_from_low_lt_0.039',
    #     #                 '#tag_avg_rise_from_low_gt_0.013', '#tag_doge_xrp_drop_from_high_ratio_lt_1.555',
    #     #                 '#tag_doge_ada_log_return_ratio_lt_1.569', '#tag_eth_ada_rise_from_low_ratio_lt_1.393',
    #     #                 '#tag_btc_ada_log_return_ratio_lt_1.435', '#tag_btc_doge_log_return_ratio_gt_-1.316',
    #     #                 '#tag_eth_ada_log_return_ratio_lt_2.903', '#tag_eth_xrp_log_return_ratio_lt_3.16',
    #     #                 '#tag_avg_log_return_ratio_lt_4.809', '#tag_xrp_log_return_lt_0.004'),
    #     #           bayesian_winrate=0.916083916083916, profit=121, total=123),
    #     # CombGrade(comb=('#tag_sui_log_return_gt_0.005', '#tag_xrp_drop_from_high_gt_0.04', '#tag_ada_ampl_lt_0.012',
    #     #                 '#tag_ada_drop_from_high_gt_0.013', '#tag_xrp_log_return_gt_0.002',
    #     #                 '#tag_avg_ampl_ratio_gt_0.605', '#tag_doge_log_return_gt_0.002', '#tag_xrp_ampl_lt_0.011',
    #     #                 '#tag_btc_drop_from_high_gt_0.008', '#tag_eth_log_return_gt_0.001',
    #     #                 '#tag_eth_doge_rise_from_low_ratio_lt_1.699', '#tag_eth_doge_ampl_ratio_lt_1.595',
    #     #                 '#tag_doge_xrp_ampl_ratio_lt_2.356', '#tag_btc_doge_drop_from_high_ratio_lt_2.326',
    #     #                 '#tag_btc_ada_log_return_ratio_lt_1.435'), bayesian_winrate=0.3567251461988304, profit=51,
    #     #           total=151)
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