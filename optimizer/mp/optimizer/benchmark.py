import os

import pandas as pd

from mp.optimizer.comb import CombGrade, get_select_combs_mask

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"optimize_main_dir")


def super_benchmark():
    combs: list[CombGrade] = [
        CombGrade(comb=('#tag_btc_eth_log_return_ratio_btc_eth_log_return_ratio_gt_1.1',
                        '#tag_eth_ada_log_return_ratio_eth_ada_log_return_ratio_gt_0.8',
                        '#tag_eth_doge_log_return_ratio_eth_doge_log_return_ratio_lt_0.75',
                        '#tag_eth_ada_amp_ratio_eth_ada_amp_ratio_gt_0.9',
                        '#tag_eth_doge_amp_ratio_eth_doge_amp_ratio_lt_0.87'), count_=88, sl_count=18, uniformity=0.0,
                  k=3.888888888888889,
                  verify_grade=CombGrade(comb=None, count_=32, sl_count=7, uniformity=None, k=3.5714285714285716,
                                         verify_grade=None)),
        CombGrade(comb=('#tag_btc_doge_log_return_ratio_btc_doge_log_return_ratio_lt_0.2',
                        '#tag_eth_doge_log_return_ratio_eth_doge_log_return_ratio_lt_1.0', '#tag_ada_price_up_false',
                        '#tag_btc_doge_amp_ratio_btc_doge_amp_ratio_lt_0.36',
                        '#tag_eth_ada_amp_ratio_eth_ada_amp_ratio_lt_0.9'), count_=89, sl_count=19, uniformity=0.0,
                  k=3.6842105263157894,
                  verify_grade=CombGrade(comb=None, count_=31, sl_count=7, uniformity=None, k=3.4285714285714284,
                                         verify_grade=None)),
        CombGrade(comb=('#tag_btc_ada_log_return_ratio_btc_ada_log_return_ratio_gt_0.62',
                        '#tag_eth_doge_log_return_ratio_eth_doge_log_return_ratio_lt_0.25',
                        '#tag_all_same_price_dir_false', '#tag_btc_eth_amp_ratio_btc_eth_amp_ratio_lt_0.65',
                        '#tag_eth_ada_amp_ratio_eth_ada_amp_ratio_gt_0.9'), count_=97, sl_count=21, uniformity=0.0,
                  k=3.619047619047619,
                  verify_grade=CombGrade(comb=None, count_=44, sl_count=10, uniformity=None, k=3.4, verify_grade=None)),
        CombGrade(comb=('#tag_btc_eth_log_return_ratio_btc_eth_log_return_ratio_lt_0.5',
                        '#tag_btc_eth_amp_ratio_btc_eth_amp_ratio_gt_0.65',
                        '#tag_btc_doge_amp_ratio_btc_doge_amp_ratio_lt_0.36',
                        '#tag_eth_ada_amp_ratio_eth_ada_amp_ratio_lt_0.67'), count_=72, sl_count=16, uniformity=0.0,
                  k=3.5,
                  verify_grade=CombGrade(comb=None, count_=32, sl_count=8, uniformity=None, k=3.0, verify_grade=None)),
        CombGrade(comb=('#tag_btc_doge_log_return_ratio_btc_doge_log_return_ratio_lt_0.2',
                        '#tag_eth_ada_log_return_ratio_eth_ada_log_return_ratio_lt_0.8', '#tag_btc_price_up_false',
                        '#tag_btc_eth_amp_ratio_btc_eth_amp_ratio_gt_0.65',
                        '#tag_btc_ada_amp_ratio_btc_ada_amp_ratio_lt_0.6'), count_=89, sl_count=18, uniformity=0.0,
                  k=3.9444444444444446,
                  verify_grade=CombGrade(comb=None, count_=42, sl_count=10, uniformity=None, k=3.2, verify_grade=None)),

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