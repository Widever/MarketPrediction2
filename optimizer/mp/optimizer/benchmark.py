import os

import pandas as pd

from mp.optimizer.comb import CombGrade, get_select_combs_mask

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"optimize_main_dir")


def super_benchmark():
    combs: list[CombGrade] = [
        CombGrade(comb=('#tag_btc_doge_amp_ratio_lt_0.25', '#tag_doge_xrp_amp_ratio_lt_0.5',
                        '#tag_eth_ada_log_return_ratio_lt_0.0', '#tag_doge_ada_amp_ratio_gt_0.7',
                        '#tag_doge_xrp_log_return_ratio_gt_0.0', '#tag_eth_ada_amp_ratio_gt_0.48',
                        '#tag_btc_price_up_false', '#tag_btc_ada_amp_ratio_lt_0.3'), count_=18, sl_count=1,
                  uniformity=0.0, k=17.0, verify_grade=None),
        CombGrade(comb=('#tag_btc_eth_amp_ratio_lt_0.39', '#tag_btc_doge_amp_ratio_gt_0.54',
                        '#tag_btc_doge_log_return_ratio_lt_-0.2', '#tag_btc_xrp_amp_ratio_gt_0.36',
                        '#tag_btc_ada_amp_ratio_gt_0.48', '#tag_doge_xrp_log_return_ratio_gt_0.0',
                        '#tag_eth_ada_amp_ratio_gt_0.48', '#tag_eth_doge_amp_ratio_gt_0.5'), count_=17, sl_count=1,
                  uniformity=0.0, k=16.0, verify_grade=None),
        CombGrade(comb=('#tag_doge_sui_log_return_ratio_gt_0.8', '#tag_doge_sui_amp_ratio_lt_0.5',
                        '#tag_btc_eth_amp_ratio_lt_0.39', '#tag_eth_ada_amp_ratio_gt_0.9',
                        '#tag_doge_ada_amp_ratio_gt_0.87', '#tag_btc_xrp_amp_ratio_gt_0.36',
                        '#tag_eth_doge_amp_ratio_gt_0.5', '#tag_eth_xrp_amp_ratio_gt_0.5'), count_=19, sl_count=1,
                  uniformity=0.0, k=18.0, verify_grade=None),
        CombGrade(
            comb=('#tag_btc_ada_amp_ratio_gt_0.48', '#tag_doge_sui_amp_ratio_lt_0.5', '#tag_btc_eth_amp_ratio_lt_0.39',
                  '#tag_btc_doge_amp_ratio_gt_0.36', '#tag_eth_xrp_log_return_ratio_gt_0.25',
                  '#tag_doge_sui_log_return_ratio_lt_0.0', '#tag_btc_xrp_log_return_ratio_gt_-0.2',
                  '#tag_btc_price_up_false'), count_=11, sl_count=1, uniformity=0.0, k=10.0, verify_grade=None),

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