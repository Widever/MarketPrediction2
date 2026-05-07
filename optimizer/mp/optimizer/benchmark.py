import os

import numpy as np
import pandas as pd

from mp.optimizer.comb import get_select_combs_mask, profit_loss_in_df
from mp.optimizer.comb_v2 import CombGrade
import datetime as dt
import mp.optimizer.init_data as data
import mp.optimizer.mark as mark

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"optimize_main_dir")
data_dir = os.path.join(data_dir, data.MAIN_SYMBOL)

def _custom_marked_points():
    from_dt = dt.datetime.fromisoformat("2026-05-05T01:40:08.320592")
    to_dt = dt.datetime.fromisoformat("2026-05-07T13:24:39.721091")
    lower_bound_timestamp = int(from_dt.timestamp() * 1000)
    upper_bound_timestamp = int(to_dt.timestamp() * 1000)

    data.init_currency_data_dict("5m", lower_bound_timestamp, upper_bound_timestamp, save_to_file=False)
    # data.init_currency_data_dict_from_online_cache(online_ohlc_dir_, index_, range_)

    data.init_log_return_dict()
    data.init_log_return_ratio_dict()
    data.init_ampl_dict()
    data.init_ampl_ratio_dict()
    data.init_peaks_and_trend_dict()
    data.init_drop_from_high_ratio_dict()
    data.init_rise_from_low_ratio_dict()

    marked_df = mark.mark_data(save_to_file=False)
    return marked_df

def super_benchmark(combs: list[CombGrade] = None, use_inner_combs: bool = False, use_custom_marked_points: bool = False):
    if use_inner_combs:
        if data.MAIN_SYMBOL == "ADAUSDT":
            combs: list[CombGrade] = [
                CombGrade(comb=('#tag_avg_ampl_gt_0.01', '#tag_ada_rise_from_low_lt_0.014', '#tag_ada_log_return_gt_-0.004',
                                '#tag_eth_drop_from_high_gt_0.015', '#tag_doge_ampl_gt_0.009',
                                '#tag_doge_xrp_ampl_ratio_lt_1.701', '#tag_eth_ada_ampl_ratio_lt_1.571',
                                '#tag_btc_eth_log_return_ratio_gt_-1.14', '#tag_doge_ada_drop_from_high_ratio_lt_2.71',
                                '#tag_eth_ada_log_return_ratio_lt_5.468', '#tag_doge_sui_log_return_ratio_lt_4.875',
                                '#tag_avg_log_return_ratio_gt_-0.767', '#tag_eth_doge_ampl_ratio_lt_1.595',
                                '#tag_doge_sui_ampl_ratio_lt_1.633', '#tag_xrp_ampl_gt_0.006'),
                          bayesian_winrate=0.648936170212766, profit=51, total=74),
                CombGrade(comb=('#tag_ada_ampl_gt_0.004', '#tag_doge_ada_log_return_ratio_gt_2.868',
                                '#tag_ada_drop_from_high_lt_0.013', '#tag_doge_sui_ampl_ratio_lt_1.178',
                                '#tag_eth_rise_from_low_lt_0.025', '#tag_eth_ada_ampl_ratio_gt_0.54',
                                '#tag_doge_sui_log_return_ratio_lt_3.042', '#tag_doge_ampl_lt_0.009',
                                '#tag_xrp_rise_from_low_lt_0.052', '#tag_btc_xrp_drop_from_high_ratio_lt_3.327',
                                '#tag_doge_ada_drop_from_high_ratio_lt_7.729', '#tag_avg_rise_from_low_lt_0.043',
                                '#tag_btc_doge_log_return_ratio_lt_2.342', '#tag_avg_log_return_ratio_gt_-3.555',
                                '#tag_avg_ampl_ratio_lt_1.181'), bayesian_winrate=0.3279569892473118, profit=51, total=166),
            ]
        else:
            raise RuntimeError(f"Unknown main symbol: {data.MAIN_SYMBOL}.")

    if use_custom_marked_points:
        marked_points_df = _custom_marked_points()
    else:
        marked_points_df = pd.read_csv(f"{data_dir}/marked_points_frozen.csv")

    profit, loss = profit_loss_in_df(marked_points_df)
    print(f"{profit=}")
    print(f"{loss=}")

    if combs:
        select_mask = get_select_combs_mask(marked_points_df, [comb_.comb for comb_ in combs])

        selected_df = marked_points_df[select_mask].reset_index(drop=True)
    else:
        selected_df = marked_points_df

    if len(selected_df) > 0:
        selected_df["interval"] = pd.cut(selected_df["index"], bins=10)
    else:
        selected_df["interval"] = pd.Series()

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
    # super_benchmark([])
    super_benchmark(use_inner_combs=True, use_custom_marked_points=True)