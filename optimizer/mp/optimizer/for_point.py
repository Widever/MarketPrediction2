import dataclasses

import pandas as pd

import mp.optimizer.init_data as data
import mp.optimizer.mark as mark
from mp.optimizer.comb import get_select_combs_mask
from mp.optimizer.comb_v2 import CombGrade


def build_marked_points_df_for_point(symbol: str):
    ohlcv_df = data.CURRENCY_DATA_DICT[symbol].ohlcv_df

    marked_points_df_data = []
    for idx, row in ohlcv_df.iterrows():
        timestamp = int(row["timestamp"])
        marked_point_dict = {
            "timestamp": timestamp,
        }

        point_values_ = mark.point_values(symbol, timestamp)
        point_values_dict = dataclasses.asdict(point_values_)
        marked_point_dict.update(point_values_dict)
        marked_points_df_data.append(marked_point_dict)

    marked_points_df = pd.DataFrame(marked_points_df_data)
    marked_points_df = mark.add_tags_for_point_values(marked_points_df)
    marked_points_df = marked_points_df.round(5)
    return marked_points_df

def check_combs_for_point() -> bool:
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


    marked_points_df = build_marked_points_df_for_point("ADAUSDT")

    select_mask = get_select_combs_mask(marked_points_df, [comb_.comb for comb_ in combs])
    selected_df = marked_points_df[select_mask].reset_index(drop=True)

    last_ts = marked_points_df.iloc[-1]["timestamp"]
    is_in_selected = (selected_df["timestamp"] == last_ts).any()

    return is_in_selected