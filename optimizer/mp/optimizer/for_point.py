import dataclasses
import random

import pandas as pd

import mp.optimizer.init_data as data
import mp.optimizer.mark as mark
from mp.optimizer.comb import CombGrade, get_select_combs_mask


def build_marked_points_df_for_point(symbol: str):
    # ohlcv_df = data.CURRENCY_DATA_DICT[symbol].ohlcv_df
    peaks_and_trend_df = data.PEAKS_AND_TREND_DICT[symbol]
    opened_points = []
    for idx, row in peaks_and_trend_df.iterrows():
        timestamp = int(row["timestamp"])
        point_values_ = mark.point_values(symbol, timestamp)

        new_opened_marked_point = mark.MarkedPoint(
            index=idx,
            timestamp=timestamp,
            values=point_values_,
            sl_price_limit=0,
            sell_price_limit=0,

            rising=row["rising"],
            falling=row["falling"],
            flat=row["flat"],
            peak_up=row["peak_up"],
            peak_down=row["peak_down"],
            change_from_last_peak=row["change_from_last_peak"]
        )
        opened_points.append(new_opened_marked_point)

    marked_points_df_data = []
    for marked_point in sorted(opened_points, key=lambda x: x.index):
        marked_point_dict = dataclasses.asdict(marked_point)
        marked_point_dict_values = marked_point_dict.pop("values")
        marked_point_dict.update(marked_point_dict_values)
        marked_points_df_data.append(marked_point_dict)

    marked_points_df = pd.DataFrame(marked_points_df_data)
    marked_points_df = mark.add_tags_for_point_values(marked_points_df)
    marked_points_df = marked_points_df.round(5)
    return marked_points_df

def check_combs_for_point() -> bool:
    combs: list[CombGrade] = [
        CombGrade(comb=('#tag_btc_eth_log_return_ratio_gt_1.1', '#tag_eth_ada_log_return_ratio_gt_0.8',
                        '#tag_eth_doge_log_return_ratio_lt_0.75', '#tag_eth_ada_amp_ratio_gt_0.9',
                        '#tag_eth_doge_amp_ratio_lt_0.87'), count_=88, sl_count=18, uniformity=0.0, k=3.888888888888889,
                  verify_grade=CombGrade(comb=None, count_=32, sl_count=7, uniformity=None, k=3.5714285714285716,
                                         verify_grade=None)),
        CombGrade(comb=('#tag_btc_doge_log_return_ratio_lt_0.2', '#tag_eth_doge_log_return_ratio_lt_1.0',
                        '#tag_ada_price_up_false', '#tag_btc_doge_amp_ratio_lt_0.36', '#tag_eth_ada_amp_ratio_lt_0.9'),
                  count_=89, sl_count=19, uniformity=0.0, k=3.6842105263157894,
                  verify_grade=CombGrade(comb=None, count_=31, sl_count=7, uniformity=None, k=3.4285714285714284,
                                         verify_grade=None)),
        CombGrade(comb=('#tag_btc_ada_log_return_ratio_gt_0.62', '#tag_eth_doge_log_return_ratio_lt_0.25',
                        '#tag_all_same_price_dir_false', '#tag_btc_eth_amp_ratio_lt_0.65',
                        '#tag_eth_ada_amp_ratio_gt_0.9'), count_=97, sl_count=21, uniformity=0.0, k=3.619047619047619,
                  verify_grade=CombGrade(comb=None, count_=44, sl_count=10, uniformity=None, k=3.4, verify_grade=None)),
        CombGrade(comb=('#tag_btc_eth_log_return_ratio_lt_0.5', '#tag_btc_eth_amp_ratio_gt_0.65',
                        '#tag_btc_doge_amp_ratio_lt_0.36', '#tag_eth_ada_amp_ratio_lt_0.67'), count_=72, sl_count=16,
                  uniformity=0.0, k=3.5,
                  verify_grade=CombGrade(comb=None, count_=32, sl_count=8, uniformity=None, k=3.0, verify_grade=None)),
        CombGrade(comb=('#tag_btc_doge_log_return_ratio_lt_0.2', '#tag_eth_ada_log_return_ratio_lt_0.8',
                        '#tag_btc_price_up_false', '#tag_btc_eth_amp_ratio_gt_0.65', '#tag_btc_ada_amp_ratio_lt_0.6'),
                  count_=89, sl_count=18, uniformity=0.0, k=3.9444444444444446,
                  verify_grade=CombGrade(comb=None, count_=42, sl_count=10, uniformity=None, k=3.2, verify_grade=None)),

    ]

    marked_points_df = build_marked_points_df_for_point("ADAUSDT")

    select_mask = get_select_combs_mask(marked_points_df, [comb_.comb for comb_ in combs])
    selected_df = marked_points_df[select_mask].reset_index(drop=True)

    last_ts = marked_points_df.iloc[-1]["timestamp"]
    is_in_selected = (selected_df["timestamp"] == last_ts).any()

    return is_in_selected