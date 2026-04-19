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

        close_price = float(row["close"])
        low_price = float(row["low"])
        high_price = float(row["high"])
        timestamp = int(row["timestamp"])
        ampl = (high_price - low_price) / low_price

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
            change_from_last_peak=row["change_from_last_peak"],
            len_from_last_peak=row["len_from_last_peak"],
            ampl=ampl
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
        CombGrade(
        comb=('#tag_btc_doge_log_return_ratio_lt_-1.316', '#tag_ada_trend_kind_eq_falling',
              '#tag_sui_trend_ch_from_peak_lt_-0.011', '#tag_avg_log_return_ratio_lt_0.468',
              '#tag_btc_ada_log_return_ratio_lt_0.451', '#tag_avg_ch_from_peak_ratio_gt_-0.098',
              '#tag_eth_doge_log_return_ratio_lt_1.081', '#tag_doge_sui_ch_from_peak_ratio_lt_0.999',
              '#tag_eth_doge_ch_from_peak_ratio_gt_0.59', '#tag_eth_ada_ch_from_peak_ratio_lt_0.904'),
        score=0.35294117647058826, props={'loss': 60, 'profit': 10, 'total': 70}, verify_grade=None),
        CombGrade(comb=('#tag_ada_trend_ch_from_peak_gt_0.048', '#tag_avg_log_return_ratio_lt_0.468',
                        '#tag_xrp_trend_ch_from_peak_gt_0.052', '#tag_btc_doge_ch_from_peak_ratio_lt_0.828',
                        '#tag_sui_trend_ch_from_peak_gt_-0.042', '#tag_btc_ada_ch_from_peak_ratio_gt_-3.617',
                        '#tag_btc_xrp_ch_from_peak_ratio_gt_-3.48', '#tag_eth_ada_ch_from_peak_ratio_gt_-1.563',
                        '#tag_eth_xrp_ch_from_peak_ratio_gt_-2.18', '#tag_doge_ada_ch_from_peak_ratio_gt_-2.031'),
                  score=0.8973063973063973, props={'loss': 11, 'profit': 483, 'total': 494}, verify_grade=None)

    ]


    marked_points_df = build_marked_points_df_for_point("ADAUSDT")

    select_mask = get_select_combs_mask(marked_points_df, [comb_.comb for comb_ in combs])
    selected_df = marked_points_df[select_mask].reset_index(drop=True)

    last_ts = marked_points_df.iloc[-1]["timestamp"]
    is_in_selected = (selected_df["timestamp"] == last_ts).any()

    return is_in_selected