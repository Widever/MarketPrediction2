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

            peak_up=row["peak_up"],
            peak_down=row["peak_down"],
            change_from_last_peak=row["change_from_last_peak"],
            len_from_last_peak=row["len_from_last_peak"],
            last_peak_type=row["last_peak_type"],
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
        CombGrade(comb=('#tag_ada_ampl_gt_0.012', '#tag_avg_drop_from_high_gt_0.03',
                        '#tag_btc_doge_log_return_ratio_lt_0.513', '#tag_eth_ampl_gt_0.008',
                        '#tag_btc_trend_kind_eq_falling', '#tag_doge_xrp_drop_from_high_ratio_lt_1.555',
                        '#tag_btc_xrp_ampl_ratio_lt_0.672', '#tag_eth_ada_ampl_ratio_lt_1.227',
                        '#tag_btc_ada_drop_from_high_ratio_lt_1.57', '#tag_btc_rise_from_low_lt_0.011'),
                  score=0.411522633744856, props={'loss': 93, 'profit': 50, 'total': 143}, verify_grade=None)
    ]


    marked_points_df = build_marked_points_df_for_point("ADAUSDT")

    select_mask = get_select_combs_mask(marked_points_df, [comb_.comb for comb_ in combs])
    selected_df = marked_points_df[select_mask].reset_index(drop=True)

    last_ts = marked_points_df.iloc[-1]["timestamp"]
    is_in_selected = (selected_df["timestamp"] == last_ts).any()

    return is_in_selected