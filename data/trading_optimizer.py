import dataclasses
import itertools
import os
import time
from binance import Client
from dataclasses import dataclass
from typing import get_origin, get_args, Self

import numpy as np
import pandas as pd

import runtime_data as rd
from statistics import mean
import dispersion as dsp
from trading_analyzer import PriceTrendInfo, DispTrendInfo, PriceTrend

DATA_DIR: str | None = None

max_trend_len = 50
flat_trend_limit = 0.02
sl_k = 0.99
sell_k = 1.005
point_value_intervals_n = 4

min_comb_k = 4
min_comb_count = 20
min_verify_comb_k = 4
print_combs_or_look = True

@dataclass(slots=True)
class PointValues:
    in_point_price_ampl: float
    in_point_growth: bool
    in_point_price_abs_change: float
    trend_avg_ampl: float
    trend_avg_ampl_gt_limit: float
    trend_max_ampl: float
    trend_min_ampl: float
    current_trend_kind: str
    current_trend_len: int
    current_trend_abs_change: float
    prev_trend_kind: str
    prev_trend_abs_change: float

    in_point_disp: float
    tail_extreme_disp_ratio: float
    tail_extreme_disp_count: int
    tail_avg_disp: float
    trend_extreme_disp_ratio: float
    trend_extreme_disp_count: int
    trend_avg_disp: float
    trend_avg_disp_gt_limit: float
    trend_max_disp: float
    trend_min_disp: float
    trend_disp_uniformity: float
    trend_disp_growth: bool
    # always_true: bool = True

@dataclass(slots=True)
class MarkedPoint:
    index: int
    timestamp: int
    values: PointValues

    sl_price_limit: float
    sell_price_limit: float

    sl: bool = False
    scope: int = 0

@dataclass(slots=True)
class CombGrade:
    comb: tuple[str, ...] | None
    count_: int
    sl_count: int
    uniformity: float
    k: float
    verify_grade: Self | None = None

class TradingOptimizer:

    def __init__(self):
        ...

    def get_price_trend(self, ohlcv_df: pd.DataFrame, for_index: int) -> PriceTrendInfo:

        end_low_price = ohlcv_df["low"].iat[for_index]
        end_high_price = ohlcv_df["high"].iat[for_index]

        check_last_n_count = max_trend_len
        flat_limit = flat_trend_limit

        lowest_known = end_low_price
        lowest_known_i = for_index

        for i in range(1, check_last_n_count):

            low_price_for_i = ohlcv_df["low"].iat[for_index - i]
            high_price_for_i = ohlcv_df["high"].iat[for_index - i]

            if low_price_for_i < lowest_known:
                lowest_known = low_price_for_i
                lowest_known_i = for_index - i
            else:
                if (high_price_for_i / lowest_known - 1) > flat_limit:
                    break

        highest_known = end_high_price
        highest_known_i = for_index

        for i in range(1, check_last_n_count):

            low_price_for_i =  ohlcv_df["low"].iat[for_index - i]
            high_price_for_i =  ohlcv_df["high"].iat[for_index - i]

            if high_price_for_i > highest_known:
                highest_known = high_price_for_i
                highest_known_i = for_index - i
            else:
                if (highest_known / low_price_for_i - 1) > flat_limit:
                    break

        up_trend_value = end_high_price / lowest_known - 1
        down_trend_value = highest_known / end_low_price - 1

        if up_trend_value > flat_limit > down_trend_value:
            trend = PriceTrend.UP
        elif down_trend_value > flat_limit > up_trend_value:
            trend = PriceTrend.DOWN
        elif up_trend_value < flat_limit and down_trend_value < flat_limit:
            trend = PriceTrend.FLAT
        elif up_trend_value > flat_limit and down_trend_value > flat_limit:
            up_trend_len = for_index - lowest_known_i
            down_trend_len = for_index - highest_known_i

            if up_trend_len == 0 and down_trend_len > 0:
                trend = PriceTrend.DOWN
            elif down_trend_len == 0 and up_trend_len > 0:
                trend = PriceTrend.UP
            elif down_trend_len > 0 and up_trend_len > 0:
                trend = PriceTrend.UP if highest_known_i < lowest_known_i else PriceTrend.DOWN
            elif down_trend_len == 0 and down_trend_len == 0:
                current_open_price = ohlcv_df["open"].iat[for_index]
                current_close_price = ohlcv_df["close"].iat[for_index]
                if current_open_price > current_close_price:
                    trend = PriceTrend.DOWN
                else:
                    trend = PriceTrend.UP
            else:
                raise RuntimeError("Expect all cases covered.")
        else:
            raise RuntimeError("Expect all cases covered.")

        if trend == PriceTrend.UP:
            trend_len = for_index - lowest_known_i + 1
            trend_value  = up_trend_value
        elif trend == PriceTrend.DOWN:
            trend_len = for_index - highest_known_i + 1
            trend_value = -down_trend_value
        elif trend == PriceTrend.FLAT:
            trend_len = check_last_n_count
            trend_value = max(down_trend_value, up_trend_value)
        else:
            raise RuntimeError("Expect all cases covered.")


        trend_start_index = for_index - trend_len + 1
        trend_ohlcva = []

        for trend_i in range(trend_start_index, trend_start_index+trend_len):
            i_open_price = float(ohlcv_df["open"].iat[trend_i])
            i_high_price = float(ohlcv_df["high"].iat[trend_i])
            i_low_price = float(ohlcv_df["low"].iat[trend_i])
            i_close_price = float(ohlcv_df["close"].iat[trend_i])
            i_volume = float(ohlcv_df["volume"].iat[trend_i])
            i_ampl = float(i_high_price / i_low_price - 1)
            trend_ohlcva.append((i_open_price, i_high_price, i_low_price, i_close_price, i_volume, i_ampl))

        ampls = [x[5] for x in trend_ohlcva]
        max_ampl = max(ampls)
        avg_ampl = mean(ampls)

        ampl_gt_limit = [x for x in ampls if x > 0.003]
        avg_ampl_gt_limit = mean(ampl_gt_limit) if ampl_gt_limit else 0.0

        # print(f"Trend: {trend.name}.")
        # print(f"Trend value: {trend_value}.")
        # print(f"Trend len: {trend_len}.")
        # print(f"Max ampl: {max_ampl}.")
        # print(f"Avg ampl: {avg_ampl}.")

        return PriceTrendInfo(
            trend_kind=trend,
            trend_value=trend_value,
            trend_len=trend_len,
            trend_start_index=for_index - trend_len + 1,
            max_ampl=max_ampl,
            avg_ampl=avg_ampl,
            avg_ampl_gt_limit=avg_ampl_gt_limit,
            trend_ohlcva=trend_ohlcva
        )

    def get_disp_trend(self, lower_disp_1, for_index: int, trend_len: int) -> DispTrendInfo:

        disps_on_trend = [
            float(lower_disp_1["disp"].iat[i]) for i in
            range(for_index-trend_len+1, for_index+1)
        ]

        max_disp = max(disps_on_trend)

        normalized_disps = [x/max_disp for x in disps_on_trend]

        high_disp_indexes = [i for i, x in enumerate(normalized_disps) if 0.7 <= x]
        medium_disp_indexes = [i for i, x in enumerate(normalized_disps) if 0.4 <= x < 0.7]
        low_disp_indexes = [i for i, x in enumerate(normalized_disps) if  x < 0.4]

        assert len(high_disp_indexes) + len(medium_disp_indexes) + len(low_disp_indexes) == len(normalized_disps)

        if trend_len > 3:
            start_indexes = [0, 1]
        else:
            start_indexes = [0]

        trend_start_high_disp = None
        for i_ in start_indexes:
            if i_ in high_disp_indexes:
                trend_start_high_disp = i_
                break

        trend_start_medium_disp = None
        if trend_start_high_disp is None:
            for i_ in start_indexes:
                if i_ in medium_disp_indexes:
                    trend_start_medium_disp = i_
                    break

        start_middle_index = trend_start_high_disp + 1 if trend_start_high_disp is not None else 0
        end_middle_index = len(normalized_disps) - 1

        middle_indexes = [x for x in range(start_middle_index, end_middle_index)]

        middle_indexes_high_disp = [x for x in middle_indexes if x in high_disp_indexes]
        middle_indexes_medium_disp = [x for x in middle_indexes if x in medium_disp_indexes]
        middle_indexes_low_disp = [x for x in middle_indexes if x in low_disp_indexes]

        end_index = len(normalized_disps) - 1
        if end_index in high_disp_indexes:
            trend_end_disp = "high"
        elif end_index in medium_disp_indexes:
            trend_end_disp = "medium"
        elif end_index in low_disp_indexes:
            trend_end_disp = "low"
        else:
            raise RuntimeError("Expect all cases covered.")


        # print(f"Disps: {[round(float(x), 5) for x in disps_on_trend]}")
        # print(f"Trend start high disp: {trend_start_high_disp}")
        # print(f"Trend start medium disp: {trend_start_medium_disp}")
        # print(f"High disp on middle, {len(middle_indexes_high_disp)}: {middle_indexes_high_disp}")
        # print(f"Medium disp on middle, {len(middle_indexes_medium_disp)}: {middle_indexes_medium_disp}")
        # print(f"Low disp on middle, {len(middle_indexes_low_disp)}: {middle_indexes_low_disp}")
        # print(f"End disp: {trend_end_disp}.")

        monotone_up = False
        avg_disp_change = 0
        max_disp_change = 0
        if len(disps_on_trend) > 1:
            disp_changes = [disps_on_trend[i] - disps_on_trend[i-1] for i in range(1, len(disps_on_trend))]
            if all(x > 0 for x in disp_changes):
                monotone_up = True

            avg_disp_change = mean(max(abs(x), 0.1) for x in disp_changes)
            max_disp_change = max(disp_changes, key=lambda x: abs(x))

        # print(f"{monotone_up=}")
        # print(f"{avg_disp_change=}")
        # print(f"{max_disp_change=}")

        is_lightning = trend_start_medium_disp is not None and trend_end_disp == "high" and not middle_indexes_high_disp and end_index - trend_start_medium_disp > 1
        is_saddle = trend_start_high_disp is not None and trend_end_disp == "high" and not middle_indexes_high_disp and end_index - trend_start_high_disp > 1

        # print(f"{is_lightning=}")
        # print(f"{is_saddle=}")

        disps_greater_limit = [x for x in disps_on_trend if x > 0.11]
        avg_disp_greater_limit = mean(disps_greater_limit) if disps_greater_limit else 0
        max_disp = max(disps_on_trend)
        min_disp = min(disps_on_trend)

        # print(f"{max_disp=}")
        # print(f"{avg_disp_greater_limit=}")

        return DispTrendInfo(
            disps_on_trend=disps_on_trend,
            trend_start_high_disp=trend_start_high_disp,
            trend_start_medium_disp=trend_start_medium_disp,
            high_disp_on_middle=middle_indexes_high_disp,
            medium_disp_on_middle=middle_indexes_medium_disp,
            low_disp_on_middle=middle_indexes_low_disp,
            end_disp_level=trend_end_disp,
            monotone_up=monotone_up,
            avg_disp_change=avg_disp_change,
            max_disp_change=max_disp_change,
            avg_disp_greater_limit=avg_disp_greater_limit,
            max_disp=max_disp,
            min_disp=min_disp,
            is_lightning=is_lightning,
            is_saddle=is_saddle
        )

    def get_last_trends(self, ohlcv_df, for_index: int, n: int = 5) -> list[PriceTrendInfo]:
        infos = []
        last_start_i = for_index
        for _ in range(n):
            trend_info = self.get_price_trend(ohlcv_df, for_index=last_start_i)
            infos.append(trend_info)

            last_start_i = trend_info.trend_start_index

            if trend_info.trend_len == 1:
                last_start_i -= 1

        infos = list(reversed(infos))
        return infos

    def point_values(self, ohlcv_df: pd.DataFrame, lower_disp_1, idx: int) -> PointValues:
        price_trend_info = self.get_price_trend(ohlcv_df, idx)
        disp_trend_info = self.get_disp_trend(lower_disp_1, idx, price_trend_info.trend_len)
        last_trends_info = self.get_last_trends(ohlcv_df, idx)

        AMPL_LIMIT = 0.002
        TREND_DISP_LIMIT = 0.01

        _in_point_open = price_trend_info.trend_ohlcva[-1][0]
        _in_point_close = price_trend_info.trend_ohlcva[-1][3]
        _in_point_ampl = price_trend_info.trend_ohlcva[-1][5]
        _in_point_price_change = _in_point_close / _in_point_open - 1
        _trend_ampls = [x[5] for x in price_trend_info.trend_ohlcva]
        _trend_ampls_gt_limit = [x for x in _trend_ampls if x > AMPL_LIMIT]
        _tail_disps = [
            float(lower_disp_1["disp"].iat[i]) for i in
            range(idx - 13, idx + 1)
        ]
        _tail_extreme_disps = [x for x in _tail_disps if x > 0.88]
        _trend_disps = disp_trend_info.disps_on_trend
        _trend_extreme_disps = [x for x in _trend_disps if x > 0.88]
        _trend_disps_gt_limit = [x for x in _trend_disps if x > TREND_DISP_LIMIT]

        _trend_disp_first_part = _trend_disps[:int(len(_trend_disps)/2)]
        _trend_disp_second_part = _trend_disps[int(len(_trend_disps)/2):]
        _trend_disp_growth = mean(_trend_disp_second_part) > mean(_trend_disp_first_part) * 1.2 if len(_trend_disps) > 2 else False

        assert lower_disp_1["disp"].iat[idx] == _tail_disps[-1]

        return PointValues(
            in_point_price_ampl=_in_point_ampl,
            in_point_growth = _in_point_price_change > 0,
            in_point_price_abs_change = abs(_in_point_price_change),
            trend_avg_ampl = mean(_trend_ampls),
            trend_avg_ampl_gt_limit = mean(_trend_ampls_gt_limit) if _trend_ampls_gt_limit else 0,
            trend_max_ampl = max(_trend_ampls),
            trend_min_ampl = min(_trend_ampls),
            current_trend_kind = price_trend_info.trend_kind.name,
            current_trend_len = price_trend_info.trend_len,
            current_trend_abs_change = abs(price_trend_info.trend_value),
            prev_trend_kind = last_trends_info[-2].trend_kind.name,
            prev_trend_abs_change = abs(last_trends_info[-2].trend_value),

            in_point_disp = _tail_disps[-1],
            tail_extreme_disp_ratio = len(_tail_extreme_disps) / len(_tail_disps),
            tail_extreme_disp_count = len(_tail_extreme_disps),
            tail_avg_disp = mean(_tail_disps),
            trend_extreme_disp_ratio = len(_trend_extreme_disps) / len(_trend_disps),
            trend_extreme_disp_count = len(_trend_extreme_disps),
            trend_avg_disp = mean(_trend_disps),
            trend_avg_disp_gt_limit = mean(_trend_disps_gt_limit) if _trend_disps_gt_limit else 0,
            trend_max_disp = max(_trend_disps),
            trend_min_disp = min(_trend_disps),
            trend_disp_uniformity = 1.0,
            trend_disp_growth = _trend_disp_growth,
        )

    def _add_interval_tag_columns(self, df: pd.DataFrame, col: str, n_intervals: int, min_max = None) -> pd.DataFrame:
        # Get min and max
        if min_max is None:
            min_val, max_val = df[col].min(), df[col].max()
        else:
            min_val, max_val = min_max

        # Build intervals
        bins = np.linspace(min_val, max_val, n_intervals + 1)

        for i in range(n_intervals):
            left, right = bins[i], bins[i + 1]

            # Last interval include right
            if i == n_intervals - 1:
                mask = (df[col] >= left)
            else:
                mask = (df[col] >= left) & (df[col] < right)

            df[f"#tag_{col}_int{i + 1}"] = mask

        return df

    def _add_str_tag_columns(self, df: pd.DataFrame, col: str, values=None) -> pd.DataFrame:

        if values is None:
            unique_values = df[col].unique()
        else:
            unique_values = values

        for val in unique_values:
            df[f"{col}_{val}"] = df[col] == val
        return df

    def _add_bool_tag_columns(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        df[f"#tag_{col}_true"] = df[col]
        df[f"#tag_{col}_false"] = ~df[col]
        return df

    def add_tags_for_point_values(self, marked_points_values_df: pd.DataFrame, use_marked_data_tags=False) -> pd.DataFrame:

        if use_marked_data_tags:
            marked_data_df = pd.read_csv(f"{DATA_DIR}/marked_points_frozen.csv")
        else:
            marked_data_df = None

        for f in dataclasses.fields(PointValues):
            typ = f.type
            # unwrap Optional[...]
            if get_origin(typ) is not None and get_origin(typ) is not list:
                args = get_args(typ)
                if args:
                    typ = args[0]

            if typ is str:
                if use_marked_data_tags:
                    values = marked_data_df[f.name].unique()
                    marked_points_values_df = self._add_str_tag_columns(marked_points_values_df, f.name, values=values)
                else:
                    marked_points_values_df = self._add_str_tag_columns(marked_points_values_df, f.name)
            elif typ in (int, float):
                n_intervals = f.metadata.get("intervals", point_value_intervals_n)
                if use_marked_data_tags:
                    min_max = (marked_data_df[f.name].min(), marked_data_df[f.name].max())
                    marked_points_values_df = self._add_interval_tag_columns(marked_points_values_df, f.name, n_intervals, min_max=min_max)
                else:
                    marked_points_values_df = self._add_interval_tag_columns(marked_points_values_df, f.name, n_intervals)
            elif typ is bool:
                marked_points_values_df = self._add_bool_tag_columns(marked_points_values_df, f.name)

        return marked_points_values_df


    def mark_data(self):
        start_time = time.time()
        closed_points: list[MarkedPoint] = []
        symbol = "ADAUSDT"

        ohlcv_df = rd.CURRENCY_DATAS[symbol].ohlcv_df
        lower_disp_1 = dsp.get_disp_1_lower()
        opened_points: list[MarkedPoint] = []

        df_len = len(ohlcv_df)
        print(f"OHLCV df len = {df_len}. Start marking data...")
        for idx, row in ohlcv_df.iterrows():
            if idx % 10000 == 0:
                print(f"Marked {idx}/{df_len} points.")

            if idx < 50:
                continue
            close_price = float(row["close"])
            low_price = float(row["low"])
            high_price = float(row["high"])
            timestamp = int(row["timestamp"])

            remaining_opened_points = []
            for opened_point in opened_points:
                if opened_point.sl_price_limit >= low_price:
                    opened_point.sl = True
                    closed_points.append(opened_point)
                elif opened_point.sell_price_limit <= high_price:
                    opened_point.sl = False
                    closed_points.append(opened_point)
                else:
                    opened_point.scope += 1
                    remaining_opened_points.append(opened_point)

            opened_points = remaining_opened_points

            point_values = self.point_values(ohlcv_df, lower_disp_1, idx)

            sl_limit_price = sl_k * close_price
            sell_limit_price = sell_k * close_price

            new_opened_marked_point = MarkedPoint(
                index=idx,
                timestamp=timestamp,
                values=point_values,
                sl_price_limit=sl_limit_price,
                sell_price_limit=sell_limit_price
            )
            opened_points.append(new_opened_marked_point)

        marked_points_df_data = []
        for marked_point in sorted(closed_points, key=lambda x: x.index):
            marked_point_dict = dataclasses.asdict(marked_point)
            marked_point_dict_values = marked_point_dict.pop("values")
            marked_point_dict.update(marked_point_dict_values)
            marked_points_df_data.append(marked_point_dict)

        marked_points_df = pd.DataFrame(marked_points_df_data)
        marked_points_df = self.add_tags_for_point_values(marked_points_df)
        marked_points_df = marked_points_df.round(5)
        marked_points_df.to_csv(f"{DATA_DIR}/marked_points_frozen.csv", index=False)

        end_time = time.time()
        print(f"Elapsed time: {end_time-start_time}s.")
        return marked_points_df

    def mark_data_in_point(self):
        symbol = "ADAUSDT"
        ohlcv_df = rd.CURRENCY_DATAS[symbol].ohlcv_df
        lower_disp_1 = dsp.get_disp_1_lower()

        idx = len(ohlcv_df) - 1
        point_values = self.point_values(ohlcv_df, lower_disp_1, idx)
        print(f"Current point values: {point_values}.")

        marked_point = MarkedPoint(
            index=idx,
            timestamp=ohlcv_df["timestamp"].iat[idx],
            values=point_values,
            sl_price_limit=0.0,
            sell_price_limit=0.0,
        )

        marked_points_df_data = []
        marked_point_dict = dataclasses.asdict(marked_point)
        marked_point_dict_values = marked_point_dict.pop("values")
        marked_point_dict.update(marked_point_dict_values)
        marked_points_df_data.append(marked_point_dict)

        marked_points_df = pd.DataFrame(marked_points_df_data)
        marked_points_df = self.add_tags_for_point_values(marked_points_df, use_marked_data_tags=True)
        marked_points_df = marked_points_df.round(5)

        marked_points_df_tail = marked_points_df.tail(n=5)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)

        print(" ========== Current point marked df:")
        print(marked_points_df_tail)
        print()

        return marked_points_df

    def check_combs_in_point(self) -> bool:
        combs: list[CombGrade] = [
            CombGrade(comb=('#tag_tail_extreme_disp_ratio_int1', '#tag_trend_avg_disp_int4'), count_=23, sl_count=3,
                      uniformity=0.5652173913043479, k=6.666666666666667,
                      verify_grade=CombGrade(comb=None, count_=9, sl_count=2, uniformity=None, k=3.5,
                                             verify_grade=None)),
            CombGrade(comb=('#tag_trend_avg_disp_int2', '#tag_trend_avg_disp_gt_limit_int3'), count_=20, sl_count=4,
                      uniformity=0.7, k=4.0,
                      verify_grade=CombGrade(comb=None, count_=8, sl_count=2, uniformity=None, k=3.0,
                                             verify_grade=None)),
            CombGrade(comb=('#tag_in_point_disp_int4', '#tag_trend_max_disp_int3'), count_=28, sl_count=8,
                      uniformity=0.7857142857142857, k=2.5,
                      verify_grade=CombGrade(comb=None, count_=12, sl_count=3, uniformity=None, k=3.0,
                                             verify_grade=None)),
            CombGrade(comb=('#tag_current_trend_len_int4', '#tag_tail_avg_disp_int3'), count_=66, sl_count=19,
                      uniformity=0.7272727272727273, k=2.473684210526316,
                      verify_grade=CombGrade(comb=None, count_=25, sl_count=7, uniformity=None, k=2.5714285714285716,
                                             verify_grade=None)),
        ]

        prep_combs = [comb_.comb for comb_ in combs]

        marked_data_in_point = self.mark_data_in_point()
        idx = len(marked_data_in_point) - 1
        for comb in prep_combs:
            comb_true = all([bool(marked_data_in_point[comb_tag].iat[idx]) for comb_tag in comb])
            if comb_true:
                print(f"!!! Current point match comb: {comb}")
                return True

        return False

    def get_select_combs_mask(self, marked_points_df: pd.DataFrame, combs: list[tuple[str, ...]]) -> pd.Series:

        n = len(marked_points_df)

        combs_masks = [
            marked_points_df[list(comb)].astype(bool).all(axis=1)
            for comb in combs
        ]

        any_comb = np.logical_or.reduce(combs_masks) if combs_masks else np.zeros(n, dtype=bool)

        mask = np.zeros(n, dtype=bool)
        skip = 0
        scopes = marked_points_df["scope"].to_numpy()

        for i in range(n):
            if skip > 0:
                skip -= 1
                continue

            if any_comb[i]:
                mask[i] = True
                skip = int(scopes[i])

        return pd.Series(mask, index=marked_points_df.index)

    def get_exclude_combs_mask(self, marked_points_df: pd.DataFrame, combs: list[tuple[str, ...]]) -> pd.Series:

        n = len(marked_points_df)

        combs_masks = [
            marked_points_df[list(comb)].astype(bool).all(axis=1)
            for comb in combs
        ]

        any_comb = np.logical_or.reduce(combs_masks) if combs_masks else np.zeros(n, dtype=bool)

        mask = np.ones(n, dtype=bool)
        skip = 0
        scopes = marked_points_df["scope"].to_numpy()

        for i in range(n):
            if skip > 0:
                skip -= 1
                mask[i] = False
                continue

            if any_comb[i]:
                mask[i] = False
                skip = int(scopes[i])

        return pd.Series(mask, index=marked_points_df.index)

    def _comb_uniformity(self, comb_df, interval_bins) -> float:
        comb_df_ = pd.DataFrame()
        comb_df_["index"] = comb_df["index"]
        comb_df_["sl"] = comb_df["sl"]
        comb_df_["interval"] = pd.cut(comb_df_["index"], bins=interval_bins)

        intervals_stat = comb_df_.groupby("interval", observed=False).agg(
            count_=("index", "size"),
            sl_count_=("sl", "sum"),
        ).reset_index()

        total_count = intervals_stat["count_"].sum()
        intervals_stat["prop_count"] = intervals_stat["count_"] / total_count
        max_prop_count = intervals_stat["prop_count"].max()

        return float(1 - max_prop_count)

    def _comb_k(self, comb_df) -> float:
        count_ = len(comb_df)
        sl_count = int(comb_df["sl"].sum())
        k = (count_ - sl_count) / sl_count if sl_count > 0 else sl_count
        return k

    def grade_comb(self, comb_df: pd.DataFrame, comb: tuple[str, ...], interval_bins) -> CombGrade:
        count_ = len(comb_df)
        sl_count = int(comb_df["sl"].sum())
        uniformity = self._comb_uniformity(comb_df, interval_bins)
        k = self._comb_k(comb_df)

        return CombGrade(
            comb=comb,
            count_=count_,
            sl_count=sl_count,
            uniformity=uniformity,
            k=k
        )

    def choose_comb(self, train_marked_points_df: pd.DataFrame, verify_marked_points_df: pd.DataFrame, all_combs: list[tuple[str, ...]], selected_combs: list[CombGrade], interval_bins) -> CombGrade:

        print(f"Selected combs count: {len(selected_combs)}.")
        if selected_combs:
            exclude_mask = self.get_exclude_combs_mask(train_marked_points_df, [selected_comb.comb for selected_comb in selected_combs])
            train_marked_points_df = train_marked_points_df[exclude_mask].reset_index(drop=True)

        print(f"Df len after exclude selected: {len(train_marked_points_df)}.")

        comb_grades = []
        for comb in all_combs:
            start_time = time.time()
            select_mask = self.get_select_combs_mask(train_marked_points_df, [comb])
            end_time = time.time()
            # print(f"Get mask elapsed time: {end_time-start_time}")

            start_time = time.time()
            comb_df = train_marked_points_df[select_mask]
            end_time = time.time()
            # print(f"Apply mask elapsed time: {end_time-start_time}")

            comb_grade = self.grade_comb(comb_df, comb, interval_bins)
            comb_grades.append(comb_grade)

        comb_grades_sorted: list[CombGrade] = list(sorted(comb_grades, key=lambda x: x.k, reverse=True))

        i = 0

        for comb_grade in comb_grades_sorted:

            if comb_grade.k < min_comb_k:
                continue

            if comb_grade.count_ < min_comb_count:
                continue

            #
            # if comb_grade.uniformity < 0.6:
            #     continue

            # Check overlearning
            if selected_combs:
                verify_exclude_mask = self.get_exclude_combs_mask(verify_marked_points_df, [selected_comb.comb for selected_comb in selected_combs])
                verify_marked_points_df = verify_marked_points_df[verify_exclude_mask].reset_index(drop=True)

            verify_select_mask = self.get_select_combs_mask(verify_marked_points_df, [comb_grade.comb])
            verify_comb_df = verify_marked_points_df[verify_select_mask]

            verify_comb_grade = self.grade_comb(verify_comb_df, comb_grade.comb, interval_bins)

            # if verify_comb_grade.count_ < 5:
            #     continue
            #
            if verify_comb_grade.k < min_verify_comb_k:
                continue

            if verify_comb_grade.count_ < (comb_grade.count_ / 2.7):
                continue

            verify_comb_grade.comb = None
            comb_grade.verify_grade = verify_comb_grade
            if print_combs_or_look:
                print(comb_grade)
                if i > 1000:
                    break
                i += 1
            else:
                return comb_grade

        raise RuntimeError("Not found comb.")

    def optimal_combs(self, limit_comb_n = 10, selected_combs = None) -> list[CombGrade]:
        full_time_start = time.time()
        train_marked_points_df = pd.read_csv(f"{DATA_DIR}/marked_points_train.csv")
        verify_marked_points_df = pd.read_csv(f"{DATA_DIR}/marked_points_verify.csv")

        interval_bins = pd.cut(train_marked_points_df["index"], bins=12).cat.categories
        print(f"Full df len: {len(train_marked_points_df)}")

        tags = [x for x in train_marked_points_df.columns if x.startswith("#tag_")]
        combinations = list(itertools.combinations(tags, 1))
        combinations += list(itertools.combinations(tags, 2))
        # combinations += list(itertools.combinations(tags, 3))
        combs: list[tuple[str, ...]] = combinations

        print(f"All combs len: {len(combs)}")

        if selected_combs is None:
            selected_combs: list[CombGrade] = []

        while True:
            try:
                if len(selected_combs) > limit_comb_n - 1:
                    break

                print("Start choosing comb...")
                start_time = time.time()

                comb_grade = self.choose_comb(train_marked_points_df, verify_marked_points_df, combs, selected_combs, interval_bins)
                selected_combs.append(comb_grade)

                print("Chosen comb:")
                print(comb_grade)

                end_time = time.time()
                print(f"For this comb elapsed {end_time-start_time}s.")

            except RuntimeError as e:
                print(e)
                break


        print(f"Selected {len(selected_combs)} combs:")
        for comb in selected_combs:
            print(f"{str(comb)},")
        print()

        full_time_end = time.time()

        print(f"Total elapsed time: {full_time_end-full_time_start}s.")

        return selected_combs

    def super_benchmark(self):

        combs: list[CombGrade] = [
            CombGrade(comb=('#tag_tail_extreme_disp_ratio_int1', '#tag_trend_avg_disp_int4'), count_=23, sl_count=3,
                      uniformity=0.5652173913043479, k=6.666666666666667,
                      verify_grade=CombGrade(comb=None, count_=9, sl_count=2, uniformity=None, k=3.5,
                                             verify_grade=None)),
            CombGrade(comb=('#tag_trend_avg_disp_int2', '#tag_trend_avg_disp_gt_limit_int3'), count_=20, sl_count=4,
                      uniformity=0.7, k=4.0,
                      verify_grade=CombGrade(comb=None, count_=8, sl_count=2, uniformity=None, k=3.0,
                                             verify_grade=None)),
            CombGrade(comb=('#tag_in_point_disp_int4', '#tag_trend_max_disp_int3'), count_=28, sl_count=8,
                      uniformity=0.7857142857142857, k=2.5,
                      verify_grade=CombGrade(comb=None, count_=12, sl_count=3, uniformity=None, k=3.0,
                                             verify_grade=None)),
            CombGrade(comb=('#tag_current_trend_len_int4', '#tag_tail_avg_disp_int3'), count_=66, sl_count=19,
                      uniformity=0.7272727272727273, k=2.473684210526316,
                      verify_grade=CombGrade(comb=None, count_=25, sl_count=7, uniformity=None, k=2.5714285714285716,
                                             verify_grade=None)),
        ]

        marked_points_df = pd.read_csv(f"{DATA_DIR}/marked_points_frozen.csv")

        select_mask = self.get_select_combs_mask(marked_points_df, [comb_.comb for comb_ in combs])

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

    def split_marked_data(self):
        marked_points_df = pd.read_csv(f"{DATA_DIR}/marked_points_frozen.csv")
        marked_points_df = marked_points_df.round(5)
        train_set_size = int(len(marked_points_df) * 0.7)
        first = marked_points_df.iloc[:train_set_size].copy()
        second = marked_points_df.iloc[train_set_size:].copy()

        first.to_csv(f"{DATA_DIR}/marked_points_train.csv", index=False)
        second.to_csv(f"{DATA_DIR}/marked_points_verify.csv", index=False)

if __name__ == "__main__":
    interval = Client.KLINE_INTERVAL_5MINUTE
    # rd.init_runtime_data(interval)
    # rd.init_runtime_data_from_cache(interval)
    optimizer = TradingOptimizer()

    flat_trend_limit = 0.02
    sl_k = 0.99
    sell_k = 1.005
    point_value_intervals_n = 4
    DATA_DIR = f"optimize_5m_interval_exp"
    min_comb_k = 2.3
    min_verify_comb_k = 2.3
    min_comb_count = 20
    print_combs_or_look = False


    os.makedirs(DATA_DIR, exist_ok=True)
    # opt = optimizer.mark_data()
    # opt = optimizer.split_marked_data()
    selected_combs = [
        # CombGrade(comb=('#tag_current_trend_len_int2', '#tag_in_point_disp_int3', '#tag_trend_disp_growth_true'), count_=94, sl_count=26, uniformity=0.8404255319148937, k=2.6153846153846154, verify_grade=CombGrade(comb=None, count_=48, sl_count=12, uniformity=None, k=3.0, verify_grade=None)),
        # CombGrade(comb=('#tag_current_trend_len_int2', '#tag_trend_avg_disp_int2'), count_=51, sl_count=12, uniformity=0.8431372549019608, k=3.25, verify_grade=CombGrade(comb=None, count_=30, sl_count=7, uniformity=None, k=3.2857142857142856, verify_grade=None)),
        # CombGrade(comb=('#tag_in_point_disp_int1', '#tag_tail_extreme_disp_count_int2', '#tag_trend_disp_growth_true'), count_=20, sl_count=6, uniformity=0.7, k=2.3333333333333335, verify_grade=CombGrade(comb=None, count_=10, sl_count=1, uniformity=None, k=9.0, verify_grade=None))
        # CombGrade(comb=('#tag_tail_extreme_disp_count_int1', '#tag_trend_avg_disp_gt_limit_int4'), count_=26, sl_count=5, uniformity=0.5384615384615384, k=4.2, verify_grade=CombGrade(comb=None, count_=10, sl_count=2, uniformity=None, k=4.0, verify_grade=None)),

    ]
    # opt = optimizer.optimal_combs(12, selected_combs)
    opt = optimizer.super_benchmark()
    # print(res)