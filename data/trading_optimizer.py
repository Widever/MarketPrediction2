import dataclasses
import itertools
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import runtime_data as rd
from statistics import mean
import dispersion as dsp
from trading_analyzer import PriceTrendInfo, PriceTrend, DispTrendInfo, LastTrendsInfo


@dataclass(slots=True)
class PointAttrs:
    extreme_disp_many: bool = False
    extreme_disp_moderate: bool = False
    extreme_disp_few: bool = False
    avg_ampl_gt_limit_gt_0_03: bool = False
    avg_ampl_gt_limit_gt_0_013: bool = False
    avg_ampl_gt_limit_gt_0_007: bool = False
    avg_ampl_gt_limit_lt: bool = False
    avg_disp_tail_gt_05: bool = False
    avg_disp_tail_gt_03: bool = False
    avg_disp_tail_gt_02: bool = False
    avg_disp_tail_lt: bool = False
    down_strick_3: bool = False
    down_strick_2: bool = False
    down_strick_1: bool = False
    down_strick_0: bool = False
    up_strick_3: bool = False
    up_strick_2: bool = False
    up_strick_1: bool = False
    up_strick_0: bool = False
    trend_len_long: bool = False
    trend_len_middle: bool = False
    trend_len_short: bool = False
    trend_max_ampl_gt_0_05: bool = False
    trend_max_ampl_gt_0_03: bool = False
    trend_max_ampl_gt_0_01: bool = False
    trend_max_ampl_lt: bool = False
    trend_kind_up: bool = False
    trend_kind_down: bool = False
    trend_kind_flat: bool = False
    trend_kind_unknown: bool = False
    prev_t_kind_down: bool = False
    prev_t_kind_up: bool = False
    max_disp_change_gt_05: bool = False
    max_disp_change_gt_03: bool = False
    max_disp_change_gt_02: bool = False
    max_disp_change_lt: bool = False
    disp_monotone_up_true: bool = False
    disp_monotone_up_false: bool = False
    disp_avg_change_gt_05: bool = False
    disp_avg_change_gt_03: bool = False
    disp_avg_change_lt: bool = False
    max_disp_gt_06: bool = False
    max_disp_gt_04: bool = False
    max_disp_gt_02: bool = False
    max_disp_lt: bool = False
    min_disp_gt_03: bool = False
    min_disp_gt_02: bool = False
    min_disp_gt_01: bool = False
    min_disp_lt: bool = False
    current_disp_gt_05: bool = False
    current_disp_gt_03: bool = False
    current_disp_gt_02: bool = False
    current_disp_gt_01: bool = False
    current_disp_lt: bool = False

@dataclass(slots=True)
class MarkedPoint:
    index: int
    timestamp: int
    attrs: PointAttrs

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

class TradingOptimizer:

    def __init__(self):
        ...

    def get_price_trend(self, ohlcv_df: pd.DataFrame, for_index: int) -> PriceTrendInfo:

        end_low_price = ohlcv_df["low"].iat[for_index]
        end_high_price = ohlcv_df["high"].iat[for_index]

        check_last_n_count = 50
        flat_limit = 0.009

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
            i_open_price = int(ohlcv_df["open"].iat[trend_i])
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

    def get_last_trends(self, ohlcv_df, for_index: int, n: int = 5) -> LastTrendsInfo:
        info = []
        last_start_i = for_index
        for _ in range(n):
            trend_info = self.get_price_trend(ohlcv_df, for_index=last_start_i)
            volume = trend_info.trend_value
            info.append((volume, trend_info.trend_len))

            last_start_i = trend_info.trend_start_index

            if trend_info.trend_len == 1:
                last_start_i -= 1

        info = list(reversed(info))
        # msg = ", ".join([f"{round(v, 4)}/{l}" for v, l in info])
        # print(f"Last {n} trend volumes: {msg}.")

        sum_volume_of_last_3 = sum(abs(x[0]) for x in info[-3:])
        sum_last_2 = abs(sum(x[0] for x in info[-2:]))
        # print(f"Last 3 trends volume sum: {sum_volume_of_last_3}")
        return LastTrendsInfo(
            volume_and_len=info,
            sum_volume_of_last_3=sum_volume_of_last_3,
            sum_last_2=sum_last_2
        )

    def create_point_attrs(self, ohlcv_df: pd.DataFrame, lower_disp_1, idx: int):
        price_trend_info = self.get_price_trend(ohlcv_df, idx)
        disp_trend_info = self.get_disp_trend(lower_disp_1, idx, price_trend_info.trend_len)
        last_trends_info = self.get_last_trends(ohlcv_df, idx)

        disp_tail = [
            float(lower_disp_1["disp"].iat[i]) for i in
            range(idx - 13, idx + 1)
        ]

        assert lower_disp_1["disp"].iat[idx] == disp_tail[-1]

        avg_disp_tail = mean(disp_tail)
        extreme_disp = len([x for x in disp_tail if x > 0.88])

        point_attrs = PointAttrs()

        if extreme_disp > 4:
            point_attrs.extreme_disp_many = True
        elif extreme_disp > 1:
            point_attrs.extreme_disp_moderate = True
        else:
            point_attrs.extreme_disp_few = True

        if price_trend_info.avg_ampl_gt_limit > 0.03:
            point_attrs.avg_ampl_gt_limit_gt_0_03 = True
        elif price_trend_info.avg_ampl_gt_limit > 0.013:
            point_attrs.avg_ampl_gt_limit_gt_0_013 = True
        elif price_trend_info.avg_ampl_gt_limit > 0.007:
            point_attrs.avg_ampl_gt_limit_gt_0_007 = True
        else:
            point_attrs.avg_ampl_gt_limit_lt = True

        if avg_disp_tail > 0.5:
            point_attrs.avg_disp_tail_gt_05 = True
        elif avg_disp_tail > 0.3:
            point_attrs.avg_disp_tail_gt_03 = True
        elif avg_disp_tail > 0.2:
            point_attrs.avg_disp_tail_gt_02 = True
        else:
            point_attrs.avg_disp_tail_lt = True

        if last_trends_info.volume_and_len[-1][0] < 0 and last_trends_info.volume_and_len[-2][0] < 0 and \
                last_trends_info.volume_and_len[-3][0] < 0:
            point_attrs.down_strick_3 = True
        elif last_trends_info.volume_and_len[-1][0] < 0 and last_trends_info.volume_and_len[-2][0] < 0:
            point_attrs.down_strick_2 = True
        elif last_trends_info.volume_and_len[-1][0] < 0:
            point_attrs.down_strick_1 = True
        else:
            point_attrs.down_strick_0 = True

        if last_trends_info.volume_and_len[-1][0] > 0 and last_trends_info.volume_and_len[-2][0] > 0 and \
                last_trends_info.volume_and_len[-3][0] > 0:
            point_attrs.up_strick_3 = True
        elif last_trends_info.volume_and_len[-1][0] > 0 and last_trends_info.volume_and_len[-2][0] > 0:
            point_attrs.up_strick_2 = True
        elif last_trends_info.volume_and_len[-1][0] > 0:
            point_attrs.up_strick_1 = True
        else:
            point_attrs.up_strick_0 = True

        if price_trend_info.trend_len > 12:
            point_attrs.trend_len_long = True
        elif price_trend_info.trend_len > 5:
            point_attrs.trend_len_middle = True
        else:
            point_attrs.trend_len_short = True

        if price_trend_info.max_ampl > 0.05:
            point_attrs.trend_max_ampl_gt_0_05 = True
        elif price_trend_info.max_ampl > 0.03:
            point_attrs.trend_max_ampl_gt_0_03 = True
        elif price_trend_info.max_ampl > 0.01:
            point_attrs.trend_max_ampl_gt_0_01 = True
        else:
            point_attrs.trend_max_ampl_lt = True

        if price_trend_info.trend_kind == PriceTrend.UP:
            point_attrs.trend_kind_up = True
        elif price_trend_info.trend_kind == PriceTrend.DOWN:
            point_attrs.trend_kind_down = True
        elif price_trend_info.trend_kind == PriceTrend.FLAT:
            point_attrs.trend_kind_flat = True
        else:
            point_attrs.trend_kind_unknown = True

        if last_trends_info.volume_and_len[-2][0] < 0:
            point_attrs.prev_t_kind_down = True
        else:
            point_attrs.prev_t_kind_up = True

        if disp_trend_info.max_disp_change > 0.5:
            point_attrs.max_disp_change_gt_05 = True
        elif disp_trend_info.max_disp_change > 0.3:
            point_attrs.max_disp_change_gt_03 = True
        elif disp_trend_info.max_disp_change > 0.2:
            point_attrs.max_disp_change_gt_02 = True
        else:
            point_attrs.max_disp_change_lt = True

        if disp_trend_info.monotone_up:
            point_attrs.disp_monotone_up_true = True
        else:
            point_attrs.disp_monotone_up_false = True

        if disp_trend_info.avg_disp_change > 0.5:
            point_attrs.disp_avg_change_gt_05 = True
        elif disp_trend_info.avg_disp_change > 0.3:
            point_attrs.disp_avg_change_gt_03 = True
        else:
            point_attrs.disp_avg_change_lt = True

        if disp_trend_info.max_disp > 0.6:
            point_attrs.max_disp_gt_06 = True
        elif disp_trend_info.max_disp > 0.4:
            point_attrs.max_disp_gt_04 = True
        elif disp_trend_info.max_disp > 0.2:
            point_attrs.max_disp_gt_02 = True
        else:
            point_attrs.max_disp_lt = True

        if disp_trend_info.min_disp > 0.3:
            point_attrs.min_disp_gt_03 = True
        elif disp_trend_info.min_disp > 0.2:
            point_attrs.min_disp_gt_02 = True
        elif disp_trend_info.min_disp > 0.1:
            point_attrs.min_disp_gt_01 = True
        else:
            point_attrs.min_disp_lt = True

        if disp_tail[-1] > 0.5:
            point_attrs.current_disp_gt_05 = True
        elif disp_tail[-1] > 0.3:
            point_attrs.current_disp_gt_03 = True
        elif disp_tail[-1] > 0.2:
            point_attrs.current_disp_gt_02 = True
        else:
            point_attrs.current_disp_lt = True

        return point_attrs

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

            point_attrs = self.create_point_attrs(ohlcv_df, lower_disp_1, idx)

            sl_limit_price = 0.98 * close_price
            sell_limit_price = 1.01 * close_price

            new_opened_marked_point = MarkedPoint(
                index=idx,
                timestamp=timestamp,
                attrs=point_attrs,
                sl_price_limit=sl_limit_price,
                sell_price_limit=sell_limit_price
            )
            opened_points.append(new_opened_marked_point)

        marked_points_df_data = []
        for marked_point in sorted(closed_points, key=lambda x: x.index):
            marked_point_dict = dataclasses.asdict(marked_point)
            marked_point_dict_attrs = marked_point_dict.pop("attrs")
            marked_point_dict.update(marked_point_dict_attrs)
            marked_points_df_data.append(marked_point_dict)

        marked_points_df = pd.DataFrame(marked_points_df_data)
        end_time = time.time()
        print(f"Elapsed time: {end_time-start_time}s.")
        return marked_points_df

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

    def choose_comb(self, marked_points_df: pd.DataFrame, all_combs: list[tuple[str, ...]], selected_combs: list[CombGrade], interval_bins) -> CombGrade:

        print(f"Selected combs count: {len(selected_combs)}.")
        if selected_combs:
            exclude_mask = self.get_exclude_combs_mask(marked_points_df, [selected_comb.comb for selected_comb in selected_combs])
            marked_points_df = marked_points_df[exclude_mask].reset_index(drop=True)

        print(f"Df len after exclude selected: {len(marked_points_df)}.")

        comb_grades = []
        for comb in all_combs:
            start_time = time.time()
            select_mask = self.get_select_combs_mask(marked_points_df, [comb])
            end_time = time.time()
            # print(f"Get mask elapsed time: {end_time-start_time}")

            start_time = time.time()
            comb_df = marked_points_df[select_mask]
            end_time = time.time()
            # print(f"Apply mask elapsed time: {end_time-start_time}")

            comb_grade = self.grade_comb(comb_df, comb, interval_bins)
            comb_grades.append(comb_grade)

        comb_grades_sorted: list[CombGrade] = list(sorted(comb_grades, key=lambda x: x.k, reverse=True))

        for comb_grade in comb_grades_sorted:
            if comb_grade.count_ < 20:
                continue

            if comb_grade.uniformity < 0.6:
                continue

            return comb_grade

        raise RuntimeError("Not found comb.")

    def optimal_combs(self) -> list[CombGrade]:
        full_time_start = time.time()
        marked_points_df = pd.read_csv("optimize2/marked_points_frozen.csv")

        interval_bins = pd.cut(marked_points_df["index"], bins=12).cat.categories
        print(f"Full df len: {len(marked_points_df)}")

        tags = list(dataclasses.asdict(PointAttrs()).keys())
        combinations = list(itertools.combinations(tags, 1))
        combinations += list(itertools.combinations(tags, 2))
        # combinations += list(itertools.combinations(tags, 3))
        combs: list[tuple[str, ...]] = combinations

        print(f"All combs len: {len(combs)}")
        selected_combs: list[CombGrade] = []

        while True:
            try:
                print("Start choosing comb...")
                start_time = time.time()
                comb_grade = self.choose_comb(marked_points_df, combs, selected_combs, interval_bins)

                print("Chosen comb:")
                print(comb_grade)

                end_time = time.time()
                print(f"For this comb elapsed {end_time-start_time}s.")

                if comb_grade.k > 4.0:
                    selected_combs.append(comb_grade)
                else:
                    break
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
            CombGrade(comb=('down_strick_2', 'trend_max_ampl_gt_0_03'), count_=20, sl_count=3, uniformity=0.55,
                      k=5.666666666666667),
            CombGrade(comb=('extreme_disp_many', 'current_disp_gt_02'), count_=40, sl_count=7, uniformity=0.65,
                      k=4.714285714285714),
            CombGrade(comb=('down_strick_3', 'min_disp_lt'), count_=22, sl_count=4, uniformity=0.7272727272727273,
                      k=4.5),
            CombGrade(comb=('avg_ampl_gt_limit_gt_0_03', 'trend_max_ampl_gt_0_03'), count_=43, sl_count=8,
                      uniformity=0.6976744186046512, k=4.375),
            CombGrade(comb=('avg_ampl_gt_limit_gt_0_013', 'down_strick_3'), count_=23, sl_count=4,
                      uniformity=0.6086956521739131, k=4.75),
            CombGrade(comb=('extreme_disp_many', 'max_disp_change_gt_03'), count_=21, sl_count=4,
                      uniformity=0.5238095238095238, k=4.25),
            CombGrade(comb=('up_strick_3', 'max_disp_change_gt_05'), count_=91, sl_count=18,
                      uniformity=0.8571428571428572, k=4.055555555555555),
            CombGrade(comb=('extreme_disp_moderate', 'trend_kind_flat'), count_=23, sl_count=4,
                      uniformity=0.7391304347826086, k=4.75),
        ]

        marked_points_df = pd.read_csv("optimize2/marked_points_frozen.csv")

        select_mask = self.get_select_combs_mask(marked_points_df, [comb_.comb for comb_ in combs])

        selected_df = marked_points_df[select_mask].reset_index(drop=True)

        selected_df["interval"] = pd.cut(selected_df["index"], bins=12)

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


if __name__ == "__main__":
    optimizer = TradingOptimizer()
    # res = optimizer.mark_data()
    # res.to_csv("optimize2/marked_points.csv", index=False)

    # opt = optimizer.optimal_combs()
    opt = optimizer.super_benchmark()
    # print(res)