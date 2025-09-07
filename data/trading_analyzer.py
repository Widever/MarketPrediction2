import csv
import itertools
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import auto, IntEnum
from statistics import mean

import pandas as pd

import dispersion as dsp
import runtime_data as rd
from data.control_panel_core import DecisionEvent, MarketEvent


class PriceTrend(IntEnum):
    UP = auto()
    DOWN = auto()
    FLAT = auto()

@dataclass
class PriceTrendInfo:
    trend_kind: PriceTrend
    trend_value: float
    trend_len: int
    trend_start_index: int
    max_ampl: float
    avg_ampl: float
    avg_ampl_gt_limit: float
    trend_ohlcva: list[tuple[int, float, float, float, float, float]]

@dataclass
class DispTrendInfo:
    disps_on_trend: list[float]
    trend_start_high_disp: int | None
    trend_start_medium_disp: int | None
    high_disp_on_middle: list[int]
    medium_disp_on_middle: list[int]
    low_disp_on_middle: list[int]
    end_disp_level: str
    monotone_up: bool
    avg_disp_change: float
    max_disp_change: float
    avg_disp_greater_limit: float
    max_disp: float
    is_lightning: bool
    is_saddle: bool

@dataclass
class LastTrendsInfo:
    volume_and_len: list[tuple[float, int]]
    sum_volume_of_last_3: float
    sum_last_2: float


class TradingAnalyzer:

    def analyze(self):
        print()
        print("Analysis:")
        price_trend_info = self.check_price_trend()
        print()
        disp_trend_info = self.check_disp_trend(price_trend_info)
        last_trends_info = self.check_last_trends()
        btc_price_trend_info = self.check_price_trend(verbose=False, symbol="BTCUSDT")
        decision, reason = self.decision(price_trend_info, disp_trend_info, last_trends_info, btc_price_trend_info)
        print(f"{decision=}, {reason=}")
        return decision, reason

    def check_trigger(self):
        lower_disp_1 = dsp.get_disp_1_lower()
        last_disps = [
            float(lower_disp_1["disp"].reset_index(drop=True).at[i]) for i in
            range(rd.VARS.simulator.current_index - 10, rd.VARS.simulator.current_index + 1)
        ]

        if last_disps[-1] > 0.5:
            return True, "big_disp"
        else:
            return False, None

    def check_price_trend(self, for_index: int = None, verbose: bool = True, symbol: str = None) -> PriceTrendInfo:

        if for_index is None:
            for_index = rd.VARS.simulator.current_index

        if symbol is None:
            ohlcv_df = rd.VARS.simulator.ohlcv_df
        else:
            ohlcv_df = rd.CURRENCY_DATAS[symbol].ohlcv_df

        end_low_price = ohlcv_df["low"].at[for_index]
        end_high_price = ohlcv_df["high"].at[for_index]

        check_last_n_count = 20
        flat_limit = 0.009

        lowest_known = end_low_price
        lowest_known_i = for_index

        for i in range(1, check_last_n_count):

            low_price_for_i =  ohlcv_df["low"].at[for_index - i]
            high_price_for_i =  ohlcv_df["high"].at[for_index - i]

            if low_price_for_i < lowest_known:
                lowest_known = low_price_for_i
                lowest_known_i = for_index - i
            else:
                if (high_price_for_i / lowest_known - 1) > flat_limit:
                    break

        highest_known = end_high_price
        highest_known_i = for_index

        for i in range(1, check_last_n_count):

            low_price_for_i =  ohlcv_df["low"].at[for_index - i]
            high_price_for_i =  ohlcv_df["high"].at[for_index - i]

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
                current_open_price = ohlcv_df["open"].at[for_index]
                current_close_price = ohlcv_df["close"].at[for_index]
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

        for trend_i in range(trend_start_index, trend_start_index+trend_len+1):
            i_open_price = int(ohlcv_df["open"].at[trend_i])
            i_high_price = float(ohlcv_df["high"].at[trend_i])
            i_low_price = float(ohlcv_df["low"].at[trend_i])
            i_close_price = float(ohlcv_df["close"].at[trend_i])
            i_volume = float(ohlcv_df["volume"].at[trend_i])
            i_ampl = float(i_high_price / i_low_price - 1)
            trend_ohlcva.append((i_open_price, i_high_price, i_low_price, i_close_price, i_volume, i_ampl))

        ampls = [x[5] for x in trend_ohlcva]
        max_ampl = max(ampls)
        avg_ampl = mean(ampls)

        ampl_gt_limit = [x for x in ampls if x > 0.003]
        avg_ampl_gt_limit = mean(ampl_gt_limit) if ampl_gt_limit else 0.0

        if verbose:
            print(f"Trend: {trend.name}.")
            print(f"Trend value: {trend_value}.")
            print(f"Trend len: {trend_len}.")
            print(f"Max ampl: {max_ampl}.")
            print(f"Avg ampl: {avg_ampl}.")

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

    def check_disp_trend(self, price_trend_info: PriceTrendInfo) -> DispTrendInfo:
        lower_disp_1 = dsp.get_disp_1_lower()

        disps_on_trend = [
            float(lower_disp_1["disp"].reset_index(drop=True).at[i]) for i in
            range(price_trend_info.trend_start_index, rd.VARS.simulator.current_index+1)
        ]

        max_disp = max(disps_on_trend)

        normalized_disps = [x/max_disp for x in disps_on_trend]

        high_disp_indexes = [i for i, x in enumerate(normalized_disps) if 0.7 <= x]
        medium_disp_indexes = [i for i, x in enumerate(normalized_disps) if 0.4 <= x < 0.7]
        low_disp_indexes = [i for i, x in enumerate(normalized_disps) if  x < 0.4]

        assert len(high_disp_indexes) + len(medium_disp_indexes) + len(low_disp_indexes) == len(normalized_disps)

        if price_trend_info.trend_len > 3:
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


        print(f"Disps: {[round(float(x), 5) for x in disps_on_trend]}")
        print(f"Trend start high disp: {trend_start_high_disp}")
        print(f"Trend start medium disp: {trend_start_medium_disp}")
        print(f"High disp on middle, {len(middle_indexes_high_disp)}: {middle_indexes_high_disp}")
        print(f"Medium disp on middle, {len(middle_indexes_medium_disp)}: {middle_indexes_medium_disp}")
        print(f"Low disp on middle, {len(middle_indexes_low_disp)}: {middle_indexes_low_disp}")
        print(f"End disp: {trend_end_disp}.")

        monotone_up = False
        avg_disp_change = 0
        max_disp_change = 0
        if len(disps_on_trend) > 1:
            disp_changes = [disps_on_trend[i] - disps_on_trend[i-1] for i in range(1, len(disps_on_trend))]
            if all(x > 0 for x in disp_changes):
                monotone_up = True

            avg_disp_change = mean(max(abs(x), 0.1) for x in disp_changes)
            max_disp_change = max(disp_changes, key=lambda x: abs(x))

        print(f"{monotone_up=}")
        print(f"{avg_disp_change=}")
        print(f"{max_disp_change=}")

        is_lightning = trend_start_medium_disp is not None and trend_end_disp == "high" and not middle_indexes_high_disp and end_index - trend_start_medium_disp > 1
        is_saddle = trend_start_high_disp is not None and trend_end_disp == "high" and not middle_indexes_high_disp and end_index - trend_start_high_disp > 1

        print(f"{is_lightning=}")
        print(f"{is_saddle=}")

        disps_greater_limit = [x for x in disps_on_trend if x > 0.11]
        avg_disp_greater_limit = mean(disps_greater_limit) if disps_greater_limit else 0
        max_disp = max(disps_on_trend)

        print(f"{max_disp=}")
        print(f"{avg_disp_greater_limit=}")

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
            is_lightning=is_lightning,
            is_saddle=is_saddle
        )


    def check_last_trends(self, n: int = 5) -> LastTrendsInfo:
        info = []
        last_start_i = rd.VARS.simulator.current_index
        for _ in range(n):
            trend_info = self.check_price_trend(for_index=last_start_i, verbose=False)
            volume = trend_info.trend_value
            info.append((volume, trend_info.trend_len))

            last_start_i = trend_info.trend_start_index

            if trend_info.trend_len == 1:
                last_start_i -= 1

        info = list(reversed(info))
        msg = ", ".join([f"{round(v, 4)}/{l}" for v, l in info])
        print(f"Last {n} trend volumes: {msg}.")

        sum_volume_of_last_3 = sum(abs(x[0]) for x in info[-3:])
        sum_last_2 = abs(sum(x[0] for x in info[-2:]))
        print(f"Last 3 trends volume sum: {sum_volume_of_last_3}")
        return LastTrendsInfo(
            volume_and_len=info,
            sum_volume_of_last_3=sum_volume_of_last_3,
            sum_last_2=sum_last_2
        )


    def decision(
        self,
        price_trend_info: PriceTrendInfo,
        disp_trend_info: DispTrendInfo,
        last_trends_info: LastTrendsInfo,
        btc_price_trend_info: PriceTrendInfo
    ):
        last_sl = rd.VARS.bot.sl_history[-1] if rd.VARS.bot.sl_history else None
        dist_to_last_sl = 50000 if last_sl is None else rd.VARS.simulator.current_index - last_sl
        lower_disp_1 = dsp.get_disp_1_lower()

        disp_tail = [
            float(lower_disp_1["disp"].reset_index(drop=True).at[i]) for i in
            range(rd.VARS.simulator.current_index-13, rd.VARS.simulator.current_index + 1)
        ]

        avg_disp_tail = mean(disp_tail)
        extreme_disp = len([x for x in disp_tail if x > 0.88])

        if dist_to_last_sl > 60:
            dist_to_last_sl_str = "to_sl_very_far"
        elif dist_to_last_sl > 10:
            dist_to_last_sl_str = "to_sl_far"
        elif dist_to_last_sl > 3:
            dist_to_last_sl_str = "to_sl_close"
        else:
            dist_to_last_sl_str = "to_sl_very_close"

        if extreme_disp > 4:
            extreme_disp_str = "extreme_disp_many"
        elif extreme_disp > 1:
            extreme_disp_str = "extreme_disp_moderate"
        else:
            extreme_disp_str = "extreme_disp_few"

        if price_trend_info.avg_ampl_gt_limit > 0.03:
            avg_ampl_str = "avg_ampl_gt_limit>0.03"
        elif price_trend_info.avg_ampl_gt_limit > 0.013:
            avg_ampl_str = "avg_ampl_gt_limit>0.013"
        elif price_trend_info.avg_ampl_gt_limit > 0.007:
            avg_ampl_str = "avg_ampl_gt_limit>0.007"
        else:
            avg_ampl_str = "avg_ampl_gt_limit<"

        if avg_disp_tail > 0.5:
            avg_disp_tail_str = "avg_disp_tail>0.5"
        elif avg_disp_tail > 0.3:
            avg_disp_tail_str = "avg_disp_tail>0.3"
        elif avg_disp_tail > 0.2:
            avg_disp_tail_str = "avg_disp_tail>0.2"
        else:
            avg_disp_tail_str = "avg_disp_tail<"

        if last_trends_info.volume_and_len[-1][0] < 0 and last_trends_info.volume_and_len[-2][0] < 0 and last_trends_info.volume_and_len[-3][0] < 0:
            down_strick_str = "down_strick_3"
        elif last_trends_info.volume_and_len[-1][0] < 0 and last_trends_info.volume_and_len[-2][0] < 0:
            down_strick_str = "down_strick_2"
        elif last_trends_info.volume_and_len[-1][0] < 0:
            down_strick_str = "down_strick_1"
        else:
            down_strick_str = "down_strick_0"

        if last_trends_info.volume_and_len[-1][0] > 0 and last_trends_info.volume_and_len[-2][0] > 0 and last_trends_info.volume_and_len[-3][0] > 0:
            up_strick_str = "down_strick_3"
        elif last_trends_info.volume_and_len[-1][0] > 0 and last_trends_info.volume_and_len[-2][0] > 0:
            up_strick_str = "up_strick_2"
        elif last_trends_info.volume_and_len[-1][0] > 0:
            up_strick_str = "up_strick_1"
        else:
            up_strick_str = "up_strick_0"

        if price_trend_info.trend_len > 12:
            trend_len_str = "trend_len_long"
        elif price_trend_info.trend_len > 5:
            trend_len_str = "trend_len_middle"
        else:
            trend_len_str = "trend_len_short"

        if price_trend_info.max_ampl > 0.05:
            trend_max_ampl_str = "trend_max_ampl>0.05"
        elif price_trend_info.max_ampl > 0.03:
            trend_max_ampl_str = "trend_max_ampl>0.03"
        elif price_trend_info.max_ampl > 0.01:
            trend_max_ampl_str = "trend_max_ampl>0.01"
        else:
            trend_max_ampl_str = "trend_max_ampl<"

        if price_trend_info.trend_kind == PriceTrend.UP:
            trend_kind_str = "trend_kind_up"
        elif price_trend_info.trend_kind == PriceTrend.DOWN:
            trend_kind_str = "trend_kind_down"
        elif price_trend_info.trend_kind == PriceTrend.FLAT:
            trend_kind_str = "trend_kind_flat"
        else:
            trend_kind_str = "trend_kind_unknown"

        if last_trends_info.volume_and_len[-2][0] < 0:
            prev_trend_kind_str = "prev_t_kind_down"
        else:
            prev_trend_kind_str = "prev_t_kind_up"

        if disp_trend_info.max_disp_change > 0.5:
            max_disp_change_str = "max_disp_change>0.5"
        elif disp_trend_info.max_disp_change > 0.3:
            max_disp_change_str = "max_disp_change>0.3"
        elif disp_trend_info.max_disp_change > 0.2:
            max_disp_change_str = "max_disp_change>0.2"
        else:
            max_disp_change_str = "max_disp_change<"

        if disp_trend_info.monotone_up:
            disp_monotone_up_str = "disp_monotone_up_true"
        else:
            disp_monotone_up_str = "disp_monotone_up_false"

        if disp_trend_info.avg_disp_change > 0.5:
            disp_avg_change_str = "disp_avg_change>0.5"
        elif disp_trend_info.avg_disp_change > 0.3:
            disp_avg_change_str = "disp_avg_change>0.3"
        else:
            disp_avg_change_str = "disp_avg_change<"

        reason = (f"{extreme_disp_str};{avg_ampl_str};{avg_disp_tail_str};{up_strick_str};{down_strick_str};"
                  f"{trend_len_str};{trend_max_ampl_str};{trend_kind_str};{prev_trend_kind_str};{max_disp_change_str};{disp_monotone_up_str};"
                  f"{disp_avg_change_str}")

        selected_combs = [
            (('avg_ampl_gt_limit>0.013', 'trend_len_short', 'up_strick_1'), (31, 4, 6.75)),
            (('up_strick_0', 'disp_monotone_up_false', 'extreme_disp_many'), (34, 5, 5.8)),
            (('trend_len_short', 'disp_avg_change>0.3', 'down_strick_0'), (26, 4, 5.5)),
            (('prev_t_kind_down', 'disp_avg_change<', 'avg_ampl_gt_limit>0.007'), (32, 6, 4.333333333333333)),
            (('trend_len_short', 'max_disp_change>0.5', 'trend_max_ampl<'), (35, 7, 4.0)),
        ]

        if any(all(tag in reason for tag in comb[0]) for comb in selected_combs):
            return True, reason

        return False, reason

        return True, "exit"

    def parse_line(self, line: str) -> dict:

        # Extract key=value pairs like [c=123], [sl=45]
        pairs = re.findall(r"\[(\w+)=(.*?)\]", line)
        data = {k: v for k, v in pairs}

        return data

    def read_benchmark_output_to_df(self, filename: str) -> pd.DataFrame:
        with open(filename, "r", encoding="utf-8") as f:
            rows = [self.parse_line(line.strip()) for line in f if line.strip()]
        return pd.DataFrame(rows)

    def _comb_uniformity(self, comb_df, interval_bins):
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



    def _optimize(self, exclude_combs: list[tuple[str]]):
        df = pd.read_csv("optimize/marked_events_40k.csv")

        tags = set(tag for index, row in df.iterrows() for tag in row["reason"].split(";"))
        wide_df_data = []

        for index, row in df.iterrows():
            reason = row["reason"]
            reason_tags = reason.split(";")

            wide_row = {k: 1 if k in reason_tags else 0 for k in tags}
            wide_row["index"] = row["index"]
            wide_row["sl"] = row["sl"]
            wide_df_data.append(wide_row)

        wide_df = pd.DataFrame(wide_df_data)
        print(f"before filter {len(wide_df)=}")
        print(f"exclude: {exclude_combs}")

        interval_bins = pd.cut(wide_df["index"], bins=12).cat.categories

        mask = None
        for exclude_comb in exclude_combs:
            mask_ = (wide_df[list(exclude_comb)] == 1).all(axis=1)
            if mask is None:
                mask = mask_
            else:
                mask = mask | mask_

        if mask is not None:
            wide_df = wide_df[~mask]

        print(f"after filter {len(wide_df)=}")

        combinations = list(itertools.combinations(tags, 1))
        combinations += list(itertools.combinations(tags, 2))
        combinations += list(itertools.combinations(tags, 3))
        # combinations += list(itertools.combinations(tags, 4))
        print(f"combinations len: {len(combinations)}")

        start_time = time.time()
        combinations_stat = defaultdict(lambda: [None, None, None])
        for comb in combinations:
            mask = (wide_df[list(comb)] == 1).all(axis=1)
            comb_df = wide_df[mask]

            count_ = len(comb_df)
            sl_count = comb_df["sl"].sum()

            combinations_stat[comb][0] = int(count_)
            combinations_stat[comb][1] = int(sl_count)
            combinations_stat[comb][2] = self._comb_uniformity(comb_df, interval_bins)

        comb_stats_sorted = [
            x
            for x in sorted(
                (
                    (comb, (count_, sl, ((count_ - sl) / sl if sl > 0 else sl), uniformity))
                    for comb, (count_, sl, uniformity) in combinations_stat.items() if count_ > 25
                )
                , key=lambda x: x[1][2], reverse=True)
        ]
        selected_comb = None
        for i, (comb, (count_, sl, k, uniformity)) in enumerate(comb_stats_sorted):
            if selected_comb is None:
                if uniformity < 0.5:
                    pass
                else:
                    selected_comb = (comb, (count_, sl, k, uniformity))
            print(f"{comb=}, {count_=}, {sl=}, k={round(k, 4)}, uniformity={round(uniformity, 4)}")
            if i > 100:
                break

        end_time = time.time()
        print(f"Elapsed: {end_time - start_time}s.")
        return selected_comb

    def _go_to_index(self):
        i = 60000
        rd.VARS.simulator.current_index = i
        print(f"Set current index {i=}.")

    def _check_fast_benchmark(self):
        # OPTIMIZE 60k, MAX COMB LEN 3
        selected_combs = [
            (('up_strick_0', 'max_disp_change<', 'extreme_disp_many'), (26, 3, 7.666666666666667)),
            (('avg_ampl_gt_limit>0.013', 'up_strick_1', 'trend_len_short'), (33, 4, 7.25)),
            (('avg_disp_tail>0.3', 'up_strick_1', 'trend_len_short'), (29, 4, 6.25)),
            (('down_strick_0', 'avg_ampl_gt_limit>0.007', 'disp_avg_change<'), (34, 6, 4.666666666666667)),
            (('down_strick_1', 'extreme_disp_moderate', 'max_disp_change<'), (28, 5, 4.6)),
            (('disp_monotone_up_false', 'avg_disp_tail>0.5', 'disp_avg_change>0.3'), (27, 5, 4.4)),
            (('disp_monotone_up_false', 'trend_kind_up', 'disp_avg_change>0.3'), (27, 5, 4.4)),
            (('trend_kind_up', 'max_disp_change>0.5', 'trend_len_long'), (26, 5, 4.2)),
            (('max_disp_change<', 'prev_t_kind_up', 'trend_len_long'), (26, 5, 4.2)),
        ]

        # OPTIMIZE 60k, MAX COMB LEN 4
        selected_combs = [
            (('up_strick_1', 'trend_len_short', 'max_disp_change<', 'trend_max_ampl>0.01'), (30, 3, 9.0)),
            (('up_strick_0', 'extreme_disp_many', 'max_disp_change<'), (26, 3, 7.666666666666667)),
            (('avg_disp_tail>0.3', 'avg_ampl_gt_limit>0.007', 'disp_monotone_up_true', 'extreme_disp_few'),
             (47, 7, 5.714285714285714)),
            (('trend_len_long', 'prev_t_kind_up', 'avg_disp_tail>0.2', 'trend_max_ampl>0.01'), (26, 4, 5.5)),
            (('avg_disp_tail>0.3', 'extreme_disp_moderate', 'down_strick_1', 'disp_avg_change<'), (31, 5, 5.2)),
            (('avg_disp_tail>0.3', 'avg_ampl_gt_limit<', 'trend_max_ampl>0.01', 'extreme_disp_few'), (30, 5, 5.0)),
            (('prev_t_kind_down', 'disp_avg_change<', 'avg_ampl_gt_limit>0.007', 'extreme_disp_few'), (29, 5, 4.8)),
            (('up_strick_1', 'avg_ampl_gt_limit>0.013'), (27, 5, 4.4)),
            (('up_strick_0', 'max_disp_change>0.5', 'trend_len_short', 'trend_max_ampl<'), (32, 6, 4.333333333333333)),
            (('down_strick_3', 'disp_avg_change<', 'extreme_disp_few'), (26, 5, 4.2)),
        ]

        # OPTIMIZE 40k, MAX COMB LEN 4
        selected_combs = [
            (('extreme_disp_moderate', 'prev_t_kind_down', 'trend_len_short', 'trend_max_ampl>0.01'), (32, 4, 7.0)),
            (('prev_t_kind_down', 'extreme_disp_many'), (29, 4, 6.25)),
            (('avg_disp_tail>0.5', 'up_strick_0', 'avg_ampl_gt_limit>0.013', 'disp_avg_change<'), (26, 4, 5.5)),
            (('down_strick_1', 'trend_len_short', 'max_disp_change>0.5', 'trend_max_ampl<'), (30, 5, 5.0)),
            (('avg_ampl_gt_limit>0.007', 'avg_disp_tail>0.3', 'disp_monotone_up_true', 'extreme_disp_few'),
             (40, 7, 4.714285714285714)),
            (('avg_disp_tail<', 'trend_len_middle', 'trend_max_ampl<', 'max_disp_change>0.3'), (28, 5, 4.6)),
            (('disp_avg_change<', 'avg_ampl_gt_limit>0.007', 'down_strick_0'), (27, 5, 4.4)),
            (('trend_len_short', 'disp_avg_change>0.3', 'down_strick_0'), (26, 5, 4.2)),
            (('avg_ampl_gt_limit<', 'trend_max_ampl>0.01', 'avg_disp_tail>0.3', 'extreme_disp_few'), (26, 5, 4.2)),
            (('trend_len_long', 'avg_disp_tail>0.2', 'up_strick_0', 'max_disp_change>0.3'), (31, 6, 4.166666666666667)),
        ]

        # OPTIMIZE 40k, MAX COMB LEN 3
        selected_combs = [
            (('trend_len_short', 'up_strick_1', 'avg_ampl_gt_limit>0.013'), (31, 4, 6.75)),
            (('trend_kind_up', 'trend_len_short', 'disp_avg_change>0.3'), (26, 4, 5.5)),
            (('extreme_disp_many', 'disp_monotone_up_false'), (43, 8, 4.375)),
            (('disp_avg_change<', 'avg_ampl_gt_limit>0.007', 'prev_t_kind_down'), (32, 6, 4.333333333333333)),
            (('trend_len_short', 'trend_max_ampl<', 'max_disp_change>0.5'), (35, 7, 4.0)),
        ]

        selected_combs = [x[0] for x in selected_combs]

        df = pd.read_csv("optimize/marked_events.csv")

        tags = set(tag for index, row in df.iterrows() for tag in row["reason"].split(";"))
        wide_df_data = []

        for index, row in df.iterrows():
            reason = row["reason"]
            reason_tags = reason.split(";")

            wide_row = {k: 1 if k in reason_tags else 0 for k in tags}
            wide_row["index"] = row["index"]
            wide_row["sl"] = row["sl"]
            wide_df_data.append(wide_row)

        wide_df = pd.DataFrame(wide_df_data)

        mask = None
        for selected_comb in selected_combs:
            mask_ = (wide_df[list(selected_comb)] == 1).all(axis=1)
            if mask is None:
                mask = mask_
            else:
                mask = mask | mask_

        wide_df = wide_df[mask]

        wide_df["interval"] = pd.cut(wide_df["index"], bins=10)

        intervals_stat = wide_df.groupby("interval", observed=False).agg(
            count_=("index", "size"),
            sl_count_=("sl", "sum"),
        ).reset_index()

        intervals_stat["k"] = (intervals_stat["count_"] - intervals_stat["sl_count_"]) / intervals_stat["sl_count_"]
        total_count = intervals_stat["count_"].sum()
        total_sl_count = intervals_stat["sl_count_"].sum()
        total_k = (total_count - total_sl_count) / total_sl_count

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        print("Intervals stat:")
        print(intervals_stat)
        print()
        print(f"Total: {total_count=}, {total_sl_count=}, {total_k=}")

    def optimize(self):
        # self._check_fast_benchmark()
        # return
        # self._go_to_index()
        # return
        print("optimize")
        selected_combs = []
        selected_combs_info = []
        start_time = time.time()
        while True:
            optimized_comb = self._optimize(selected_combs)
            if optimized_comb is None:
                print("Not found optimized comb.")
                break

            (comb, (count_, sl, k, uniformity)) = optimized_comb

            # user_input = input("Exclude comb: ")
            # if user_input == "stop":
            #     break

            # exclude_comb = tuple(res for s in user_input.split(",") if (res := s.strip().replace("'", "")))
            # selected_combs.append(exclude_comb)

            if k < 4:
                break
            else:
                selected_combs.append(comb)
                selected_combs_info.append((comb, (count_, sl, round(k, 4), round(uniformity, 4))))

        print(">>> selected_combs: ")
        for comb in selected_combs:
            print(comb)

        print(">>> selected_combs_info: ")
        for info in selected_combs_info:
            print(info, ",")

        end_time = time.time()
        print(f"Elapsed time: {(end_time-start_time)/60} min.")

    def big_benchmark_count(self) -> int:
        return 70000

    def extract_events(self, events):

        last_true_decision_event: DecisionEvent | None = None

        output_filename = "optimize/marked_events_new.csv"
        with open(output_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            columns = ["index", "reason", "sl"]
            writer.writerow(columns)

            for index, events in sorted(events.items(), key=lambda x: x[0]):
                for event in events:
                    if event.__class__.__name__ == "DecisionEvent":
                        if event.decision:
                            last_true_decision_event = event

                    elif event.__class__.__name__ == "MarketEvent" and event.type == "sell":
                        assert last_true_decision_event is not None
                        writer.writerow([index, last_true_decision_event.decision_reason, 0])


                    elif event.__class__.__name__ == "MarketEvent" and event.type == "stop_loss":
                        assert last_true_decision_event is not None
                        writer.writerow([index, last_true_decision_event.decision_reason, 1])
