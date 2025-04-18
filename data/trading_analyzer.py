from dataclasses import dataclass
from enum import auto, IntEnum

import dispersion as dsp
import runtime_data as rd


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


class TradingAnalyzer:

    def print_analyze(self):
        print()
        print("Analysis:")
        trend_info = self.check_price_trend()
        print()
        self.check_disp_trend(trend_info)
        self.check_last_trends()


    def check_price_trend(self, for_index: int = None, verbose: bool = True) -> PriceTrendInfo:

        if for_index is None:
            for_index = rd.VARS.simulator.current_index

        end_low_price = rd.VARS.simulator.ohlcv_df["low"].at[for_index]
        end_high_price = rd.VARS.simulator.ohlcv_df["high"].at[for_index]

        check_last_n_count = 20
        flat_limit = 0.009

        lowest_known = end_low_price
        lowest_known_i = for_index

        for i in range(1, check_last_n_count):

            low_price_for_i =  rd.VARS.simulator.ohlcv_df["low"].at[for_index - i]
            high_price_for_i =  rd.VARS.simulator.ohlcv_df["high"].at[for_index - i]

            if low_price_for_i < lowest_known:
                lowest_known = low_price_for_i
                lowest_known_i = for_index - i
            else:
                if (high_price_for_i / lowest_known - 1) > flat_limit:
                    break

        highest_known = end_high_price
        highest_known_i = for_index

        for i in range(1, check_last_n_count):

            low_price_for_i =  rd.VARS.simulator.ohlcv_df["low"].at[for_index - i]
            high_price_for_i =  rd.VARS.simulator.ohlcv_df["high"].at[for_index - i]

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
                current_open_price = rd.VARS.simulator.ohlcv_df["open"].at[for_index]
                current_close_price = rd.VARS.simulator.ohlcv_df["close"].at[for_index]
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

        if verbose:
            print(f"Trend: {trend.name}.")
            print(f"Trend value: {trend_value}.")
            print(f"Trend len: {trend_len}.")

        return PriceTrendInfo(
            trend_kind=trend,
            trend_value=trend_value,
            trend_len=trend_len,
            trend_start_index=for_index - trend_len + 1
        )

    def check_disp_trend(self, price_trend_info: PriceTrendInfo):
        lower_disp_1 = dsp.get_disp_1_lower()

        disps_on_trend = [
            lower_disp_1["disp"].reset_index(drop=True).at[i] for i in
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

    def check_last_trends(self, n: int = 5):
        info = []
        last_start_i = rd.VARS.simulator.current_index + 1
        for _ in range(n):
            trend_info = self.check_price_trend(for_index=last_start_i-1, verbose=False)
            volume = trend_info.trend_value
            info.append((volume, trend_info.trend_len))
            last_start_i = trend_info.trend_start_index

        msg = ", ".join([f"{round(v, 4)}/{l}" for v, l in reversed(info)])
        print(f"Last {n} trend volumes: {msg}.")
