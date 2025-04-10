import importlib
from enum import auto, IntEnum

import dispersion as dsp
import runtime_data as rd


class PriceTrend(IntEnum):
    UP = auto()
    DOWN = auto()
    FLAT = auto()


class TradingAnalyzer:

    def print_analyze(self):
        print("Analyze...")
        self.check_current_price_trend()


    def check_disp_closing(self):
        importlib.reload(dsp)
        lower_disp_1 = dsp.get_disp_1_lower()
        lower_disp_2 = dsp.get_disp_2_lower()
        lower_disp_3 = dsp.get_disp_3_lower()

        current_disp_1 = lower_disp_1["disp"].reset_index(drop=True).at[rd.VARS.simulator.current_index]
        current_disp_2 = lower_disp_2["disp"].reset_index(drop=True).at[rd.VARS.simulator.current_index]
        current_disp_3 = lower_disp_3["disp"].reset_index(drop=True).at[rd.VARS.simulator.current_index]

        disp_on_interval = []
        end_disp = current_disp_1
        close_limit = 0.8 * end_disp
        for i in range(1, 30):

            disp_1_for_i = lower_disp_1["disp"].reset_index(drop=True).at[rd.VARS.simulator.current_index - i]


    def check_current_price_trend(self):

        end_low_price = rd.VARS.simulator.ohlcv_df["low"].at[rd.VARS.simulator.current_index]
        end_high_price = rd.VARS.simulator.ohlcv_df["high"].at[rd.VARS.simulator.current_index]

        check_last_n_count = 20
        flat_limit = 0.009

        lowest_known = end_low_price
        lowest_known_i = rd.VARS.simulator.current_index

        for i in range(1, check_last_n_count):

            low_price_for_i =  rd.VARS.simulator.ohlcv_df["low"].at[rd.VARS.simulator.current_index - i]

            if low_price_for_i < lowest_known:
                lowest_known = low_price_for_i
                lowest_known_i = rd.VARS.simulator.current_index - i
            else:
                if (low_price_for_i / lowest_known - 1) > flat_limit:
                    break

        highest_known = end_high_price
        highest_known_i = rd.VARS.simulator.current_index

        for i in range(1, check_last_n_count):

            high_price_for_i =  rd.VARS.simulator.ohlcv_df["high"].at[rd.VARS.simulator.current_index - i]

            if high_price_for_i > highest_known:
                highest_known = high_price_for_i
                highest_known_i = rd.VARS.simulator.current_index - i
            else:
                if (highest_known / high_price_for_i - 1) > flat_limit:
                    break

        up_trend_value = end_high_price / lowest_known - 1
        down_trend_value = highest_known / end_low_price - 1
        print(f"{up_trend_value=}")
        print(f"{down_trend_value=}")

        if up_trend_value > flat_limit > down_trend_value:
            trend = PriceTrend.UP
        elif down_trend_value > flat_limit > up_trend_value:
            trend = PriceTrend.DOWN
        elif up_trend_value < flat_limit and down_trend_value < flat_limit:
            trend = PriceTrend.FLAT
        elif up_trend_value > flat_limit and down_trend_value > flat_limit:
            up_trend_len = rd.VARS.simulator.current_index - lowest_known_i
            down_trend_len = rd.VARS.simulator.current_index - highest_known_i

            if up_trend_len == 0 and down_trend_len > 0:
                trend = PriceTrend.DOWN
            elif down_trend_len == 0 and up_trend_len > 0:
                trend = PriceTrend.UP
            elif down_trend_len > 0 and up_trend_len > 0:
                trend = PriceTrend.UP if highest_known_i < lowest_known_i else PriceTrend.DOWN
            elif down_trend_len == 0 and down_trend_len == 0:
                trend = PriceTrend.DOWN
            else:
                raise RuntimeError("Expect all cases covered.")
        else:
            raise RuntimeError("Expect all cases covered.")

        if trend == PriceTrend.UP:
            trend_len = rd.VARS.simulator.current_index - lowest_known_i
            trend_value  = up_trend_value
        elif trend == PriceTrend.DOWN:
            trend_len = rd.VARS.simulator.current_index - highest_known_i
            trend_value = down_trend_value
        elif trend == PriceTrend.FLAT:
            trend_len = check_last_n_count
            trend_value = max(down_trend_value, up_trend_value)
        else:
            raise RuntimeError("Expect all cases covered.")

        print()
        print(f"Trend: {trend.name}.")
        print(f"Trend value: {trend_value}.")
        print(f"Trend len: {trend_len+1}.")
