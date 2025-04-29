from dataclasses import dataclass
from enum import auto, IntEnum
from statistics import mean

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
    is_lightning: bool
    is_saddle: bool

@dataclass
class LastTrendsInfo:
    volume_and_len: list[tuple[float, int]]


class TradingAnalyzer:

    def print_analyze(self):
        print()
        print("Analysis:")
        price_trend_info = self.check_price_trend()
        print()
        disp_trend_info = self.check_disp_trend(price_trend_info)
        last_trends_info = self.check_last_trends()
        btc_price_trend_info = self.check_price_trend(verbose=False, symbol="BTCUSDT")
        decision, reason = self.decision(price_trend_info, disp_trend_info, last_trends_info, btc_price_trend_info)
        return decision, reason



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

        msg = ", ".join([f"{round(v, 4)}/{l}" for v, l in reversed(info)])
        print(f"Last {n} trend volumes: {msg}.")

        return LastTrendsInfo(
            volume_and_len=info
        )

    def decision(
        self,
        price_trend_info: PriceTrendInfo,
        disp_trend_info: DispTrendInfo,
        last_trends_info: LastTrendsInfo,
        btc_price_trend_info: PriceTrendInfo
    ):
        print(f"{btc_price_trend_info.trend_kind=}")
        print(f"{btc_price_trend_info.trend_value=}")

        if price_trend_info.trend_kind != PriceTrend.DOWN:
            print(f"Decision False: Trend is {price_trend_info.trend_kind.name}.")
            return False, "trend_not_down"

        if (min_disp := min(disp_trend_info.disps_on_trend)) > 0.32:
            print(f"Decision False: Min disp is too big: {min_disp}.")
            return False, "min_disp_big"


        if abs(price_trend_info.trend_value) > 0.032:
            print(f"Decision False: Trend volume is too big - {price_trend_info.trend_value}.")
            return False, "trend_volume_big"

        if (prev_trend_vol := abs(last_trends_info.volume_and_len[1][0])) > 0.037:
            print(f"Decision False: Prev trend volume is too big - {prev_trend_vol}.")
            return False, "prev_trend_volume_big"

        if (sum_trend_vol := abs(last_trends_info.volume_and_len[1][0]) + abs(price_trend_info.trend_value)) > 0.069:
            print(f"Decision False: Sum trend volume is too big - {sum_trend_vol}.")
            return False, "sum_trend_volume_big"

        if abs(disp_trend_info.max_disp_change) > 0.65:
            print(f"Decision False: Trend max disp change is too big - {disp_trend_info.max_disp_change}.")
            return False, "max_disp_change_big"

        if disp_trend_info.is_lightning:
            if price_trend_info.trend_value > -0.011:
                print("Decision False: Lightning, but trend is weak.")
                return False, "lightning_but_trend_weak"
            elif price_trend_info.trend_len > 3:
                print(f"Decision False: Long lightning ({price_trend_info.trend_len}).")
                return False, "lightning_but_long"
            elif disp_trend_info.monotone_up:
                print(f"Decision False: Lightning and monotone up at the same time.")
                return False, "lightning_but_monotone_up"
            else:
                # if btc_price_trend_info.trend_kind == PriceTrend.UP:
                #     print(f"Decision False: Lightning, but BTC trend is {btc_price_trend_info.trend_kind.name}.")
                #     return False

                print(f"Decision True: Good lightning.")
                return True, "good_lightning"

        if disp_trend_info.is_saddle:
            if price_trend_info.trend_value < -0.025:
                print("Decision False: Saddle on big down trend.")
                return False, "saddle_but_volume_big"
            elif price_trend_info.trend_len > 5:
                print(f"Decision False: Long saddle ({price_trend_info.trend_len}).")
                return False, "saddle_but_long"
            else:
                # if btc_price_trend_info.trend_kind == PriceTrend.UP:
                #     print(f"Decision False: Saddle, but BTC trend is {btc_price_trend_info.trend_kind.name}.")
                #     return False

                if (avg_div_max :=  disp_trend_info.avg_disp_change / abs(disp_trend_info.max_disp_change)) > 0.6:
                    print(f"Decision False: Saddle, but {avg_div_max=} > 0.6.")
                    return False, "saddle_but_avg_div_max_is_close"

                print("Decision True: Good saddle.")
                return True, "good_saddle"

        if disp_trend_info.monotone_up:
            assert len(disp_trend_info.disps_on_trend) > 1

            if disp_trend_info.disps_on_trend[-2] / disp_trend_info.disps_on_trend[-1] > 0.85:
                print("Decision False: Monotone up, but last 2 disps are too close.")
                return False, "monotone_up_but_last_2_close"

            if abs(disp_trend_info.max_disp_change) > 0.6:
                print("Decision False: Monotone up on big disp change.")
                return False, "monotone_up_but_big_disp_change"
            # if max(disp_trend_info.disps_on_trend) > 0.59:
            #     print("Decision False: Monotone up on big disp.")
            #     return False
            elif disp_trend_info.max_disp_change < 0.2:
                print("Decision False: Monotone up on small disp change.")
                return False, "monotone_up_but_small_disp_change"
            elif (max_volume := abs(max(last_trends_info.volume_and_len[:4], key=lambda x: abs(x[0]))[0])) > 0.025:
                print(f"Decision False: Monotone up, but big volume trend in last 3: {max_volume=}.")
                return False, "monotone_up_but_volume_in_last_n_big"
            elif (avg_div_max :=  disp_trend_info.avg_disp_change / abs(disp_trend_info.max_disp_change)) > 0.9:
                print(f"Decision False: Monotone up, but {avg_div_max=} > 0.9.")
                return False, "monotone_up_but_avg_div_max_is_close"
            else:
                print(f"Decision True: Good monotone up.")
                return True, "good_monotone_up"

        print("Decision False. No triggers.")
        return False, "no_triggers"