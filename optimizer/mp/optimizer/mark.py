import dataclasses
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, auto
from statistics import mean
from typing import get_origin, get_args, Any

import numpy as np
import pandas as pd

import mp.optimizer.init_data as data

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"optimize_main_dir")

max_trend_len = 50
flat_trend_limit = 0.02

sl_k = 0.98
sell_k = 1.01

@dataclass(slots=True)
class PointValues:
    btc_eth_log_return_ratio: float = field(metadata={"intervals": [0.0, 0.5, 1.1, 2.1]})
    btc_ada_log_return_ratio: float = field(metadata={"intervals": [0.0, 0.25, 0.62, 1.2]})
    btc_doge_log_return_ratio: float = field(metadata={"intervals": [-0.2, 0.2, 0.6, 1.2]})
    btc_xrp_log_return_ratio: float = field(metadata={"intervals": [-0.2, 0.2, 0.6, 1.2]})
    eth_ada_log_return_ratio: float = field(metadata={"intervals": [0.0, 0.4, 0.8, 1.6]})
    eth_doge_log_return_ratio: float = field(metadata={"intervals": [0.25, 0.75, 1.0, 1.6]})
    eth_xrp_log_return_ratio: float = field(metadata={"intervals": [0.25, 0.75, 1.0, 1.6]})
    doge_ada_log_return_ratio: float = field(metadata={"intervals": [0.0, 0.4, 0.8, 1.6]})
    doge_sui_log_return_ratio: float = field(metadata={"intervals": [0.0, 0.4, 0.8, 1.6]})
    doge_xrp_log_return_ratio: float = field(metadata={"intervals": [0.0, 0.4, 0.8, 1.6]})
    ######
    btc_eth_amp_ratio: float = field(metadata={"intervals": [0.39, 0.52, 0.65, 1.2]})
    btc_ada_amp_ratio: float = field(metadata={"intervals": [0.3, 0.48, 0.6, 1.2]})
    btc_doge_amp_ratio: float = field(metadata={"intervals": [0.25, 0.36, 0.54, 1.1]})
    btc_xrp_amp_ratio: float = field(metadata={"intervals": [0.25, 0.36, 0.54, 1.1]})
    eth_ada_amp_ratio: float = field(metadata={"intervals": [0.48, 0.67, 0.9, 1.8]})
    eth_doge_amp_ratio: float = field(metadata={"intervals": [0.5, 0.7, 0.87, 1.8]})
    eth_xrp_amp_ratio: float = field(metadata={"intervals": [0.5, 0.7, 0.87, 1.8]})
    doge_ada_amp_ratio: float = field(metadata={"intervals": [0.5, 0.7, 0.87, 1.8]})
    doge_sui_amp_ratio: float = field(metadata={"intervals": [0.5, 0.7, 0.87, 1.8]})
    doge_xrp_amp_ratio: float = field(metadata={"intervals": [0.5, 0.7, 0.87, 1.8]})
    ######
    btc_eth_ch_from_peak_ratio: float = field(metadata={"intervals": [-1.0, 0.5, 1.0, 2.0]})
    btc_ada_ch_from_peak_ratio: float = field(metadata={"intervals": [-1.0, 0.5, 1.0, 2.0]})
    btc_doge_ch_from_peak_ratio: float = field(metadata={"intervals": [-1.0, 0.5, 1.0, 2.0]})
    btc_xrp_ch_from_peak_ratio: float = field(metadata={"intervals": [-1.0, 0.5, 1.0, 2.0]})
    eth_ada_ch_from_peak_ratio: float = field(metadata={"intervals": [-0.5, 0.8, 1.4, 2.5]})
    eth_doge_ch_from_peak_ratio: float = field(metadata={"intervals": [-0.5, 0.8, 1.4, 2.5]})
    eth_xrp_ch_from_peak_ratio: float = field(metadata={"intervals": [-0.5, 0.8, 1.4, 2.5]})
    doge_ada_ch_from_peak_ratio: float = field(metadata={"intervals": [-0.2, 0.8, 1.2, 2.5]})
    doge_sui_ch_from_peak_ratio: float = field(metadata={"intervals": [-1.0, 1.0, 1.5, 2.5]})
    doge_xrp_ch_from_peak_ratio: float = field(metadata={"intervals": [-0.5, 0.5, 1.0, 2.5]})
    #####
    btc_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})
    eth_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})
    ada_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})
    doge_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})
    xrp_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})
    sui_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})
    #####
    btc_trend_ch_from_peak: float = field(metadata={"intervals": [-0.04, -0.03, -0.02, -0.01, 0.01]})
    eth_trend_ch_from_peak: float = field(metadata={"intervals": [-0.04, -0.03, -0.02, -0.01, 0.01]})
    ada_trend_ch_from_peak: float = field(metadata={"intervals": [-0.04, -0.03, -0.02, -0.01, 0.01]})
    doge_trend_ch_from_peak: float = field(metadata={"intervals": [-0.04, -0.03, -0.02, -0.01, 0.01]})
    xrp_trend_ch_from_peak: float = field(metadata={"intervals": [-0.04, -0.03, -0.02, -0.01, 0.01]})
    sui_trend_ch_from_peak: float = field(metadata={"intervals": [-0.04, -0.03, -0.02, -0.01, 0.01]})

@dataclass(slots=True)
class MarkedPoint:
    index: int
    timestamp: int
    values: PointValues

    sl_price_limit: float
    sell_price_limit: float

    sl: bool = False
    scope: int = 0

    rising: bool = False
    falling: bool = False
    flat: bool = False
    peak_up: bool = False
    peak_down: bool = False
    change_from_last_peak: float = 0.0
    ampl: float = 0.0



class PriceTrend(IntEnum):
    UP = auto()
    DOWN = auto()
    FLAT = auto()
    UNKNOWN = auto()


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


def get_price_trend(symbol: str, timestamp: int) -> PriceTrendInfo:

    ohlcv_df = data.CURRENCY_DATA_DICT[symbol].ohlcv_df

    matches = ohlcv_df.index[ohlcv_df["timestamp"] == timestamp]

    if matches.empty:
        raise RuntimeError(f"Not found timestamp {timestamp} is ohlcv_df for symbol {symbol}.")

    for_index = matches[0]

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

def _add_interval_tag_columns(df: pd.DataFrame, col: str, intervals: list[float]) -> pd.DataFrame:

    shift_interval = 0.0
    for i, gt_v in enumerate(intervals):

        gt_v += shift_interval

        # if i == 0:
        df[f"#tag_{col}_lt_{gt_v}"] = df[col] < gt_v
        df[f"#tag_{col}_gt_{gt_v}"] = df[col] >= gt_v

    return df

def _add_eq_tag_columns(df: pd.DataFrame, col: str, possible_values: list[int | float | str]) -> pd.DataFrame:

    for eq_v in possible_values:
        df[f"#tag_{col}_eq_{eq_v}"] = df[col] == eq_v

    return df

def _add_str_tag_columns(df: pd.DataFrame, col: str, values=None) -> pd.DataFrame:

    raise RuntimeError("Need to refactor.")
    if values is None:
        unique_values = df[col].unique()
    else:
        unique_values = values

    for val in unique_values:
        df[f"#tag_{col}_{val}"] = df[col] == val
    return df

def _add_bool_tag_columns(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[f"#tag_{col}_true"] = df[col]
    df[f"#tag_{col}_false"] = ~df[col]
    return df

def add_tags_for_point_values(marked_points_values_df: pd.DataFrame) -> pd.DataFrame:

    for f in dataclasses.fields(PointValues):
        typ = f.type
        # unwrap Optional[...]
        if get_origin(typ) is not None and get_origin(typ) is not list:
            args = get_args(typ)
            if args:
                typ = args[0]

        if typ is str:
            possible_values = f.metadata.get("enum", None)
            if possible_values is not None:
                marked_points_values_df = _add_eq_tag_columns(marked_points_values_df, f.name, possible_values)
            else:
                raise RuntimeError("Expect enum metadata.")
        elif typ in (int, float):
            intervals = f.metadata.get("intervals", None)
            possible_values = f.metadata.get("enum", None)
            if intervals is not None:
                marked_points_values_df = _add_interval_tag_columns(marked_points_values_df, f.name, intervals)
            elif possible_values is not None:
                marked_points_values_df = _add_eq_tag_columns(marked_points_values_df, f.name, possible_values)
            else:
                raise RuntimeError("Expect intervals or enum metadata.")
        elif typ is bool:
            marked_points_values_df = _add_bool_tag_columns(marked_points_values_df, f.name)

    return marked_points_values_df

def _interval_str_value(intervals: list[float], value: float) -> str:
    for bound in intervals:
        if value < bound:
            return f"lt_{bound}"

    return f"gt_{intervals[-1]}"

def _log_return_ratio(symbol_1: str, symbol_2: str, timestamp: int) -> float:
    df = data.DEVIATION_K_DICT[(symbol_1, symbol_2)]
    idx = df["timestamp"].values
    values = df["deviation_k"].values

    return values[idx == timestamp][0]

def _ampl_ratio(symbol_1: str, symbol_2: str, timestamp: int) -> float:
    df = data.AMPL_RATIO_DICT[(symbol_1, symbol_2)]
    idx = df["timestamp"].values
    values = df["ampl_ratio"].values

    return values[idx == timestamp][0]

def _trend_ch_from_peak_ratio(symbol_1: str, symbol_2: str, timestamp: int) -> float:
    symbol_1_trend_df = data.PEAKS_AND_TREND_DICT[symbol_1]
    symbol_2_trend_df = data.PEAKS_AND_TREND_DICT[symbol_2]

    idx = symbol_1_trend_df["timestamp"].values

    trend_value_ratio = symbol_1_trend_df["change_from_last_peak"].values[idx == timestamp][0] / symbol_2_trend_df["change_from_last_peak"].values[idx == timestamp][0]

    return trend_value_ratio

def _trend_len(symbol: str, timestamp: int) -> int:
    trend_df = data.TREND_DICT[symbol]
    idx = trend_df["timestamp"].values
    values = trend_df["trend_len"].values

    return values[idx == timestamp][0]

def _trend_ch_from_peak(symbol: str, timestamp: int) -> float:
    trend_df = data.PEAKS_AND_TREND_DICT[symbol]
    idx = trend_df["timestamp"].values
    values = trend_df["change_from_last_peak"].values

    return values[idx == timestamp][0]

def _trend_kind(symbol: str, timestamp: int) -> str:
    trend_df = data.PEAKS_AND_TREND_DICT[symbol]
    idx = trend_df["timestamp"].values
    rising = trend_df["rising"].values[idx == timestamp][0]
    flat = trend_df["flat"].values[idx == timestamp][0]
    falling = trend_df["falling"].values[idx == timestamp][0]
    
    if rising:
        return "rising"
    elif flat:
        return "flat"
    elif falling:
        return "falling"
    else:
        return "unknown"

def point_values(symbol: str, timestamp: int) -> PointValues:

    return PointValues(
        btc_eth_log_return_ratio=_log_return_ratio("BTCUSDT", "ETHUSDT", timestamp),
        btc_ada_log_return_ratio=_log_return_ratio("BTCUSDT", "ADAUSDT", timestamp),
        btc_doge_log_return_ratio=_log_return_ratio("BTCUSDT", "DOGEUSDT", timestamp),
        btc_xrp_log_return_ratio=_log_return_ratio("BTCUSDT", "XRPUSDT", timestamp),
        eth_ada_log_return_ratio=_log_return_ratio("ETHUSDT", "ADAUSDT", timestamp),
        eth_doge_log_return_ratio=_log_return_ratio("ETHUSDT", "DOGEUSDT", timestamp),
        eth_xrp_log_return_ratio=_log_return_ratio("ETHUSDT", "XRPUSDT", timestamp),
        doge_ada_log_return_ratio=_log_return_ratio("DOGEUSDT", "ADAUSDT", timestamp),
        doge_xrp_log_return_ratio=_log_return_ratio("DOGEUSDT", "XRPUSDT", timestamp),
        doge_sui_log_return_ratio=_log_return_ratio("DOGEUSDT", "SUIUSDT", timestamp),
        #############
        btc_eth_amp_ratio=_ampl_ratio("BTCUSDT", "ETHUSDT", timestamp),
        btc_ada_amp_ratio=_ampl_ratio("BTCUSDT", "ADAUSDT", timestamp),
        btc_doge_amp_ratio=_ampl_ratio("BTCUSDT", "DOGEUSDT", timestamp),
        btc_xrp_amp_ratio=_ampl_ratio("BTCUSDT", "XRPUSDT", timestamp),
        eth_ada_amp_ratio=_ampl_ratio("ETHUSDT", "ADAUSDT", timestamp),
        eth_doge_amp_ratio=_ampl_ratio("ETHUSDT", "DOGEUSDT", timestamp),
        eth_xrp_amp_ratio=_ampl_ratio("ETHUSDT", "XRPUSDT", timestamp),
        doge_ada_amp_ratio=_ampl_ratio("DOGEUSDT", "ADAUSDT", timestamp),
        doge_xrp_amp_ratio=_ampl_ratio("DOGEUSDT", "XRPUSDT", timestamp),
        doge_sui_amp_ratio=_ampl_ratio("DOGEUSDT", "SUIUSDT", timestamp),
        ################
        btc_eth_ch_from_peak_ratio=_trend_ch_from_peak_ratio("BTCUSDT", "ETHUSDT", timestamp),
        btc_ada_ch_from_peak_ratio=_trend_ch_from_peak_ratio("BTCUSDT", "ADAUSDT", timestamp),
        btc_doge_ch_from_peak_ratio=_trend_ch_from_peak_ratio("BTCUSDT", "DOGEUSDT", timestamp),
        btc_xrp_ch_from_peak_ratio=_trend_ch_from_peak_ratio("BTCUSDT", "XRPUSDT", timestamp),
        eth_ada_ch_from_peak_ratio=_trend_ch_from_peak_ratio("ETHUSDT", "ADAUSDT", timestamp),
        eth_doge_ch_from_peak_ratio=_trend_ch_from_peak_ratio("ETHUSDT", "DOGEUSDT", timestamp),
        eth_xrp_ch_from_peak_ratio=_trend_ch_from_peak_ratio("ETHUSDT", "XRPUSDT", timestamp),
        doge_ada_ch_from_peak_ratio=_trend_ch_from_peak_ratio("DOGEUSDT", "ADAUSDT", timestamp),
        doge_xrp_ch_from_peak_ratio=_trend_ch_from_peak_ratio("DOGEUSDT", "XRPUSDT", timestamp),
        doge_sui_ch_from_peak_ratio=_trend_ch_from_peak_ratio("DOGEUSDT", "SUIUSDT", timestamp),
        ########
        btc_trend_kind=_trend_kind("BTCUSDT", timestamp),
        eth_trend_kind=_trend_kind("ETHUSDT", timestamp),
        ada_trend_kind=_trend_kind("ADAUSDT", timestamp),
        doge_trend_kind=_trend_kind("DOGEUSDT", timestamp),
        xrp_trend_kind=_trend_kind("XRPUSDT", timestamp),
        sui_trend_kind=_trend_kind("SUIUSDT", timestamp),
        #######
        btc_trend_ch_from_peak=_trend_ch_from_peak("BTCUSDT", timestamp),
        eth_trend_ch_from_peak=_trend_ch_from_peak("ETHUSDT", timestamp),
        ada_trend_ch_from_peak=_trend_ch_from_peak("ADAUSDT", timestamp),
        doge_trend_ch_from_peak=_trend_ch_from_peak("DOGEUSDT", timestamp),
        xrp_trend_ch_from_peak=_trend_ch_from_peak("XRPUSDT", timestamp),
        sui_trend_ch_from_peak=_trend_ch_from_peak("SUIUSDT", timestamp),
    )

def detect_peaks(df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    result = df.copy()

    result["rising"] = False
    result["falling"] = False
    result["flat"] = False
    result["peak_up"] = False
    result["peak_down"] = False
    result["change_from_last_peak"] = 0.0

    trend = PriceTrend.UNKNOWN

    last_high = result.iloc[0]["high"]
    last_low = result.iloc[0]["low"]
    last_high_idx = 0
    last_low_idx = 0

    last_peak_price = None

    for i in range(1, len(result)):
        high = result.iloc[i]["high"]
        low = result.iloc[i]["low"]
        close = result.iloc[i]["close"]

        # Оновлюємо екстремуми
        if high > last_high:
            last_high = high
            last_high_idx = i

        if low < last_low:
            last_low = low
            last_low_idx = i

        drop_from_high = (last_high - low) / last_high
        rise_from_low = (high - last_low) / last_low

        # Падіння більше threshold → фіксуємо peak_up
        if drop_from_high >= threshold and trend != PriceTrend.DOWN:
            result.loc[last_high_idx, "peak_up"] = True
            last_peak_price = last_high
            trend = PriceTrend.DOWN

            last_low = low
            last_low_idx = i

        # Ріст більше threshold → фіксуємо peak_down
        elif rise_from_low >= threshold and trend != PriceTrend.UP:
            result.loc[last_low_idx, "peak_down"] = True
            last_peak_price = last_low
            trend = PriceTrend.UP

            last_high = high
            last_high_idx = i

        # Тренд/flat
        if drop_from_high < threshold and rise_from_low < threshold:
            result.loc[i, "flat"] = True
            trend = PriceTrend.FLAT
        elif trend == PriceTrend.UP:
            result.loc[i, "rising"] = True
        elif trend == PriceTrend.DOWN:
            result.loc[i, "falling"] = True

        # Зміна від останнього піку
        if last_peak_price is not None:
            result.loc[i, "change_from_last_peak"] = (
                low - last_peak_price
            ) / last_peak_price
        else:
            result.loc[i, "change_from_last_peak"] = 0.0

    return result

def mark_data():
    start_time = time.time()
    closed_points: list[MarkedPoint] = []
    symbol = "ADAUSDT"

    os.makedirs(data_dir, exist_ok=True)

    # ohlcv_df = data.CURRENCY_DATA_DICT[symbol].ohlcv_df
    peaks_and_trend_df = data.PEAKS_AND_TREND_DICT[symbol]
    
    opened_points: list[MarkedPoint] = []

    df_len = len(peaks_and_trend_df)
    print(f"OHLCV+peaks+trend df len = {df_len}. Start marking data...")
    for idx, row in peaks_and_trend_df.iterrows():
        if idx % 10000 == 0:
            print(f"Marked {idx}/{df_len} points.")

        if idx < 50:
            continue
        close_price = float(row["close"])
        low_price = float(row["low"])
        high_price = float(row["high"])
        timestamp = int(row["timestamp"])
        ampl = (high_price - low_price) / low_price

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

        point_values_ = point_values(symbol, timestamp)

        sl_limit_price = sl_k * close_price
        sell_limit_price = sell_k * close_price

        new_opened_marked_point = MarkedPoint(
            index=idx,
            timestamp=timestamp,
            values=point_values_,
            sl_price_limit=sl_limit_price,
            sell_price_limit=sell_limit_price,
            
            rising=row["rising"],
            falling=row["falling"],
            flat=row["flat"],
            peak_up=row["peak_up"],
            peak_down=row["peak_down"],
            change_from_last_peak=row["change_from_last_peak"],
            ampl=ampl
        )
        opened_points.append(new_opened_marked_point)

    marked_points_df_data = []
    for marked_point in sorted(closed_points, key=lambda x: x.index):
        marked_point_dict = dataclasses.asdict(marked_point)
        marked_point_dict_values = marked_point_dict.pop("values")
        marked_point_dict.update(marked_point_dict_values)
        marked_points_df_data.append(marked_point_dict)

    marked_points_df = pd.DataFrame(marked_points_df_data)
    marked_points_df = add_tags_for_point_values(marked_points_df)
    marked_points_df = marked_points_df.round(5)
    marked_points_df.to_csv(f"{data_dir}/marked_points_frozen.csv", index=False)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}s.")
    return marked_points_df

def split_marked_data():
    marked_points_df = pd.read_csv(f"{data_dir}/marked_points_frozen.csv")
    marked_points_df = marked_points_df.round(5)
    train_set_size = int(len(marked_points_df) * 0.7)
    first = marked_points_df.iloc[:train_set_size].copy()
    second = marked_points_df.iloc[train_set_size:].copy()

    first.to_csv(f"{data_dir}/marked_points_train.csv", index=False)
    second.to_csv(f"{data_dir}/marked_points_verify.csv", index=False)

def adjust_point_values():
    poin_values_lines = [
    'btc_eth_log_return_ratio: float = field(metadata={"intervals": [0.0, 0.5, 1.1, 2.1]})',
    'btc_ada_log_return_ratio: float = field(metadata={"intervals": [0.0, 0.25, 0.62, 1.2]})',
    'btc_doge_log_return_ratio: float = field(metadata={"intervals": [-0.2, 0.2, 0.6, 1.2]})',
    'btc_xrp_log_return_ratio: float = field(metadata={"intervals": [-0.2, 0.2, 0.6, 1.2]})',
    'eth_ada_log_return_ratio: float = field(metadata={"intervals": [0.0, 0.4, 0.8, 1.6]})',
    'eth_doge_log_return_ratio: float = field(metadata={"intervals": [0.25, 0.75, 1.0, 1.6]})',
    'eth_xrp_log_return_ratio: float = field(metadata={"intervals": [0.25, 0.75, 1.0, 1.6]})',
    'doge_ada_log_return_ratio: float = field(metadata={"intervals": [0.0, 0.4, 0.8, 1.6]})',
    'doge_sui_log_return_ratio: float = field(metadata={"intervals": [0.0, 0.4, 0.8, 1.6]})',
    'doge_xrp_log_return_ratio: float = field(metadata={"intervals": [0.0, 0.4, 0.8, 1.6]})',
    '######',
    'btc_eth_amp_ratio: float = field(metadata={"intervals": [0.39, 0.52, 0.65, 1.2]})',
    'btc_ada_amp_ratio: float = field(metadata={"intervals": [0.3, 0.48, 0.6, 1.2]})',
    'btc_doge_amp_ratio: float = field(metadata={"intervals": [0.25, 0.36, 0.54, 1.1]})',
    'btc_xrp_amp_ratio: float = field(metadata={"intervals": [0.25, 0.36, 0.54, 1.1]})',
    'eth_ada_amp_ratio: float = field(metadata={"intervals": [0.48, 0.67, 0.9, 1.8]})',
    'eth_doge_amp_ratio: float = field(metadata={"intervals": [0.5, 0.7, 0.87, 1.8]})',
    'eth_xrp_amp_ratio: float = field(metadata={"intervals": [0.5, 0.7, 0.87, 1.8]})',
    'doge_ada_amp_ratio: float = field(metadata={"intervals": [0.5, 0.7, 0.87, 1.8]})',
    'doge_sui_amp_ratio: float = field(metadata={"intervals": [0.5, 0.7, 0.87, 1.8]})',
    'doge_xrp_amp_ratio: float = field(metadata={"intervals": [0.5, 0.7, 0.87, 1.8]})',
    '######',
    'btc_eth_ch_from_peak_ratio: float = field(metadata={"intervals": [-1.0, 0.5, 1.0, 2.0]})',
    'btc_ada_ch_from_peak_ratio: float = field(metadata={"intervals": [-1.0, 0.5, 1.0, 2.0]})',
    'btc_doge_ch_from_peak_ratio: float = field(metadata={"intervals": [-1.0, 0.5, 1.0, 2.0]})',
    'btc_xrp_ch_from_peak_ratio: float = field(metadata={"intervals": [-1.0, 0.5, 1.0, 2.0]})',
    'eth_ada_ch_from_peak_ratio: float = field(metadata={"intervals": [-0.5, 0.8, 1.4, 2.5]})',
    'eth_doge_ch_from_peak_ratio: float = field(metadata={"intervals": [-0.5, 0.8, 1.4, 2.5]})',
    'eth_xrp_ch_from_peak_ratio: float = field(metadata={"intervals": [-0.5, 0.8, 1.4, 2.5]})',
    'doge_ada_ch_from_peak_ratio: float = field(metadata={"intervals": [-0.2, 0.8, 1.2, 2.5]})',
    'doge_sui_ch_from_peak_ratio: float = field(metadata={"intervals": [-1.0, 1.0, 1.5, 2.5]})',
    'doge_xrp_ch_from_peak_ratio: float = field(metadata={"intervals": [-0.5, 0.5, 1.0, 2.5]})',
    '#####',
    'btc_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})',
    'eth_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})',
    'ada_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})',
    'doge_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})',
    'xrp_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})',
    'sui_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})',
    '#####',
    'btc_trend_ch_from_peak: float = field(metadata={"intervals": [-0.04, -0.03, -0.02, -0.01, 0.01]})',
    'eth_trend_ch_from_peak: float = field(metadata={"intervals": [-0.04, -0.03, -0.02, -0.01, 0.01]})',
    'ada_trend_ch_from_peak: float = field(metadata={"intervals": [-0.04, -0.03, -0.02, -0.01, 0.01]})',
    'doge_trend_ch_from_peak: float = field(metadata={"intervals": [-0.04, -0.03, -0.02, -0.01, 0.01]})',
    'xrp_trend_ch_from_peak: float = field(metadata={"intervals": [-0.04, -0.03, -0.02, -0.01, 0.01]})',
    'sui_trend_ch_from_peak: float = field(metadata={"intervals": [-0.04, -0.03, -0.02, -0.01, 0.01]})',
    ]
    print('\n'.join(poin_values_lines))

