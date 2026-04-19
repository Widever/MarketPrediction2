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
    btc_eth_log_return_ratio: float = field(metadata={"intervals": [-1.14, -0.129, 0.882, 1.894]})
    btc_ada_log_return_ratio: float = field(metadata={"intervals": [-0.533, 0.451, 1.435, 2.419]})
    btc_doge_log_return_ratio: float = field(metadata={"intervals": [-1.316, 0.513, 2.342, 4.171]})
    btc_xrp_log_return_ratio: float = field(metadata={"intervals": [-0.29, 1.54, 3.371, 5.201]})
    eth_ada_log_return_ratio: float = field(metadata={"intervals": [0.338, 2.903, 5.468, 8.033]})
    eth_doge_log_return_ratio: float = field(metadata={"intervals": [1.081, 2.897, 4.712, 6.527]})
    eth_xrp_log_return_ratio: float = field(metadata={"intervals": [3.16, 6.737, 10.314, 13.89]})
    doge_ada_log_return_ratio: float = field(metadata={"intervals": [1.569, 2.868, 4.166, 5.465]})
    doge_sui_log_return_ratio: float = field(metadata={"intervals": [-0.623, 1.209, 3.042, 4.875]})
    doge_xrp_log_return_ratio: float = field(metadata={"intervals": [-1.226, -0.33, 0.566, 1.462]})
    avg_log_return_ratio: float = field(metadata={"intervals": [0.468, 0.562, 0.655, 0.749]})
    ######
    btc_eth_amp_ratio: float = field(metadata={"intervals": [0.39, 0.676, 0.963, 1.249]})
    btc_ada_amp_ratio: float = field(metadata={"intervals": [0.305, 0.557, 0.808, 1.06]})
    btc_doge_amp_ratio: float = field(metadata={"intervals": [0.324, 0.596, 0.868, 1.14]})
    btc_xrp_amp_ratio: float = field(metadata={"intervals": [0.369, 0.672, 0.974, 1.277]})
    eth_ada_amp_ratio: float = field(metadata={"intervals": [0.54, 0.883, 1.227, 1.571]})
    eth_doge_amp_ratio: float = field(metadata={"intervals": [0.562, 0.906, 1.251, 1.595]})
    eth_xrp_amp_ratio: float = field(metadata={"intervals": [0.7, 1.173, 1.646, 2.119]})
    doge_ada_amp_ratio: float = field(metadata={"intervals": [0.775, 1.21, 1.646, 2.081]})
    doge_sui_amp_ratio: float = field(metadata={"intervals": [0.724, 1.178, 1.633, 2.088]})
    doge_xrp_amp_ratio: float = field(metadata={"intervals": [1.046, 1.701, 2.356, 3.011]})
    avg_ampl_ratio: float = field(metadata={"intervals": [0.605, 0.797, 0.989, 1.181]})
    ######
    btc_eth_ch_from_peak_ratio: float = field(metadata={"intervals": [-3.689, -1.995, -0.301, 1.393]})
    btc_ada_ch_from_peak_ratio: float = field(metadata={"intervals": [-3.617, -2.242, -0.868, 0.507]})
    btc_doge_ch_from_peak_ratio: float = field(metadata={"intervals": [-3.816, -2.268, -0.72, 0.828]})
    btc_xrp_ch_from_peak_ratio: float = field(metadata={"intervals": [-3.48, -1.974, -0.469, 1.037]})
    eth_ada_ch_from_peak_ratio: float = field(metadata={"intervals": [-1.563, -0.741, 0.082, 0.904]})
    eth_doge_ch_from_peak_ratio: float = field(metadata={"intervals": [-2.09, -1.197, -0.303, 0.59]})
    eth_xrp_ch_from_peak_ratio: float = field(metadata={"intervals": [-2.18, -1.169, -0.158, 0.853]})
    doge_ada_ch_from_peak_ratio: float = field(metadata={"intervals": [-2.031, -1.119, -0.207, 0.705]})
    doge_sui_ch_from_peak_ratio: float = field(metadata={"intervals": [-1.691, -0.794, 0.103, 0.999]})
    doge_xrp_ch_from_peak_ratio: float = field(metadata={"intervals": [-1.277, -0.389, 0.499, 1.386]})
    avg_ch_from_peak_ratio: float = field(metadata={"intervals": [-0.538, -0.098, 0.342, 0.782]})
    #####
    btc_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})
    eth_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})
    ada_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})
    doge_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})
    xrp_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})
    sui_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})
    #####
    btc_trend_ch_from_peak: float = field(metadata={"intervals": [-0.04, -0.003, 0.033, 0.069]})
    eth_trend_ch_from_peak: float = field(metadata={"intervals": [-0.039, -0.009, 0.021, 0.051]})
    ada_trend_ch_from_peak: float = field(metadata={"intervals": [-0.044, -0.013, 0.018, 0.048]})
    doge_trend_ch_from_peak: float = field(metadata={"intervals": [-0.042, -0.009, 0.023, 0.055]})
    xrp_trend_ch_from_peak: float = field(metadata={"intervals": [-0.036, -0.007, 0.023, 0.052]})
    sui_trend_ch_from_peak: float = field(metadata={"intervals": [-0.042, -0.011, 0.021, 0.052]})
    avg_trend_ch_from_peak: float = field(metadata={"intervals": [-0.034, -0.01, 0.013, 0.036]})

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
    len_from_last_peak: float = 0
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

def _avg_point_values(point_values_: PointValues, attrs: tuple[str, ...]) -> float:
    return sum([getattr(point_values_, attr) for attr in attrs]) / len(attrs)

def point_values(symbol: str, timestamp: int) -> PointValues:

    point_v = PointValues(
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
        avg_log_return_ratio=0.0,
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
        avg_ampl_ratio=0.0,
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
        avg_ch_from_peak_ratio=0.0,
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
        avg_trend_ch_from_peak=0.0,
    )
    point_v.avg_log_return_ratio = _avg_point_values(
        point_v,
        (
            "btc_eth_log_return_ratio",
            "btc_ada_log_return_ratio",
            "btc_doge_log_return_ratio",
            "btc_xrp_log_return_ratio",
            "eth_ada_log_return_ratio",
            "eth_doge_log_return_ratio",
            "eth_xrp_log_return_ratio",
            "doge_ada_log_return_ratio",
            "doge_xrp_log_return_ratio",
            "doge_sui_log_return_ratio",
        )
    )
    
    point_v.avg_ampl_ratio = _avg_point_values(
        point_v,
        (
            "btc_eth_amp_ratio",
            "btc_ada_amp_ratio",
            "btc_doge_amp_ratio",
            "btc_xrp_amp_ratio",
            "eth_ada_amp_ratio",
            "eth_doge_amp_ratio",
            "eth_xrp_amp_ratio",
            "doge_ada_amp_ratio",
            "doge_xrp_amp_ratio",
            "doge_sui_amp_ratio",
        )
    )

    point_v.avg_ch_from_peak_ratio = _avg_point_values(
        point_v,
        (
            "btc_eth_ch_from_peak_ratio",
            "btc_ada_ch_from_peak_ratio",
            "btc_doge_ch_from_peak_ratio",
            "btc_xrp_ch_from_peak_ratio",
            "eth_ada_ch_from_peak_ratio",
            "eth_doge_ch_from_peak_ratio",
            "eth_xrp_ch_from_peak_ratio",
            "doge_ada_ch_from_peak_ratio",
            "doge_xrp_ch_from_peak_ratio",
            "doge_sui_ch_from_peak_ratio",
        )
    )

    point_v.avg_trend_ch_from_peak = _avg_point_values(
        point_v,
        (
            "btc_trend_ch_from_peak",
            "eth_trend_ch_from_peak",
            "ada_trend_ch_from_peak",
            "doge_trend_ch_from_peak",
            "xrp_trend_ch_from_peak",
            "sui_trend_ch_from_peak",
        )
    )
    return point_v

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
    last_peak_idx = None

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
            last_peak_idx = last_high_idx
            trend = PriceTrend.DOWN

            last_low = low
            last_low_idx = i

        # Ріст більше threshold → фіксуємо peak_down
        elif rise_from_low >= threshold and trend != PriceTrend.UP:
            result.loc[last_low_idx, "peak_down"] = True
            last_peak_price = last_low
            last_peak_idx = last_low_idx
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
            result.loc[i, "len_from_last_peak"] = i - last_peak_idx
        else:
            result.loc[i, "change_from_last_peak"] = 0.0
            result.loc[i, "len_from_last_peak"] = 0

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
            len_from_last_peak=row["len_from_last_peak"],
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

def get_intervals_from_data(col_name: str, n: int, left_pct: float, right_pct: float) -> str:
    marked_points_df = pd.read_csv(f"{data_dir}/marked_points_frozen.csv")
    marked_points_df = marked_points_df.round(5)

    sorted_col = list(sorted(marked_points_df[col_name]))
    cut_left = int(len(sorted_col) * left_pct)
    cut_right = int(len(sorted_col) * right_pct)

    values = sorted_col[cut_left:]
    if cut_right > 0:
        values = values[:-cut_right]

    start_v = values[0]
    interval_width = (values[-1] - start_v) / n
    intervals = [round(start_v + interval_width * i, 3) for i in range(1, n)]
    return str(intervals)


def adjust_point_values():
    v = get_intervals_from_data("avg_log_return_ratio", 5, 0.004, 0.004)
    point_values_lines = [
    f'btc_eth_log_return_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_eth_log_return_ratio", 5, 0.01, 0.01)}}})',
    f'btc_ada_log_return_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_ada_log_return_ratio", 5, 0.02, 0.02)}}})',
    f'btc_doge_log_return_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_doge_log_return_ratio", 5, 0.01, 0.01)}}})',
    f'btc_xrp_log_return_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_xrp_log_return_ratio", 5, 0.01, 0.01)}}})',
    f'eth_ada_log_return_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_ada_log_return_ratio", 5, 0.01, 0.01)}}})',
    f'eth_doge_log_return_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_doge_log_return_ratio", 5, 0.01, 0.01)}}})',
    f'eth_xrp_log_return_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_xrp_log_return_ratio", 5, 0.01, 0.01)}}})',
    f'doge_ada_log_return_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_ada_log_return_ratio", 5, 0.01, 0.01)}}})',
    f'doge_sui_log_return_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_sui_log_return_ratio", 5, 0.01, 0.01)}}})',
    f'doge_xrp_log_return_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_xrp_log_return_ratio", 5, 0.01, 0.01)}}})',
    f'avg_log_return_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("avg_log_return_ratio", 5, 0.004, 0.004)}}})',
    '######',
    f'btc_eth_amp_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_eth_amp_ratio", 5, 0.01, 0.01)}}})',
    f'btc_ada_amp_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_ada_amp_ratio", 5, 0.01, 0.01)}}})',
    f'btc_doge_amp_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_doge_amp_ratio", 5, 0.01, 0.01)}}})',
    f'btc_xrp_amp_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_xrp_amp_ratio", 5, 0.01, 0.01)}}})',
    f'eth_ada_amp_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_ada_amp_ratio", 5, 0.01, 0.01)}}})',
    f'eth_doge_amp_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_doge_amp_ratio", 5, 0.01, 0.01)}}})',
    f'eth_xrp_amp_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_xrp_amp_ratio", 5, 0.01, 0.01)}}})',
    f'doge_ada_amp_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_ada_amp_ratio", 5, 0.01, 0.01)}}})',
    f'doge_sui_amp_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_sui_amp_ratio", 5, 0.01, 0.01)}}})',
    f'doge_xrp_amp_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_xrp_amp_ratio", 5, 0.01, 0.01)}}})',
    f'avg_ampl_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("avg_ampl_ratio", 5, 0.01, 0.01)}}})',
    '######',
    f'btc_eth_ch_from_peak_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_eth_ch_from_peak_ratio", 5, 0.01, 0.01)}}})',
    f'btc_ada_ch_from_peak_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_ada_ch_from_peak_ratio", 5, 0.01, 0.01)}}})',
    f'btc_doge_ch_from_peak_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_doge_ch_from_peak_ratio", 5, 0.01, 0.01)}}})',
    f'btc_xrp_ch_from_peak_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_xrp_ch_from_peak_ratio", 5, 0.01, 0.01)}}})',
    f'eth_ada_ch_from_peak_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_ada_ch_from_peak_ratio", 5, 0.01, 0.01)}}})',
    f'eth_doge_ch_from_peak_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_doge_ch_from_peak_ratio", 5, 0.01, 0.01)}}})',
    f'eth_xrp_ch_from_peak_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_xrp_ch_from_peak_ratio", 5, 0.01, 0.01)}}})',
    f'doge_ada_ch_from_peak_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_ada_ch_from_peak_ratio", 5, 0.01, 0.01)}}})',
    f'doge_sui_ch_from_peak_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_sui_ch_from_peak_ratio", 5, 0.01, 0.01)}}})',
    f'doge_xrp_ch_from_peak_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_xrp_ch_from_peak_ratio", 5, 0.01, 0.01)}}})',
    f'avg_ch_from_peak_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("avg_ch_from_peak_ratio", 5, 0.01, 0.01)}}})',
    '#####',
    'btc_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})',
    'eth_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})',
    'ada_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})',
    'doge_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})',
    'xrp_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})',
    'sui_trend_kind: str = field(metadata={"enum": ["rising", "falling", "flat"]})',
    '#####',
    f'btc_trend_ch_from_peak: float = field(metadata={{"intervals": {get_intervals_from_data("btc_trend_ch_from_peak", 5, 0.01, 0.01)}}})',
    f'eth_trend_ch_from_peak: float = field(metadata={{"intervals": {get_intervals_from_data("eth_trend_ch_from_peak", 5, 0.01, 0.01)}}})',
    f'ada_trend_ch_from_peak: float = field(metadata={{"intervals": {get_intervals_from_data("ada_trend_ch_from_peak", 5, 0.01, 0.01)}}})',
    f'doge_trend_ch_from_peak: float = field(metadata={{"intervals": {get_intervals_from_data("doge_trend_ch_from_peak", 5, 0.01, 0.01)}}})',
    f'xrp_trend_ch_from_peak: float = field(metadata={{"intervals": {get_intervals_from_data("xrp_trend_ch_from_peak", 5, 0.01, 0.01)}}})',
    f'sui_trend_ch_from_peak: float = field(metadata={{"intervals": {get_intervals_from_data("sui_trend_ch_from_peak", 5, 0.01, 0.01)}}})',
    f'avg_trend_ch_from_peak: float = field(metadata={{"intervals": {get_intervals_from_data("avg_trend_ch_from_peak", 5, 0.01, 0.01)}}})',
    ]
    print('\n'.join(point_values_lines))

