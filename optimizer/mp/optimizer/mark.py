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
    avg_log_return_ratio: float = field(metadata={"intervals": [-3.555, -0.767, 2.021, 4.809]})
    ######
    btc_log_return: float = field(metadata={"intervals": [-0.003, -0.001, 0.001, 0.002]})
    eth_log_return: float = field(metadata={"intervals": [-0.003, -0.001, 0.001, 0.003]})
    ada_log_return: float = field(metadata={"intervals": [-0.004, -0.001, 0.002, 0.005]})
    doge_log_return: float = field(metadata={"intervals": [-0.005, -0.002, 0.002, 0.005]})
    xrp_log_return: float = field(metadata={"intervals": [-0.004, -0.001, 0.002, 0.004]})
    sui_log_return: float = field(metadata={"intervals": [-0.005, -0.001, 0.002, 0.005]})
    avg_log_return: float = field(metadata={"intervals": [-0.004, -0.001, 0.001, 0.004]})
    ######
    btc_eth_ampl_ratio: float = field(metadata={"intervals": [0.39, 0.676, 0.963, 1.249]})
    btc_ada_ampl_ratio: float = field(metadata={"intervals": [0.305, 0.557, 0.808, 1.06]})
    btc_doge_ampl_ratio: float = field(metadata={"intervals": [0.324, 0.596, 0.868, 1.14]})
    btc_xrp_ampl_ratio: float = field(metadata={"intervals": [0.369, 0.672, 0.974, 1.277]})
    eth_ada_ampl_ratio: float = field(metadata={"intervals": [0.54, 0.883, 1.227, 1.571]})
    eth_doge_ampl_ratio: float = field(metadata={"intervals": [0.562, 0.906, 1.251, 1.595]})
    eth_xrp_ampl_ratio: float = field(metadata={"intervals": [0.7, 1.173, 1.646, 2.119]})
    doge_ada_ampl_ratio: float = field(metadata={"intervals": [0.775, 1.21, 1.646, 2.081]})
    doge_sui_ampl_ratio: float = field(metadata={"intervals": [0.724, 1.178, 1.633, 2.088]})
    doge_xrp_ampl_ratio: float = field(metadata={"intervals": [1.046, 1.701, 2.356, 3.011]})
    avg_ampl_ratio: float = field(metadata={"intervals": [0.605, 0.797, 0.989, 1.181]})
    ######
    btc_ampl: float = field(metadata={"intervals": [0.002, 0.003, 0.005, 0.006]})
    eth_ampl: float = field(metadata={"intervals": [0.002, 0.004, 0.006, 0.008]})
    ada_ampl: float = field(metadata={"intervals": [0.004, 0.006, 0.009, 0.012]})
    doge_ampl: float = field(metadata={"intervals": [0.003, 0.006, 0.009, 0.011]})
    xrp_ampl: float = field(metadata={"intervals": [0.003, 0.006, 0.008, 0.011]})
    sui_ampl: float = field(metadata={"intervals": [0.004, 0.007, 0.01, 0.013]})
    avg_ampl: float = field(metadata={"intervals": [0.003, 0.005, 0.007, 0.01]})
    ######
    btc_eth_drop_from_high_ratio: float = field(metadata={"intervals": [2.162, 4.21, 6.258, 8.306]})
    btc_ada_drop_from_high_ratio: float = field(metadata={"intervals": [1.57, 3.065, 4.56, 6.055]})
    btc_doge_drop_from_high_ratio: float = field(metadata={"intervals": [1.191, 2.326, 3.461, 4.596]})
    btc_xrp_drop_from_high_ratio: float = field(metadata={"intervals": [1.144, 2.235, 3.327, 4.418]})
    eth_ada_drop_from_high_ratio: float = field(metadata={"intervals": [1.101, 2.114, 3.126, 4.139]})
    eth_doge_drop_from_high_ratio: float = field(metadata={"intervals": [0.802, 1.55, 2.298, 3.047]})
    eth_xrp_drop_from_high_ratio: float = field(metadata={"intervals": [0.954, 1.857, 2.76, 3.662]})
    doge_ada_drop_from_high_ratio: float = field(metadata={"intervals": [2.71, 5.219, 7.729, 10.238]})
    doge_sui_drop_from_high_ratio: float = field(metadata={"intervals": [1.948, 3.771, 5.593, 7.416]})
    doge_xrp_drop_from_high_ratio: float = field(metadata={"intervals": [1.555, 3.01, 4.465, 5.919]})
    avg_drop_from_high_ratio: float = field(metadata={"intervals": [1.17, 1.921, 2.672, 3.424]})
    #####
    btc_drop_from_high: float = field(metadata={"intervals": [0.008, 0.016, 0.024, 0.032]})
    eth_drop_from_high: float = field(metadata={"intervals": [0.008, 0.015, 0.021, 0.028]})
    ada_drop_from_high: float = field(metadata={"intervals": [0.013, 0.025, 0.037, 0.048]})
    doge_drop_from_high: float = field(metadata={"intervals": [0.016, 0.031, 0.045, 0.06]})
    xrp_drop_from_high: float = field(metadata={"intervals": [0.014, 0.027, 0.04, 0.053]})
    sui_drop_from_high: float = field(metadata={"intervals": [0.017, 0.031, 0.045, 0.06]})
    avg_drop_from_high: float = field(metadata={"intervals": [0.012, 0.021, 0.03, 0.039]})
    #####
    btc_eth_rise_from_low_ratio: float = field(metadata={"intervals": [2.24, 4.374, 6.508, 8.641]})
    btc_ada_rise_from_low_ratio: float = field(metadata={"intervals": [1.811, 3.541, 5.27, 6.999]})
    btc_doge_rise_from_low_ratio: float = field(metadata={"intervals": [1.231, 2.406, 3.58, 4.755]})
    btc_xrp_rise_from_low_ratio: float = field(metadata={"intervals": [1.496, 2.94, 4.384, 5.828]})
    eth_ada_rise_from_low_ratio: float = field(metadata={"intervals": [1.393, 2.69, 3.987, 5.284]})
    eth_doge_rise_from_low_ratio: float = field(metadata={"intervals": [0.877, 1.699, 2.521, 3.344]})
    eth_xrp_rise_from_low_ratio: float = field(metadata={"intervals": [1.262, 2.472, 3.681, 4.891]})
    doge_ada_rise_from_low_ratio: float = field(metadata={"intervals": [2.957, 5.725, 8.493, 11.261]})
    doge_sui_rise_from_low_ratio: float = field(metadata={"intervals": [2.327, 4.537, 6.747, 8.958]})
    doge_xrp_rise_from_low_ratio: float = field(metadata={"intervals": [2.314, 4.525, 6.737, 8.948]})
    avg_rise_from_low_ratio: float = field(metadata={"intervals": [1.352, 2.251, 3.15, 4.049]})
    #####
    btc_rise_from_low: float = field(metadata={"intervals": [0.011, 0.02, 0.03, 0.039]})
    eth_rise_from_low: float = field(metadata={"intervals": [0.01, 0.017, 0.025, 0.033]})
    ada_rise_from_low: float = field(metadata={"intervals": [0.014, 0.026, 0.039, 0.051]})
    doge_rise_from_low: float = field(metadata={"intervals": [0.02, 0.038, 0.056, 0.074]})
    xrp_rise_from_low: float = field(metadata={"intervals": [0.018, 0.035, 0.052, 0.069]})
    sui_rise_from_low: float = field(metadata={"intervals": [0.02, 0.037, 0.055, 0.072]})
    avg_rise_from_low: float = field(metadata={"intervals": [0.013, 0.023, 0.033, 0.043]})
    #####

@dataclass(slots=True)
class MarkedPoint:
    index: int
    timestamp: int
    values: PointValues

    sl_price_limit: float
    sell_price_limit: float

    sl: bool = False
    scope: int = 0

    peak_up: bool = False
    peak_down: bool = False
    hemi_peak_up: bool = False
    hemi_peak_down: bool = False
    max_hemi_peak_up: bool = False
    max_hemi_peak_down: bool = False
    change_from_last_peak: float = 0.0
    len_from_last_peak: int = 0
    last_peak_type: str = None
    len_peak_to_peak: int = 0


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
    new_cols = {}
    for i, gt_v in enumerate(intervals):

        gt_v += shift_interval

        # if i == 0:
        new_cols[f"#tag_{col}_lt_{gt_v}"] = df[col] < gt_v
        new_cols[f"#tag_{col}_gt_{gt_v}"] = df[col] >= gt_v

    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    return df

def _add_eq_tag_columns(df: pd.DataFrame, col: str, possible_values: list[int | float | str]) -> pd.DataFrame:
    new_cols = {}
    for eq_v in possible_values:
        new_cols[f"#tag_{col}_eq_{eq_v}"] = df[col] == eq_v
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
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
    new_cols = {
        f"#tag_{col}_true": df[col],
        f"#tag_{col}_false": ~df[col]
    }
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
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
    df = data.LOG_RETURN_RATIO_DICT[(symbol_1, symbol_2)]
    idx = df["timestamp"].values
    values = df["log_return_ratio"].values

    return values[idx == timestamp][0]

def _log_return(symbol_1: str, timestamp: int) -> float:
    df = data.LOG_RETURN_DICT[symbol_1]
    idx = df["timestamp"].values
    values = df["log_return"].values

    return values[idx == timestamp][0]

def _ampl_ratio(symbol_1: str, symbol_2: str, timestamp: int) -> float:
    df = data.AMPL_RATIO_DICT[(symbol_1, symbol_2)]
    idx = df["timestamp"].values
    values = df["ampl_ratio"].values

    return values[idx == timestamp][0]

def _ampl(symbol_1: str, timestamp: int) -> float:
    df = data.AMPL_DICT[symbol_1]
    idx = df["timestamp"].values
    values = df["ampl"].values

    return values[idx == timestamp][0]

def _drop_from_high(symbol_1: str, timestamp: int) -> float:
    df = data.PEAKS_AND_TREND_DICT[symbol_1]
    idx = df["timestamp"].values
    values = df["drop_from_high"].values

    return values[idx == timestamp][0]

def _drop_from_high_ratio(symbol_1: str, symbol_2, timestamp: int) -> float:
    df = data.DROP_FROM_HIGH_RATIO_DICT[(symbol_1, symbol_2)]
    idx = df["timestamp"].values
    values = df["drop_from_high_ratio"].values

    return values[idx == timestamp][0]

def _rise_from_low(symbol_1: str, timestamp: int) -> float:
    df = data.PEAKS_AND_TREND_DICT[symbol_1]
    idx = df["timestamp"].values
    values = df["rise_from_low"].values

    return values[idx == timestamp][0]

def _rise_from_low_ratio(symbol_1: str, symbol_2, timestamp: int) -> float:
    df = data.RISE_FROM_LOW_RATIO_DICT[(symbol_1, symbol_2)]
    idx = df["timestamp"].values
    values = df["rise_from_low_ratio"].values

    return values[idx == timestamp][0]

def _trend_ch_from_peak_ratio(symbol_1: str, symbol_2: str, timestamp: int) -> float:
    symbol_1_trend_df = data.PEAKS_AND_TREND_DICT[symbol_1]
    symbol_2_trend_df = data.PEAKS_AND_TREND_DICT[symbol_2]

    idx = symbol_1_trend_df["timestamp"].values

    if not (symbol_2_change_from_last_peak := symbol_2_trend_df["change_from_last_peak"].values[idx == timestamp][0]):
        return 0.0

    trend_value_ratio = symbol_1_trend_df["change_from_last_peak"].values[idx == timestamp][0] / symbol_2_change_from_last_peak

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

def _avg_point_values(point_values_: PointValues, attrs: tuple[str, ...]) -> float:
    values = [getattr(point_values_, attr) for attr in attrs]
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    r = values.sum() / len(attrs)
    return r

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
        ############
        btc_log_return=_log_return("BTCUSDT", timestamp),
        eth_log_return=_log_return("ETHUSDT", timestamp),
        ada_log_return=_log_return("ADAUSDT", timestamp),
        doge_log_return=_log_return("DOGEUSDT", timestamp),
        xrp_log_return=_log_return("XRPUSDT", timestamp),
        sui_log_return=_log_return("SUIUSDT", timestamp),
        avg_log_return=0.0,
        #############
        btc_eth_ampl_ratio=_ampl_ratio("BTCUSDT", "ETHUSDT", timestamp),
        btc_ada_ampl_ratio=_ampl_ratio("BTCUSDT", "ADAUSDT", timestamp),
        btc_doge_ampl_ratio=_ampl_ratio("BTCUSDT", "DOGEUSDT", timestamp),
        btc_xrp_ampl_ratio=_ampl_ratio("BTCUSDT", "XRPUSDT", timestamp),
        eth_ada_ampl_ratio=_ampl_ratio("ETHUSDT", "ADAUSDT", timestamp),
        eth_doge_ampl_ratio=_ampl_ratio("ETHUSDT", "DOGEUSDT", timestamp),
        eth_xrp_ampl_ratio=_ampl_ratio("ETHUSDT", "XRPUSDT", timestamp),
        doge_ada_ampl_ratio=_ampl_ratio("DOGEUSDT", "ADAUSDT", timestamp),
        doge_sui_ampl_ratio=_ampl_ratio("DOGEUSDT", "SUIUSDT", timestamp),
        doge_xrp_ampl_ratio=_ampl_ratio("DOGEUSDT", "XRPUSDT", timestamp),
        avg_ampl_ratio=0.0,
        ################
        btc_ampl=_ampl("BTCUSDT", timestamp),
        eth_ampl=_ampl("ETHUSDT", timestamp),
        ada_ampl=_ampl("ADAUSDT", timestamp),
        doge_ampl=_ampl("DOGEUSDT", timestamp),
        xrp_ampl=_ampl("XRPUSDT", timestamp),
        sui_ampl=_ampl("SUIUSDT", timestamp),
        avg_ampl=0.0,
        #######
        btc_eth_drop_from_high_ratio=_drop_from_high_ratio("BTCUSDT", "ETHUSDT", timestamp),
        btc_ada_drop_from_high_ratio=_drop_from_high_ratio("BTCUSDT", "ADAUSDT", timestamp),
        btc_doge_drop_from_high_ratio=_drop_from_high_ratio("BTCUSDT", "DOGEUSDT", timestamp),
        btc_xrp_drop_from_high_ratio=_drop_from_high_ratio("BTCUSDT", "XRPUSDT", timestamp),
        eth_ada_drop_from_high_ratio=_drop_from_high_ratio("ETHUSDT", "ADAUSDT", timestamp),
        eth_doge_drop_from_high_ratio=_drop_from_high_ratio("ETHUSDT", "DOGEUSDT", timestamp),
        eth_xrp_drop_from_high_ratio=_drop_from_high_ratio("ETHUSDT", "XRPUSDT", timestamp),
        doge_ada_drop_from_high_ratio=_drop_from_high_ratio("DOGEUSDT", "ADAUSDT", timestamp),
        doge_sui_drop_from_high_ratio=_drop_from_high_ratio("DOGEUSDT", "SUIUSDT", timestamp),
        doge_xrp_drop_from_high_ratio=_drop_from_high_ratio("DOGEUSDT", "XRPUSDT", timestamp),
        avg_drop_from_high_ratio=0.0,
        #######
        btc_drop_from_high=_drop_from_high("BTCUSDT", timestamp),
        eth_drop_from_high=_drop_from_high("ETHUSDT", timestamp),
        ada_drop_from_high=_drop_from_high("ADAUSDT", timestamp),
        doge_drop_from_high=_drop_from_high("DOGEUSDT", timestamp),
        xrp_drop_from_high=_drop_from_high("XRPUSDT", timestamp),
        sui_drop_from_high=_drop_from_high("SUIUSDT", timestamp),
        avg_drop_from_high=0.0,
        #######
        btc_eth_rise_from_low_ratio=_rise_from_low_ratio("BTCUSDT", "ETHUSDT", timestamp),
        btc_ada_rise_from_low_ratio=_rise_from_low_ratio("BTCUSDT", "ADAUSDT", timestamp),
        btc_doge_rise_from_low_ratio=_rise_from_low_ratio("BTCUSDT", "DOGEUSDT", timestamp),
        btc_xrp_rise_from_low_ratio=_rise_from_low_ratio("BTCUSDT", "XRPUSDT", timestamp),
        eth_ada_rise_from_low_ratio=_rise_from_low_ratio("ETHUSDT", "ADAUSDT", timestamp),
        eth_doge_rise_from_low_ratio=_rise_from_low_ratio("ETHUSDT", "DOGEUSDT", timestamp),
        eth_xrp_rise_from_low_ratio=_rise_from_low_ratio("ETHUSDT", "XRPUSDT", timestamp),
        doge_ada_rise_from_low_ratio=_rise_from_low_ratio("DOGEUSDT", "ADAUSDT", timestamp),
        doge_sui_rise_from_low_ratio=_rise_from_low_ratio("DOGEUSDT", "SUIUSDT", timestamp),
        doge_xrp_rise_from_low_ratio=_rise_from_low_ratio("DOGEUSDT", "XRPUSDT", timestamp),
        avg_rise_from_low_ratio=0.0,
        #######
        btc_rise_from_low=_rise_from_low("BTCUSDT", timestamp),
        eth_rise_from_low=_rise_from_low("ETHUSDT", timestamp),
        ada_rise_from_low=_rise_from_low("ADAUSDT", timestamp),
        doge_rise_from_low=_rise_from_low("DOGEUSDT", timestamp),
        xrp_rise_from_low=_rise_from_low("XRPUSDT", timestamp),
        sui_rise_from_low=_rise_from_low("SUIUSDT", timestamp),
        avg_rise_from_low=0.0,
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

    point_v.avg_log_return = _avg_point_values(
        point_v,
        (
            "btc_log_return",
            "eth_log_return",
            "ada_log_return",
            "doge_log_return",
            "xrp_log_return",
            "sui_log_return",
        )
    )

    point_v.avg_ampl_ratio = _avg_point_values(
        point_v,
        (
            "btc_eth_ampl_ratio",
            "btc_ada_ampl_ratio",
            "btc_doge_ampl_ratio",
            "btc_xrp_ampl_ratio",
            "eth_ada_ampl_ratio",
            "eth_doge_ampl_ratio",
            "eth_xrp_ampl_ratio",
            "doge_ada_ampl_ratio",
            "doge_xrp_ampl_ratio",
            "doge_sui_ampl_ratio",
        )
    )

    point_v.avg_ampl = _avg_point_values(
        point_v,
        (
            "btc_ampl",
            "eth_ampl",
            "ada_ampl",
            "doge_ampl",
            "xrp_ampl",
            "sui_ampl",
        )
    )

    point_v.avg_drop_from_high_ratio = _avg_point_values(
        point_v,
        (
            "btc_eth_drop_from_high_ratio",
            "btc_ada_drop_from_high_ratio",
            "btc_doge_drop_from_high_ratio",
            "btc_xrp_drop_from_high_ratio",
            "eth_ada_drop_from_high_ratio",
            "eth_doge_drop_from_high_ratio",
            "eth_xrp_drop_from_high_ratio",
            "doge_ada_drop_from_high_ratio",
            "doge_xrp_drop_from_high_ratio",
            "doge_sui_drop_from_high_ratio",
        )
    )

    point_v.avg_drop_from_high = _avg_point_values(
        point_v,
        (
            "btc_drop_from_high",
            "eth_drop_from_high",
            "ada_drop_from_high",
            "doge_drop_from_high",
            "xrp_drop_from_high",
            "sui_drop_from_high",
        )
    )

    point_v.avg_rise_from_low_ratio = _avg_point_values(
        point_v,
        (
            "btc_eth_rise_from_low_ratio",
            "btc_ada_rise_from_low_ratio",
            "btc_doge_rise_from_low_ratio",
            "btc_xrp_rise_from_low_ratio",
            "eth_ada_rise_from_low_ratio",
            "eth_doge_rise_from_low_ratio",
            "eth_xrp_rise_from_low_ratio",
            "doge_ada_rise_from_low_ratio",
            "doge_xrp_rise_from_low_ratio",
            "doge_sui_rise_from_low_ratio",
        )
    )

    point_v.avg_rise_from_low = _avg_point_values(
        point_v,
        (
            "btc_rise_from_low",
            "eth_rise_from_low",
            "ada_rise_from_low",
            "doge_rise_from_low",
            "xrp_rise_from_low",
            "sui_rise_from_low",
        )
    )
    
    return point_v

def detect_trends(peaks_df: pd.DataFrame) -> pd.DataFrame:
    result = peaks_df.copy().reset_index(drop=True)

    result["change_from_last_peak"] = 0.0
    result["len_from_last_peak"] = 0.0
    result["len_peak_to_peak"] = 0.0
    result["last_peak_type"] = None

    last_peak_price = None
    last_peak_idx = None
    last_peak_type = None

    for i in range(1, len(result)):
        low = result.iloc[i]["low"]
        high = result.iloc[i]["high"]

        if last_peak_price is not None:
            if last_peak_type == "up":
                change_from_last_peak = (last_peak_price - low) / last_peak_price
            elif last_peak_type == "down":
                change_from_last_peak = (high - last_peak_price) / last_peak_price
            else:
                raise ValueError("Unknown type")

            result.loc[i, "change_from_last_peak"] = change_from_last_peak
            result.loc[i, "len_from_last_peak"] = i - last_peak_idx
            result.loc[i, "last_peak_type"] = last_peak_type
        else:
            result.loc[i, "change_from_last_peak"] = 0.0
            result.loc[i, "len_from_last_peak"] = 0
            result.loc[i, "last_peak_type"] = None

        peak_up = result.iloc[i]["peak_up"]
        peak_down = result.iloc[i]["peak_down"]

        if peak_up:
            last_peak_price = high
            last_peak_idx = i
            last_peak_type = "up"
        elif peak_down:
            last_peak_price = low
            last_peak_idx = i
            last_peak_type = "down"

    current_len_peak_to_peak = 0
    reset: bool = True
    for j in range(len(result)-1, -1, -1):
        if reset:
            current_len_peak_to_peak = result.iloc[j]["len_from_last_peak"]
            reset = False

        result.loc[j, "len_peak_to_peak"] = current_len_peak_to_peak

        peak_up = result.iloc[j]["peak_up"]
        peak_down = result.iloc[j]["peak_down"]

        if peak_up or peak_down:
            reset = True

    return result

def detect_hemi_peaks(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    result = df.copy().reset_index(drop=True)

    result["hemi_peak_up"] = False
    result["hemi_peak_down"] = False
    result["max_hemi_peak_up"] = False
    result["max_hemi_peak_down"] = False

    trend = PriceTrend.UNKNOWN

    last_high = result.iloc[0]["high"]
    last_low = result.iloc[0]["low"]
    last_high_idx = 0
    last_low_idx = 0

    for i in range(1, len(result)):
        high = result.iloc[i]["high"]
        low = result.iloc[i]["low"]

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
            result.loc[last_high_idx, "hemi_peak_up"] = True
            trend = PriceTrend.DOWN

            last_low = low
            last_low_idx = i

        # Ріст більше threshold → фіксуємо peak_down
        elif rise_from_low >= threshold and trend != PriceTrend.UP:
            result.loc[last_low_idx, "hemi_peak_down"] = True
            trend = PriceTrend.UP

            last_high = high
            last_high_idx = i

    # --- Compute max_hemi_peak_up / max_hemi_peak_down between consecutive peaks ---
    _mark_max_hemi_peaks(result)

    return result


def _mark_max_hemi_peaks(result: pd.DataFrame) -> None:
    """
    For each segment between two consecutive peaks (peak_up or peak_down),
    mark the hemi_peak_up with the highest 'high' as max_hemi_peak_up,
    and the hemi_peak_down with the lowest 'low' as max_hemi_peak_down.
    Hemi-peaks that coincide with an actual peak are excluded.
    """
    peak_mask = result["peak_up"] | result["peak_down"]
    peak_indices = result.index[peak_mask].tolist()

    if len(peak_indices) < 2:
        return

    for start, end in zip(peak_indices, peak_indices[1:]):
        # Segment between two peaks: exclusive of the peak rows themselves
        segment = result.iloc[start + 1:end]
        if segment.empty:
            continue

        # Candidates must be hemi-peaks AND not sit on an actual peak
        up_candidates = segment[
            segment["hemi_peak_up"] & ~(segment["peak_up"] | segment["peak_down"])
        ]
        down_candidates = segment[
            segment["hemi_peak_down"] & ~(segment["peak_up"] | segment["peak_down"])
        ]

        if not up_candidates.empty:
            max_up_idx = up_candidates["high"].idxmax()
            result.loc[max_up_idx, "max_hemi_peak_up"] = True

        if not down_candidates.empty:
            min_down_idx = down_candidates["low"].idxmin()
            result.loc[min_down_idx, "max_hemi_peak_down"] = True

def detect_peaks(df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    result = df.copy().reset_index(drop=True)

    result["peak_up"] = False
    result["peak_down"] = False
    result["drop_from_high"] = 0.0
    result["rise_from_low"] = 0.0

    trend = PriceTrend.UNKNOWN

    last_high = result.iloc[0]["high"]
    last_low = result.iloc[0]["low"]
    last_high_idx = 0
    last_low_idx = 0

    for i in range(1, len(result)):
        high = result.iloc[i]["high"]
        low = result.iloc[i]["low"]

        # Оновлюємо екстремуми
        if high > last_high:
            last_high = high
            last_high_idx = i

        if low < last_low:
            last_low = low
            last_low_idx = i

        drop_from_high = (last_high - low) / last_high
        rise_from_low = (high - last_low) / last_low

        result.loc[i, "drop_from_high"] = drop_from_high
        result.loc[i, "rise_from_low"] = rise_from_low

        # Падіння більше threshold → фіксуємо peak_up
        if drop_from_high >= threshold and trend != PriceTrend.DOWN:
            result.loc[last_high_idx, "peak_up"] = True
            trend = PriceTrend.DOWN

            last_low = low
            last_low_idx = i

        # Ріст більше threshold → фіксуємо peak_down
        elif rise_from_low >= threshold and trend != PriceTrend.UP:
            result.loc[last_low_idx, "peak_down"] = True
            trend = PriceTrend.UP

            last_high = high
            last_high_idx = i

    result = detect_hemi_peaks(result, threshold/3)
    result = detect_trends(result)
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

            peak_up=row["peak_up"],
            peak_down=row["peak_down"],
            hemi_peak_up=row["hemi_peak_up"],
            hemi_peak_down=row["hemi_peak_down"],
            max_hemi_peak_up=row["max_hemi_peak_up"],
            max_hemi_peak_down=row["max_hemi_peak_down"],
            change_from_last_peak=row["change_from_last_peak"],
            len_from_last_peak=row["len_from_last_peak"],
            last_peak_type=row["last_peak_type"],
            len_peak_to_peak=row["len_peak_to_peak"],
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
    # v = get_intervals_from_data("avg_log_return", 5, 0.01, 0.01)
    # print(v)
    # return
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
    f'btc_log_return: float = field(metadata={{"intervals": {get_intervals_from_data("btc_log_return", 5, 0.01, 0.01)}}})',
    f'eth_log_return: float = field(metadata={{"intervals": {get_intervals_from_data("eth_log_return", 5, 0.02, 0.02)}}})',
    f'ada_log_return: float = field(metadata={{"intervals": {get_intervals_from_data("ada_log_return", 5, 0.01, 0.01)}}})',
    f'doge_log_return: float = field(metadata={{"intervals": {get_intervals_from_data("doge_log_return", 5, 0.01, 0.01)}}})',
    f'xrp_log_return: float = field(metadata={{"intervals": {get_intervals_from_data("xrp_log_return", 5, 0.01, 0.01)}}})',
    f'sui_log_return: float = field(metadata={{"intervals": {get_intervals_from_data("sui_log_return", 5, 0.01, 0.01)}}})',
    f'avg_log_return: float = field(metadata={{"intervals": {get_intervals_from_data("avg_log_return", 5, 0.01, 0.01)}}})',
    '######',
    f'btc_eth_ampl_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_eth_ampl_ratio", 5, 0.01, 0.01)}}})',
    f'btc_ada_ampl_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_ada_ampl_ratio", 5, 0.01, 0.01)}}})',
    f'btc_doge_ampl_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_doge_ampl_ratio", 5, 0.01, 0.01)}}})',
    f'btc_xrp_ampl_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_xrp_ampl_ratio", 5, 0.01, 0.01)}}})',
    f'eth_ada_ampl_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_ada_ampl_ratio", 5, 0.01, 0.01)}}})',
    f'eth_doge_ampl_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_doge_ampl_ratio", 5, 0.01, 0.01)}}})',
    f'eth_xrp_ampl_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_xrp_ampl_ratio", 5, 0.01, 0.01)}}})',
    f'doge_ada_ampl_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_ada_ampl_ratio", 5, 0.01, 0.01)}}})',
    f'doge_sui_ampl_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_sui_ampl_ratio", 5, 0.01, 0.01)}}})',
    f'doge_xrp_ampl_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_xrp_ampl_ratio", 5, 0.01, 0.01)}}})',
    f'avg_ampl_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("avg_ampl_ratio", 5, 0.01, 0.01)}}})',
    '######',
    f'btc_ampl: float = field(metadata={{"intervals": {get_intervals_from_data("btc_ampl", 5, 0.01, 0.01)}}})',
    f'eth_ampl: float = field(metadata={{"intervals": {get_intervals_from_data("eth_ampl", 5, 0.02, 0.02)}}})',
    f'ada_ampl: float = field(metadata={{"intervals": {get_intervals_from_data("ada_ampl", 5, 0.01, 0.01)}}})',
    f'doge_ampl: float = field(metadata={{"intervals": {get_intervals_from_data("doge_ampl", 5, 0.01, 0.01)}}})',
    f'xrp_ampl: float = field(metadata={{"intervals": {get_intervals_from_data("xrp_ampl", 5, 0.01, 0.01)}}})',
    f'sui_ampl: float = field(metadata={{"intervals": {get_intervals_from_data("sui_ampl", 5, 0.01, 0.01)}}})',
    f'avg_ampl: float = field(metadata={{"intervals": {get_intervals_from_data("avg_ampl", 5, 0.01, 0.01)}}})',
    '######',
    f'btc_eth_drop_from_high_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_eth_drop_from_high_ratio", 5, 0.01, 0.01)}}})',
    f'btc_ada_drop_from_high_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_ada_drop_from_high_ratio", 5, 0.01, 0.01)}}})',
    f'btc_doge_drop_from_high_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_doge_drop_from_high_ratio", 5, 0.01, 0.01)}}})',
    f'btc_xrp_drop_from_high_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_xrp_drop_from_high_ratio", 5, 0.01, 0.01)}}})',
    f'eth_ada_drop_from_high_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_ada_drop_from_high_ratio", 5, 0.01, 0.01)}}})',
    f'eth_doge_drop_from_high_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_doge_drop_from_high_ratio", 5, 0.01, 0.01)}}})',
    f'eth_xrp_drop_from_high_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_xrp_drop_from_high_ratio", 5, 0.01, 0.01)}}})',
    f'doge_ada_drop_from_high_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_ada_drop_from_high_ratio", 5, 0.01, 0.01)}}})',
    f'doge_sui_drop_from_high_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_sui_drop_from_high_ratio", 5, 0.01, 0.01)}}})',
    f'doge_xrp_drop_from_high_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_xrp_drop_from_high_ratio", 5, 0.01, 0.01)}}})',
    f'avg_drop_from_high_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("avg_drop_from_high_ratio", 5, 0.01, 0.01)}}})',
    '#####',
    f'btc_drop_from_high: float = field(metadata={{"intervals": {get_intervals_from_data("btc_drop_from_high", 5, 0.01, 0.01)}}})',
    f'eth_drop_from_high: float = field(metadata={{"intervals": {get_intervals_from_data("eth_drop_from_high", 5, 0.02, 0.02)}}})',
    f'ada_drop_from_high: float = field(metadata={{"intervals": {get_intervals_from_data("ada_drop_from_high", 5, 0.01, 0.01)}}})',
    f'doge_drop_from_high: float = field(metadata={{"intervals": {get_intervals_from_data("doge_drop_from_high", 5, 0.01, 0.01)}}})',
    f'xrp_drop_from_high: float = field(metadata={{"intervals": {get_intervals_from_data("xrp_drop_from_high", 5, 0.01, 0.01)}}})',
    f'sui_drop_from_high: float = field(metadata={{"intervals": {get_intervals_from_data("sui_drop_from_high", 5, 0.01, 0.01)}}})',
    f'avg_drop_from_high: float = field(metadata={{"intervals": {get_intervals_from_data("avg_drop_from_high", 5, 0.01, 0.01)}}})',
    '#####',
    f'btc_eth_rise_from_low_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_eth_rise_from_low_ratio", 5, 0.01, 0.01)}}})',
    f'btc_ada_rise_from_low_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_ada_rise_from_low_ratio", 5, 0.01, 0.01)}}})',
    f'btc_doge_rise_from_low_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_doge_rise_from_low_ratio", 5, 0.01, 0.01)}}})',
    f'btc_xrp_rise_from_low_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("btc_xrp_rise_from_low_ratio", 5, 0.01, 0.01)}}})',
    f'eth_ada_rise_from_low_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_ada_rise_from_low_ratio", 5, 0.01, 0.01)}}})',
    f'eth_doge_rise_from_low_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_doge_rise_from_low_ratio", 5, 0.01, 0.01)}}})',
    f'eth_xrp_rise_from_low_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("eth_xrp_rise_from_low_ratio", 5, 0.01, 0.01)}}})',
    f'doge_ada_rise_from_low_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_ada_rise_from_low_ratio", 5, 0.01, 0.01)}}})',
    f'doge_sui_rise_from_low_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_sui_rise_from_low_ratio", 5, 0.01, 0.01)}}})',
    f'doge_xrp_rise_from_low_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("doge_xrp_rise_from_low_ratio", 5, 0.01, 0.01)}}})',
    f'avg_rise_from_low_ratio: float = field(metadata={{"intervals": {get_intervals_from_data("avg_rise_from_low_ratio", 5, 0.01, 0.01)}}})',
    '#####',
    f'btc_rise_from_low: float = field(metadata={{"intervals": {get_intervals_from_data("btc_rise_from_low", 5, 0.01, 0.01)}}})',
    f'eth_rise_from_low: float = field(metadata={{"intervals": {get_intervals_from_data("eth_rise_from_low", 5, 0.02, 0.02)}}})',
    f'ada_rise_from_low: float = field(metadata={{"intervals": {get_intervals_from_data("ada_rise_from_low", 5, 0.01, 0.01)}}})',
    f'doge_rise_from_low: float = field(metadata={{"intervals": {get_intervals_from_data("doge_rise_from_low", 5, 0.01, 0.01)}}})',
    f'xrp_rise_from_low: float = field(metadata={{"intervals": {get_intervals_from_data("xrp_rise_from_low", 5, 0.01, 0.01)}}})',
    f'sui_rise_from_low: float = field(metadata={{"intervals": {get_intervals_from_data("sui_rise_from_low", 5, 0.01, 0.01)}}})',
    f'avg_rise_from_low: float = field(metadata={{"intervals": {get_intervals_from_data("avg_rise_from_low", 5, 0.01, 0.01)}}})',
    ]
    print('\n'.join(point_values_lines))

