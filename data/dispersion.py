from typing import List

import pandas as pd
import runtime_data as rd
# from runtime_data import CURRENCY_DATAS
from validation import validate_disp_df

_DISP_CACHE = {}

def reset_disp_cache():
    global _DISP_CACHE
    _DISP_CACHE = {}

def _get_basic_disp_for_set(set_: List[str], from_col: str, to_col: str) -> pd.DataFrame:

    if len(set_) < 2:
        raise RuntimeError("Len of set for dispersion should be >2.")

    from_to_changes = None

    for i, symbol in enumerate(set_):
        currency_data = rd.CURRENCY_DATAS.get(symbol)
        if currency_data is None:
            raise RuntimeError(f"Not found currency data for {symbol}.")

        from_to_change = currency_data.ohlcv_df[from_col] / currency_data.ohlcv_df[to_col] - 1
        if not from_to_change.apply(lambda x: x >= 0).all():
            raise RuntimeError("Not all values are >=0.")

        if from_to_changes is not None and len(from_to_changes) != len(from_to_change):
            raise RuntimeError("Result df and current df len mismatch.")

        if i == 0:
            from_to_changes = pd.DataFrame()
            from_to_changes["timestamp"] = currency_data.ohlcv_df["timestamp"]
            from_to_changes["target"] = from_to_change
        else:
            add_from_to_change = pd.DataFrame()
            add_from_to_change["timestamp"] = currency_data.ohlcv_df["timestamp"]
            add_from_to_change[f"ch_{i}"] = from_to_change

            if len(add_from_to_change) != len(from_to_changes):
                raise RuntimeError("Add df len mismatch result df len.")

            from_to_changes = pd.merge(from_to_changes, add_from_to_change, "left", on="timestamp")

    disp_df = pd.DataFrame()
    disp_df["timestamp"] = from_to_changes["timestamp"]
    for i, symbol in enumerate(set_):
        if i == 0:
            continue
        currency_data = rd.CURRENCY_DATAS.get(symbol)
        change_from_t = from_to_changes["target"] - from_to_changes[f"ch_{i}"]

        volume_k = 0.005 * currency_data.ohlcv_df["volume"].max()
        change_from_t = change_from_t.abs()#  * volume_k
        if i == 1:
            disp_df["disp"] = change_from_t
        else:
            disp_df["disp"] = disp_df["disp"] + change_from_t

    return disp_df

def _normalize_peaks(disp_df: pd.DataFrame):
    disps = sorted(disp_df["disp"])

    norm_index = int(len(disps) * 0.99)
    tear_1_index = int(len(disps) * 0.99)
    tear_2_index = int(len(disps) * 0.99)
    tear_3_index = int(len(disps) * 0.99)

    norm_value = disps[norm_index]
    tear_1_value = disps[tear_1_index]
    tear_2_value = disps[tear_2_index]
    tear_3_value = disps[tear_3_index]
    if not isinstance(norm_value, float):
        raise RuntimeError(f"Invalid norm value: {norm_value}.")

    def normalize_func(x):
        if x > tear_3_value:
            return norm_value * 1.1
        # elif x > tear_2_value:
        #     return norm_value * 1.5
        # elif x > tear_1_value:
        #     return norm_value * 1.3
        # elif x > norm_value:
        #     return norm_value * 1.2
        else:
            return x

    disp_df["disp"] = disp_df["disp"].apply(normalize_func)

def _normalize_range(disp_df: pd.DataFrame):
    max_disp = disp_df["disp"].max()
    min_disp = disp_df["disp"].min()

    def normalize_func(x):
        return (x - min_disp) / (max_disp - min_disp)

    disp_df["disp"] = disp_df["disp"].apply(normalize_func)

def _get_lower_disp_for_set(set_: List[str]) -> pd.DataFrame:
    disp_df = _get_basic_disp_for_set(set_, "open", "low")
    _normalize_peaks(disp_df)
    _normalize_range(disp_df)
    return disp_df

def _get_upper_disp_for_set(set_: List[str]) -> pd.DataFrame:
    return _get_basic_disp_for_set(set_, "high", "open")

set_1 = [
    "BTCUSDT",
    "ETHUSDT",
    "ADAUSDT",
    "BNBUSDT",
    # "DOGEUSDT",
    # "XRPUSDT",
    # "AVAXUSDT",
    # "SUIUSDT",
]

def get_disp_1_lower() -> pd.DataFrame:

    name_in_cache = "disp_1_lower"

    # if (cached_disp := _DISP_CACHE.get(name_in_cache)) is not None:
    #     return cached_disp

    disp = _get_lower_disp_for_set(set_1)

    validate_disp_df(disp)

    if len(disp) != len(rd.CURRENCY_DATAS.get(set_1[0]).ohlcv_df):
        raise RuntimeError("Disp len is invalid.")

    _DISP_CACHE[name_in_cache] = disp
    return disp


def get_disp_1_upper() -> pd.DataFrame:
    disp = _get_upper_disp_for_set(set_1)

    validate_disp_df(disp)

    if len(disp) != len(rd.CURRENCY_DATAS.get(set_1[0]).ohlcv_df):
        raise RuntimeError("Disp len is invalid.")

    return disp

set_2 = [
    # "BTCUSDT",
    "ETHUSDT",
    "ADAUSDT",
    "BNBUSDT",
    # "DOGEUSDT",
    # "XRPUSDT",
    # "AVAXUSDT",
    # "SUIUSDT",
]

def get_disp_2_lower() -> pd.DataFrame:
    disp = _get_lower_disp_for_set(set_2)

    validate_disp_df(disp)

    if len(disp) != len(rd.CURRENCY_DATAS.get(set_1[0]).ohlcv_df):
        raise RuntimeError("Disp len is invalid.")

    return disp


def get_disp_2_upper() -> pd.DataFrame:
    disp = _get_upper_disp_for_set(set_2)

    validate_disp_df(disp)

    if len(disp) != len(rd.CURRENCY_DATAS.get(set_1[0]).ohlcv_df):
        raise RuntimeError("Disp len is invalid.")

    return disp

set_3 = [
    # "BTCUSDT",
    # "ETHUSDT",
    "ADAUSDT",
    # "BNBUSDT",
    "DOGEUSDT",
    "XRPUSDT",
    "AVAXUSDT",
    # "SUIUSDT",
]

def get_disp_3_lower() -> pd.DataFrame:
    disp = _get_lower_disp_for_set(set_3)

    validate_disp_df(disp)

    if len(disp) != len(rd.CURRENCY_DATAS.get(set_1[0]).ohlcv_df):
        raise RuntimeError("Disp len is invalid.")

    return disp


def get_disp_3_upper() -> pd.DataFrame:
    disp = _get_upper_disp_for_set(set_3)

    validate_disp_df(disp)

    if len(disp) != len(rd.CURRENCY_DATAS.get(set_1[0]).ohlcv_df):
        raise RuntimeError("Disp len is invalid.")

    return disp