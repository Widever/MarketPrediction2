from typing import List

import pandas as pd
import runtime_data as rd
# from data.runtime_data import CURRENCY_DATAS
from data.validation import validate_disp_df

set_1 = [
    "BTCUSDT",
    "ETHUSDT",
    "ADAUSDT",
    "BNBUSDT",
    "DOGEUSDT",
    "XRPUSDT",
    "AVAXUSDT",
    "SUIUSDT",
]

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
    for i in range(1, len(set_)):
        change_from_t = from_to_changes["target"] - from_to_changes[f"ch_{i}"]
        change_from_t = change_from_t.abs()
        if i == 1:
            disp_df["disp"] = change_from_t
        else:
            disp_df["disp"] = disp_df["disp"] + change_from_t

    return disp_df

def _get_lower_disp_for_set(set_: List[str]) -> pd.DataFrame:
    return _get_basic_disp_for_set(set_, "open", "low")

def _get_upper_disp_for_set(set_: List[str]) -> pd.DataFrame:
    return _get_basic_disp_for_set(set_, "high", "open")

def get_disp_1_lower() -> pd.DataFrame:
    disp = _get_lower_disp_for_set(set_1)

    validate_disp_df(disp)

    if len(disp) != len(rd.CURRENCY_DATAS.get(set_1[0]).ohlcv_df):
        raise RuntimeError("Disp len is invalid.")

    return disp


def get_disp_1_upper() -> pd.DataFrame:
    disp = _get_upper_disp_for_set(set_1)

    validate_disp_df(disp)

    if len(disp) != len(rd.CURRENCY_DATAS.get(set_1[0]).ohlcv_df):
        raise RuntimeError("Disp len is invalid.")

    return disp