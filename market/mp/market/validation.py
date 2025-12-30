import pandas as pd


def validate_ohlcv_df(ohlcv_df: pd.DataFrame | None) -> None:
    if ohlcv_df is None:
        raise RuntimeError("ohlc_df is None")

    ohlcv_df: pd.DataFrame
    if len(ohlcv_df) == 0:
        raise RuntimeError("ohlc_df is empty")

    expected_columns = {"timestamp", "open", "high", "low", "close", "volume"}
    if (columns := set(ohlcv_df.columns)) != expected_columns:
        raise RuntimeError(f"Invalid ohlc_df columns: {columns}")

    if not ohlcv_df["timestamp"].apply(lambda x: isinstance(x, int)).all():
        raise RuntimeError("Not all values in 'timestamp' are int.")

    if not ohlcv_df["open"].apply(lambda x: isinstance(x, float)).all():
        raise RuntimeError("Not all values in 'open' are float.")

    if not ohlcv_df["high"].apply(lambda x: isinstance(x, float)).all():
        raise RuntimeError("Not all values in 'high' are float.")

    if not ohlcv_df["low"].apply(lambda x: isinstance(x, float)).all():
        raise RuntimeError("Not all values in 'low' are float.")

    if not ohlcv_df["close"].apply(lambda x: isinstance(x, float)).all():
        raise RuntimeError("Not all values in 'close' are float.")

    if not ohlcv_df["volume"].apply(lambda x: isinstance(x, float)).all():
        raise RuntimeError("Not all values in 'volume' are float.")

    timestamp_intervals = ohlcv_df["timestamp"][1:].reset_index(drop=True) - ohlcv_df["timestamp"][:len(ohlcv_df)-1].reset_index(drop=True)
    if not (timestamp_intervals == timestamp_intervals[0]).all():
        raise RuntimeError("Not all timestamps have the same interval.")


def validate_disp_df(disp_df: pd.DataFrame | None) -> None:
    if disp_df is None:
        raise RuntimeError("disp_df is None")

    disp_df: pd.DataFrame
    if len(disp_df) == 0:
        raise RuntimeError("disp_df is empty")

    expected_columns = {"timestamp", "disp"}
    if (columns := set(disp_df.columns)) != expected_columns:
        raise RuntimeError(f"Invalid disp_df columns: {columns}")

    if not disp_df["timestamp"].apply(lambda x: isinstance(x, int)).all():
        raise RuntimeError("Not all values in 'timestamp' are int.")

    if not disp_df["disp"].apply(lambda x: isinstance(x, float)).all():
        raise RuntimeError("Not all values in 'disp' are float.")

    if not disp_df["disp"].apply(lambda x: x >= 0).all():
        raise RuntimeError("Not all values in 'disp' are >=0.")