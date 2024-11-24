import pandas as pd


def validate_ohlc_df(ohlc_df: pd.DataFrame | None) -> None:
    if ohlc_df is None:
        raise RuntimeError("ohlc_df is None")

    if len(ohlc_df) == 0:
        raise RuntimeError("ohlc_df is empty")

    expected_columns = {"timestamp", "open", "high", "low", "close"}
    if (columns := set(ohlc_df.columns)) != expected_columns:
        raise RuntimeError(f"Invalid ohlc_df columns: {columns}")
