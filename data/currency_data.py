import pandas as pd

from data.binance_data_provider import BinanceDataProvider


class CurrencyData:
    """Class to describe and update specific currency data."""

    def __init__(self):

        # Currency name, like 'BTCUSDT'
        self.currency: str | None = None

        # Data provider
        self.binance_data_provider: BinanceDataProvider | None = None

        # Open|High|Low|Close price data for currency
        self.ohlcv_df: pd.DataFrame | None = None

        # TODO: add default lower bound timestamp
        self.lower_bound_timestamp: int = 0

        # If None - get the newest available data
        self.upper_bound_timestamp: int | None = None


    def get_start_timestamp(self) -> int:
        """Return start (first) timestamp in self.ohlc_df."""

        if self.ohlcv_df is None or len(self.ohlcv_df) == 0:
            raise RuntimeError("Currency data ohlc_df is empty. Please run update().")

    def get_end_timestamp(self) -> int:
        """Return end (last) timestamp in self.ohlc_df."""

        if self.ohlcv_df is None or len(self.ohlcv_df) == 0:
            raise RuntimeError("Currency data ohlc_df is empty. Please run update().")

    def update(self) -> None:
        """Update currency data using binance data provider."""

        if self.binance_data_provider is None:
            raise RuntimeError("Binance data provider is required to update currency data.")

    def remove_last_row(self) -> None:
        """Remove last row from ohlc_df but left at least one."""

        if self.ohlcv_df is None:
            raise RuntimeError("Currency data ohlc_df is empty. Please run update().")

        if len(self.ohlcv_df) < 2:
            raise RuntimeError("Can not remove last row if only one row in ohlc_df.")

    def get_price_change_df(self) -> pd.DataFrame:

        if self.ohlcv_df is None or len(self.ohlcv_df) == 0:
            raise RuntimeError("Currency data ohlc_df is empty. Please run update().")
