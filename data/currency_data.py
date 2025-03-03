
import pandas as pd

from data.binance_data_provider import BinanceDataProvider
from data.validation import validate_ohlcv_df


class CurrencyData:
    """Class to describe and update specific currency data."""

    def __init__(self):

        # Currency name, like 'BTCUSDT'
        self.symbol: str | None = None

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

        print()
        print(f">>> Update CurrencyData for {self.symbol}.")
        if self.ohlcv_df is None:
            print(f">>> Update {self.symbol} with new df.")
            # Fetch Kline (candlestick) data
            ohlcv_df = self.binance_data_provider.get_currency_ohlc_data(
                self.symbol, self.lower_bound_timestamp, self.upper_bound_timestamp
            )
            self.ohlcv_df = ohlcv_df
        else:
            if (
                    self.upper_bound_timestamp is None or
                    (last_timestamp := int(self.ohlcv_df["timestamp"].iat[-1])) < self.upper_bound_timestamp
            ):
                add_ohlcv_df = self.binance_data_provider.get_currency_ohlc_data(
                    self.symbol, last_timestamp, self.upper_bound_timestamp
                )

                if len(add_ohlcv_df) > 0:

                    old_ohlc_df = self.ohlcv_df

                    if old_ohlc_df["timestamp"].iat[-1] != add_ohlcv_df["timestamp"].iat[0]:
                        raise RuntimeError("Add df should start with last timestamp in old df.")

                    old_ohlc_df = old_ohlc_df[:-1]
                    new_ohlcv_df = pd.concat((old_ohlc_df, add_ohlcv_df)).reset_index(drop=True)
                    print(f">>> Update {self.symbol} with {len(add_ohlcv_df)} entries.")
                    self.ohlcv_df = new_ohlcv_df

        validate_ohlcv_df(self.ohlcv_df)
        print(f">>> CurrencyData for {self.symbol} has been updated, final len = {len(self.ohlcv_df)}.")

    def remove_last_row(self) -> None:
        """Remove last row from ohlc_df but left at least one."""

        if self.ohlcv_df is None:
            raise RuntimeError("Currency data ohlc_df is empty. Please run update().")

        if len(self.ohlcv_df) < 2:
            raise RuntimeError("Can not remove last row if only one row in ohlc_df.")

    def get_price_change_df(self) -> pd.DataFrame:

        if self.ohlcv_df is None or len(self.ohlcv_df) == 0:
            raise RuntimeError("Currency data ohlc_df is empty. Please run update().")
