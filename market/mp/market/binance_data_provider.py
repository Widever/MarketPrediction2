import pandas as pd
from binance import Client


class BinanceDataProvider:

    def __init__(self, interval):
        api_key = 'your_api_key'
        api_secret = 'your_api_secret'
        self.client = Client(api_key, api_secret)
        self.interval = interval

    def get_currency_ohlc_data(self, symbol: str, from_timestamp: int, to_timestamp: int | None) -> pd.DataFrame:
        interval = Client.KLINE_INTERVAL_15MINUTE
        klines = self.client.get_historical_klines(
            symbol, self.interval, from_timestamp, to_timestamp
        )

        # Convert to a Pandas DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Convert numeric columns to appropriate data types
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(
            float)
        df['timestamp'] = df['timestamp'].astype(int)

        ohlcv_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        return ohlcv_df
