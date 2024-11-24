import pandas as pd
from binance.client import Client

def generate_btc_ohlcv_df() -> pd.DataFrame:
    # Define your Binance API credentials
    api_key = 'your_api_key'
    api_secret = 'your_api_secret'

    # Initialize the Binance client
    client = Client(api_key, api_secret)

    # Define the symbol, interval, and time range
    symbol = 'BTCUSDT'  # Example: BTC/USDT pair
    interval = Client.KLINE_INTERVAL_15MINUTE
    start_str = '2 day ago'  # Example: fetch data for the last 24 hours

    # Fetch Kline (candlestick) data
    klines = client.get_historical_klines(symbol, interval, start_str)

    # Convert to a Pandas DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('Europe/Kyiv')

    # Convert numeric columns to appropriate data types
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    ohlcv_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return ohlcv_df
