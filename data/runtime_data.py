import os
from typing import TypedDict, Any

import pandas as pd

from binance_data_provider import BinanceDataProvider
from currency_data import CurrencyData
from binance import Client
import datetime as dt


CURRENCY_DATAS = {}

def init_runtime_data(interval):
    symbols = (
        "BTCUSDT",
        "ETHUSDT",
        "ADAUSDT",
        "BNBUSDT",
        "DOGEUSDT",
        "XRPUSDT",
        "AVAXUSDT",
        "SUIUSDT",
    )

    lower_bound_timestamp = int(dt.datetime(2025, 1, 5).timestamp() * 1000)
    upper_bound_timestamp = int(dt.datetime.now().timestamp() * 1000)
    binance_data_provider = BinanceDataProvider(interval)

    cache_dir = f"currency_data_{interval}"
    os.makedirs(cache_dir, exist_ok=True)

    for symbol in symbols:
        currency_data = CurrencyData()
        currency_data.binance_data_provider = binance_data_provider
        currency_data.symbol = symbol
        currency_data.lower_bound_timestamp = lower_bound_timestamp
        currency_data.upper_bound_timestamp = upper_bound_timestamp
        CURRENCY_DATAS[symbol] = currency_data
        currency_data.update()
        file_path = os.path.join(cache_dir, f"{symbol}.csv")
        currency_data.ohlcv_df.to_csv(file_path, index=False)

def init_runtime_data_from_cache(interval):
    symbols = (
        "BTCUSDT",
        "ETHUSDT",
        "ADAUSDT",
        "BNBUSDT",
        "DOGEUSDT",
        "XRPUSDT",
        "AVAXUSDT",
        "SUIUSDT",
    )
    cache_dir = f"currency_data_{interval}"
    for symbol in symbols:
        currency_data = CurrencyData()
        currency_data.symbol = symbol
        CURRENCY_DATAS[symbol] = currency_data
        file_path = os.path.join(cache_dir, f"{symbol}.csv")
        currency_data.ohlcv_df = pd.read_csv(file_path)

print("DSCLMR")
# init_runtime_data()

class Vars:
    def __init__(self):
        self.simulator = None
        self.bot = None

VARS = Vars()