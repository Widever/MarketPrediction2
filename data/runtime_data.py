from typing import TypedDict, Any

from data.binance_data_provider import BinanceDataProvider
from data.currency_data import CurrencyData
import datetime as dt


CURRENCY_DATAS = {}

def init_runtime_data():
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

    lower_bound_timestamp = int(dt.datetime(2024, 9, 24).timestamp() * 1000)
    upper_bound_timestamp = int(dt.datetime.now().timestamp() * 1000)
    binance_data_provider = BinanceDataProvider()

    for symbol in symbols:
        currency_data = CurrencyData()
        currency_data.binance_data_provider = binance_data_provider
        currency_data.symbol = symbol
        currency_data.lower_bound_timestamp = lower_bound_timestamp
        currency_data.upper_bound_timestamp = upper_bound_timestamp
        CURRENCY_DATAS[symbol] = currency_data
        currency_data.update()

print("DSCLMR")
# init_runtime_data()

class Vars:
    def __init__(self):
        self.simulator = None
        self.bot = None

VARS = Vars()