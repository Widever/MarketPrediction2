from data.binance_data_provider import BinanceDataProvider
from data.currency_data import CurrencyData
import datetime as dt

from data.trading_simulator import TradingSimulator

CURRENCY_DATAS = {}

def init_runtime_data():
    symbols = (
        "BTCUSDT",
        "ETHUSDT",
        "ADAUSDT",
    )

    lower_bound_timestamp = int(dt.datetime(2024, 11, 23).timestamp() * 1000)
    upper_bound_timestamp = None
    binance_data_provider = BinanceDataProvider()

    for symbol in symbols:
        currency_data = CurrencyData()
        currency_data.binance_data_provider = binance_data_provider
        currency_data.symbol = symbol
        currency_data.lower_bound_timestamp = lower_bound_timestamp
        currency_data.upper_bound_timestamp = upper_bound_timestamp
        CURRENCY_DATAS[symbol] = currency_data
        currency_data.update()

init_runtime_data()

SIMULATOR = TradingSimulator()
SIMULATOR.balance_stable = 1000