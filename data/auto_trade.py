import time

import runtime_data as rd
import trading_simulator as ts
import trading_analyzer as ta
import datetime as dt

from data.binance_data_provider import BinanceDataProvider
from data.currency_data import CurrencyData


def cleanup_runtime_data():
    rd.CURRENCY_DATAS = {}

def update_runtime_data():
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

    now_dt = dt.datetime.now()
    two_days_ago = now_dt - dt.timedelta(days=2)
    lower_bound_timestamp = int(two_days_ago.timestamp() * 1000)
    upper_bound_timestamp = int(now_dt.timestamp() * 1000)
    binance_data_provider = BinanceDataProvider()

    expected_interval_timestamp = now_dt.replace(minute=(now_dt.minute // 5) * 5, second=0, microsecond=0).timestamp()

    for symbol in symbols:
        retry_count = 0
        while True:
            if retry_count > 5:
                msg = f"Can not get data for symbol {symbol}, {expected_interval_timestamp=}."
                raise ValueError(msg)

            currency_data = CurrencyData()
            currency_data.binance_data_provider = binance_data_provider
            currency_data.symbol = symbol
            currency_data.lower_bound_timestamp = lower_bound_timestamp
            currency_data.upper_bound_timestamp = upper_bound_timestamp
            rd.CURRENCY_DATAS[symbol] = currency_data
            currency_data.update()

            last_data_timestamp = int(currency_data.ohlcv_df["timestamp"].at[-1])
            if last_data_timestamp < expected_interval_timestamp:
                time.sleep(5)
                retry_count += 1
                continue

def validate_runtime_data():
    timestamps_verify = None
    for symbol, currency_data in rd.CURRENCY_DATAS.items():
        timestamps = list(currency_data.ohlcv_df["timestamp"])
        if timestamps_verify is None:
            timestamps_verify = timestamps
        else:
            if timestamps_verify != timestamps:
                raise ValueError(f"Different timestamps in symbol {symbol}.")

def schedule():
    while True:
        try:
            update_runtime_data()
            validate_runtime_data()

            simulator = ts.TradingSimulator()
            simulator.balance_stable = 1000
            simulator.ohlcv_df = rd.CURRENCY_DATAS["ADAUSDT"].ohlcv_df
            simulator.current_index = len(simulator.ohlcv_df) - 1
            rd.VARS.simulator = simulator

            analyzer = ta.TradingAnalyzer()
            decision, reason = analyzer.analyze()

        except Exception as e:
            print(e)

            now_dt = dt.datetime.now()
            current_interval_dt = now_dt.replace(minute=(now_dt.minute // 5) * 5, second=0, microsecond=0)
            next_interval_dt = current_interval_dt + dt.timedelta(minutes=5)

            diff_to_next_interval = next_interval_dt - now_dt
            seconds_to_next_interval = diff_to_next_interval.total_seconds()
            seconds_to_wait = seconds_to_next_interval - 5

            print(f"Waiting {seconds_to_wait}s to retry...")


if __name__ == "__main__":
    schedule()