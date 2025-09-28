import time

import runtime_data as rd
import trading_optimizer as to
import datetime as dt

from data.binance_data_provider import BinanceDataProvider
from data.currency_data import CurrencyData
from binance import Client

from data.execute_order import create_test_client, get_open_orders


class Logger:
    def __init__(self, filename: str = "auto_trade_logs.txt"):
        self.filename = filename

    def write(self, message: str):
        message = f"{dt.datetime.now().isoformat()} - {message}."
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(message + "\n")

_file_logger = Logger()

def cleanup_runtime_data():
    rd.CURRENCY_DATAS = {}

def update_runtime_data(interval):
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
    binance_data_provider = BinanceDataProvider(interval)

    for symbol in symbols:
        currency_data = CurrencyData()
        currency_data.binance_data_provider = binance_data_provider
        currency_data.symbol = symbol
        currency_data.lower_bound_timestamp = lower_bound_timestamp
        currency_data.upper_bound_timestamp = upper_bound_timestamp
        rd.CURRENCY_DATAS[symbol] = currency_data
        currency_data.update()


def validate_runtime_data():
    timestamps_verify = None
    for symbol, currency_data in rd.CURRENCY_DATAS.items():
        timestamps = list(currency_data.ohlcv_df["timestamp"])
        if timestamps_verify is None:
            timestamps_verify = timestamps
        else:
            if timestamps_verify != timestamps:
                raise ValueError(f"Different timestamps in symbol {symbol}.")


def get_next_interval_time(
        now: dt.datetime | None = None,
        interval_minutes: int = 15,
        min_gap_minutes: int = 5
) -> dt.datetime:
    if now is None:
        now = dt.datetime.now()

    base = now.replace(second=0, microsecond=0)
    minutes_from_hour = base.minute

    # Next interval
    next_minute = ((minutes_from_hour // interval_minutes) + 1) * interval_minutes

    # Change hour if seconds == 60
    if next_minute >= 60:
        next_time = base.replace(minute=0) + dt.timedelta(hours=1)
    else:
        next_time = base.replace(minute=next_minute)

    # Skip interval if < min gap
    if (next_time - now) < dt.timedelta(minutes=min_gap_minutes):
        next_time += dt.timedelta(minutes=interval_minutes)

    return next_time


def wait_until_next_interval(interval_minutes: int = 15, min_gap_minutes: int = 5):
    now = dt.datetime.now()
    next_time = get_next_interval_time(now, interval_minutes, min_gap_minutes)
    wait_seconds = (next_time - now).total_seconds()
    wait_seconds -= 30
    print(f"Wait {wait_seconds:.0f}s to {next_time}")
    _file_logger.write(f"Wait {wait_seconds:.0f}s to {next_time}")
    time.sleep(wait_seconds)

def is_ready_to_check_decision(client):
    open_orders = get_open_orders(client)
    print(">>>>>>")
    print(open_orders)
    return not open_orders


def schedule():
    to.DATA_DIR = "optimize_15m_interval"
    interval = Client.KLINE_INTERVAL_15MINUTE
    interval_mins = 15
    min_gap_mins = 1
    client = create_test_client()

    wait_until_next_interval(interval_mins, min_gap_mins)

    while True:
        try:
            if not is_ready_to_check_decision(client):
                raise RuntimeError(f"Dont check now, open orders is not empty.")

            print(f"Update data and check combs. now={dt.datetime.now().isoformat()}.")
            _file_logger.write(f"Update data and check combs. now={dt.datetime.now().isoformat()}.")

            start_time = time.time()
            update_runtime_data(interval)
            validate_runtime_data()
            end_time = time.time()

            print(f"Data updated and validated in {end_time-start_time}s.")
            _file_logger.write(f"Data updated and validated in {end_time-start_time}s.")

            last_timestamp = int(rd.CURRENCY_DATAS["ADAUSDT"].ohlcv_df["timestamp"].iat[-1])
            last_timestamp_dt = dt.datetime.fromtimestamp(last_timestamp/1000)
            print(f"Last timestamp = {last_timestamp}")
            print(f"Last timestamp dt = {last_timestamp_dt.isoformat()}")

            optimizer = to.TradingOptimizer()
            decision = optimizer.check_combs_in_point()

            if decision:
                print("Decision is True. Place order...")
            else:
                raise RuntimeError("Decision False.")

        except Exception as e:
            print(repr(e))
            _file_logger.write(repr(e))
            wait_until_next_interval(interval_mins, min_gap_mins)


if __name__ == "__main__":
    schedule()