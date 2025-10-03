import os
import time
import traceback
from decimal import Decimal

import runtime_data as rd
import trading_optimizer as to
import datetime as dt

from binance_data_provider import BinanceDataProvider
from currency_data import CurrencyData
from binance import Client

from execute_order import create_test_client, get_open_orders, get_available_quote_balance, buy_market_and_wait, \
    place_sell_with_sl_tp, get_current_price


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


def schedule():
    to.DATA_DIR = "optimize_5m_interval"
    symbol = "ADAUSDT"
    asset = "ADA"
    interval = Client.KLINE_INTERVAL_5MINUTE
    interval_mins = 5
    min_gap_mins = 1
    client = create_test_client()

    wait_until_next_interval(interval_mins, min_gap_mins)

    online_ohlcv_dir = f"online_data_{interval}"
    os.makedirs(online_ohlcv_dir, exist_ok=True)

    while True:
        try:
            current_price = get_current_price(client, symbol)
            print(f"//// {current_price=}")
            _file_logger.write(f"//// {current_price=}")

            usdt_balance = get_available_quote_balance(client, "USDT")
            asset_balance = get_available_quote_balance(client, asset)
            print(f">>>> Current balance. usdt: {usdt_balance}, crypto({asset}): {asset_balance}.")
            _file_logger.write(f">>>> Current balance. usdt: {usdt_balance}, crypto({asset}): {asset_balance}.")

            open_orders = get_open_orders(client)
            if open_orders:
                raise RuntimeError(f"++++ Dont check now, open orders is not empty. {len(open_orders)=}.")

            print(f"Update data and check combs. now={dt.datetime.now().isoformat()}.")
            _file_logger.write(f"Update data and check combs. now={dt.datetime.now().isoformat()}.")

            start_time = time.time()
            update_runtime_data(interval)
            validate_runtime_data()
            end_time = time.time()

            print(f"Data updated and validated in {end_time-start_time}s.")
            _file_logger.write(f"Data updated and validated in {end_time-start_time}s.")

            # Add online ohlcv data
            header_line = "timestamp,open,high,low,close,volume"
            for symbol, currency_data in rd.CURRENCY_DATAS.items():
                file_path = os.path.join(online_ohlcv_dir, f"{symbol}.csv")

                file_exists = os.path.exists(file_path)
                is_empty = not file_exists or os.path.getsize(file_path) == 0
                currency_df = currency_data.ohlcv_df
                with open(file_path, "a", encoding="utf-8") as f:
                    if is_empty:
                        f.write(header_line + "\n")

                    data_line = f"{currency_df["timestamp"].iat[-1]},{currency_df["open"].iat[-1]},{currency_df["high"].iat[-1]},{currency_df["low"].iat[-1]},{currency_df["close"].iat[-1]},{currency_df["volume"].iat[-1]}"
                    f.write(data_line + "\n")

            last_timestamp = int(rd.CURRENCY_DATAS[symbol].ohlcv_df["timestamp"].iat[-1])
            last_timestamp_dt = dt.datetime.fromtimestamp(last_timestamp/1000)
            print(f"Last timestamp = {last_timestamp}")
            print(f"Last timestamp dt = {last_timestamp_dt.isoformat()}")

            optimizer = to.TradingOptimizer()
            decision = optimizer.check_combs_in_point()
            if decision:
                print("!!!! Decision is True. Place order...")
                _file_logger.write("!!!! Decision is True. Place order...")

                # Buy asset
                buy_crypto_response = buy_market_and_wait(client, symbol, quote_order_qty=float(usdt_balance))
                buy_order_avg_price = float(Decimal(buy_crypto_response["cummulativeQuoteQty"]) / Decimal(buy_crypto_response["executedQty"]))
                print(f"Buy successfully, avg_price: {buy_order_avg_price}.")
                _file_logger.write(f"Buy successfully, avg_price: {buy_order_avg_price}.")

                # Sell or stop loss
                asset_balance = get_available_quote_balance(client, asset)
                sell_crypto = place_sell_with_sl_tp(client, symbol, float(asset_balance), buy_order_avg_price)
                limit_order = next(x for x in sell_crypto.get("orderReports") if x.get("type") == "LIMIT_MAKER")
                limit_sell_price = limit_order.get("price")
                sl_order = next(x for x in sell_crypto.get("orderReports") if x.get("type") == "STOP_LOSS_LIMIT")
                sl_trigger_price = sl_order.get("stopPrice")
                sl_limit_price = sl_order.get("price")

                print(f"**** Sell order with stop loss placed successfully. {limit_sell_price=}, {sl_trigger_price=}, {sl_limit_price=}.")
                _file_logger.write(f"**** Sell order with stop loss placed successfully. {limit_sell_price=}, {sl_trigger_price=}, {sl_limit_price=}.")
            else:
                raise RuntimeError("Decision False.")

        except Exception as e:
            traceback.print_exception(e)
            _file_logger.write(f"---- {repr(e)}")
            wait_until_next_interval(interval_mins, min_gap_mins)


if __name__ == "__main__":
    schedule()