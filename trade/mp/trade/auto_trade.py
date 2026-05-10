import time
import traceback
from decimal import Decimal


import mp.optimizer.for_point as for_point
import mp.optimizer.init_data as data
from binance import Client

import os
import json
import datetime as dt
from dataclasses import dataclass, asdict
from mp.trade.execute_order import create_test_client, get_open_orders, get_available_quote_balance, buy_market_and_wait, \
    place_sell_all_with_sl_tp, get_current_price


data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"auto_trade_dir")



@dataclass
class AggregatedLogs:
    symbols: list[str]

    start_timestamp: int
    start_datetime: str

    last_timestamp: int
    last_datetime: str

    check_count: int
    decision_true_count: int
    decision_false_count: int
    error_count: int

    # {symbol: {mode: count}}
    decision_true_stat: dict[str, dict[str, int]]
    success_buy_stat: dict[str, dict[str, int]]

    def add_decision_true(self, symbol: str, mode: str):
        if symbol not in self.symbols:
            self.symbols.append(symbol)

        if symbol not in self.decision_true_stat:
            self.decision_true_stat[symbol] = {}
        self.decision_true_stat[symbol][mode] = (
            self.decision_true_stat[symbol].get(mode, 0) + 1
        )

        self.decision_true_count += 1

    def add_success_buy(self, symbol: str, mode: str):
        if symbol not in self.symbols:
            self.symbols.append(symbol)

        if symbol not in self.success_buy_stat:
            self.success_buy_stat[symbol] = {}
        self.success_buy_stat[symbol][mode] = (
            self.success_buy_stat[symbol].get(mode, 0) + 1
        )

    def add_error(self):
        self.error_count += 1

    def add_check(self):
        self.check_count += 1


class Logger:
    def __init__(
            self,
            stream_filename: str = "auto_trade_stream_logs.txt",
            aggregated_filename: str = "auto_trade_aggregated_logs.txt",
    ):
        self.stream_filename = stream_filename
        self.aggregated_filename = aggregated_filename

    def write_stream(self, message: str):
        os.makedirs(data_dir, exist_ok=True)

        message = f"{dt.datetime.now().isoformat()} - {message}."
        with open(os.path.join(data_dir, self.stream_filename), "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def _default_aggregated_logs(self) -> AggregatedLogs:
        return AggregatedLogs(
            symbols=[],
            start_timestamp=None,
            start_datetime=None,
            last_timestamp=None,
            last_datetime=None,
            check_count=0,
            decision_true_count=0,
            decision_false_count=0,
            error_count=0,
            decision_true_stat={},
            success_buy_stat={},
        )

    def read_aggregated_logs(self) -> AggregatedLogs:
        path = os.path.join(data_dir, self.aggregated_filename)
        if not os.path.exists(path):
            return self._default_aggregated_logs()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return AggregatedLogs(**data)

    def write_aggregated_logs(self, aggregated_logs: AggregatedLogs, last_timestamp: int | None = None):
        os.makedirs(data_dir, exist_ok=True)

        # Set start_* only if not already set
        if last_timestamp:
            if aggregated_logs.start_timestamp is None:
                aggregated_logs.start_timestamp = last_timestamp
                aggregated_logs.start_datetime = dt.datetime.fromtimestamp(
                    last_timestamp / 1000
                ).isoformat()

            # Always update last_*
            aggregated_logs.last_timestamp = last_timestamp
            aggregated_logs.last_datetime = dt.datetime.fromtimestamp(
                last_timestamp / 1000
            ).isoformat()

        path = os.path.join(data_dir, self.aggregated_filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(aggregated_logs), f, indent=2, ensure_ascii=False)

        return aggregated_logs



_file_logger = Logger()



def update_runtime_data(interval, online_ohlc_dir_, index_, range_,):

    now_dt = dt.datetime.now()
    two_days_ago = now_dt - dt.timedelta(days=2)
    lower_bound_timestamp = int(two_days_ago.timestamp() * 1000)
    upper_bound_timestamp = int(now_dt.timestamp() * 1000)

    data.init_currency_data_dict(interval, lower_bound_timestamp, upper_bound_timestamp, save_to_file=False)
    # data.init_currency_data_dict_from_online_cache(online_ohlc_dir_, index_, range_)

    data.init_log_return_dict()
    data.init_log_return_ratio_dict()
    data.init_ampl_dict()
    data.init_ampl_ratio_dict()
    data.init_peaks_and_trend_dict()
    data.init_drop_from_high_ratio_dict()
    data.init_rise_from_low_ratio_dict()


def validate_runtime_data():
    timestamps_verify = None
    for symbol, currency_data in data.CURRENCY_DATA_DICT.items():
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
    _file_logger.write_stream(f"Wait {wait_seconds:.0f}s to {next_time}")
    time.sleep(wait_seconds)


def schedule():
    symbols = {
        "ADAUSDT": "ADA",
        "ETHUSDT": "ETH",
        "DOGEUSDT": "DOGE",
        "XRPUSDT": "XRP",
        "AVAXUSDT": "AVAX",
        "SUIUSDT": "SUI"
    }

    modes = {
        "ADAUSDT": (1, 2),
        "ETHUSDT": (1, 2),
        "DOGEUSDT": (1, 2),
        "XRPUSDT": (1,),
        "AVAXUSDT": (1, 2),
        "SUIUSDT": (1,)
    }

    bets_usdt = {
        "ADAUSDT": 40,
        "ETHUSDT": 40,
        "DOGEUSDT": 40,
        "XRPUSDT": 40,
        "AVAXUSDT": 40,
        "SUIUSDT": 40
    }

    interval = Client.KLINE_INTERVAL_5MINUTE
    interval_mins = 5
    min_gap_mins = 1
    client = create_test_client()

    wait_until_next_interval(interval_mins, min_gap_mins)

    online_ohlcv_dir = os.path.join(data_dir, f"online_data_{interval}")
    os.makedirs(online_ohlcv_dir, exist_ok=True)

    index = 100
    last_index = 600
    range_ = 96
    while True:
        try:
            current_prices = [f"{symbol}: {get_current_price(client, symbol)}" for symbol, asset in symbols.items()]
            current_prices_str = " | ".join(current_prices)
            current_prices_str = f"//// current_prices. {current_prices_str}"
            print(current_prices_str)
            _file_logger.write_stream(current_prices_str)

            usdt_balance = get_available_quote_balance(client, "USDT")
            assets_balance = [f"{asset}: {get_available_quote_balance(client, asset)}" for symbol, asset in symbols.items()]
            assets_balance_str = " | ".join(assets_balance)
            assets_balance_str = f">>>> Current balance. usdt: {usdt_balance}, cryptos. {assets_balance_str}."
            print(assets_balance_str)
            _file_logger.write_stream(assets_balance_str)

            print(f"Update data and check combs. now={dt.datetime.now().isoformat()}.")
            _file_logger.write_stream(f"Update data and check combs. now={dt.datetime.now().isoformat()}.")

            start_time = time.time()

            update_runtime_data(interval, online_ohlcv_dir, index, range_)

            index += 1
            # if index >= last_index:
            #     break

            validate_runtime_data()
            end_time = time.time()

            print(f"Data updated and validated in {end_time-start_time}s.")
            _file_logger.write_stream(f"Data updated and validated in {end_time - start_time}s.")

            # Add online ohlcv data
            header_line = "timestamp,open,high,low,close,volume"
            for symbol_, currency_data in data.CURRENCY_DATA_DICT.items():
                file_path = os.path.join(online_ohlcv_dir, f"{symbol_}.csv")

                file_exists = os.path.exists(file_path)
                is_empty = not file_exists or os.path.getsize(file_path) == 0
                currency_df = currency_data.ohlcv_df
                with open(file_path, "a", encoding="utf-8") as f:
                    if is_empty:
                        f.write(header_line + "\n")

                    data_line = f"{currency_df["timestamp"].iat[-1]},{currency_df["open"].iat[-1]},{currency_df["high"].iat[-1]},{currency_df["low"].iat[-1]},{currency_df["close"].iat[-1]},{currency_df["volume"].iat[-1]}"
                    f.write(data_line + "\n")

            last_timestamp = int(data.CURRENCY_DATA_DICT[next(iter(symbols.keys()))].ohlcv_df["timestamp"].iat[-1])
            last_timestamp_dt = dt.datetime.fromtimestamp(last_timestamp/1000)
            print(f"Last timestamp = {last_timestamp}")
            print(f"Last timestamp dt = {last_timestamp_dt.isoformat()}")

            aggregated_logs = _file_logger.read_aggregated_logs()
            aggregated_logs = _file_logger.write_aggregated_logs(aggregated_logs, last_timestamp)

            for symbol, asset in symbols.items():
                bet_usdt = bets_usdt[symbol]
                open_orders = get_open_orders(client, symbol)
                if open_orders:
                    print(f"++++ Dont check now for {symbol=}, open orders is not empty. {len(open_orders)=}.")
                    continue
                for mode in modes[symbol]:
                    mode_for_log = f"Mode_{mode}"
                    decision = for_point.check_combs_for_point(symbol, mode)
                    aggregated_logs.add_check()
                    if decision:
                        decision_true_str = f"!!!! Decision is True. {symbol=}, {mode=}. Place order..."
                        print(decision_true_str)
                        _file_logger.write_stream(decision_true_str)
                        aggregated_logs.add_decision_true(symbol, mode_for_log)
                        aggregated_logs = _file_logger.write_aggregated_logs(aggregated_logs, last_timestamp)

                        # Buy asset
                        usdt_balance = get_available_quote_balance(client, "USDT")

                        if bet_usdt > float(usdt_balance) * 0.8:
                            raise RuntimeError(f"Insufficient funds in your balance to purchase {symbol}, {bet_usdt=}, {float(usdt_balance)=}.")

                        buy_crypto_response = buy_market_and_wait(client, symbol, quote_order_qty=bet_usdt)
                        buy_order_avg_price = float(Decimal(buy_crypto_response["cummulativeQuoteQty"]) / Decimal(buy_crypto_response["executedQty"]))
                        print(f"Buy successfully, avg_price: {buy_order_avg_price}, {bet_usdt=}.")
                        _file_logger.write_stream(f"Buy successfully, avg_price: {buy_order_avg_price}.")
                        aggregated_logs.add_success_buy(symbol, mode_for_log)
                        aggregated_logs = _file_logger.write_aggregated_logs(aggregated_logs, last_timestamp)

                        # Sell or stop loss
                        sell_crypto = place_sell_all_with_sl_tp(client, symbol, buy_order_avg_price)
                        limit_order = next(x for x in sell_crypto.get("orderReports") if x.get("type") == "LIMIT_MAKER")
                        limit_sell_price = limit_order.get("price")
                        sl_order = next(x for x in sell_crypto.get("orderReports") if x.get("type") == "STOP_LOSS_LIMIT")
                        sl_trigger_price = sl_order.get("stopPrice")
                        sl_limit_price = sl_order.get("price")

                        print(f"**** Sell order with stop loss placed successfully. {limit_sell_price=}, {sl_trigger_price=}, {sl_limit_price=}.")
                        _file_logger.write_stream(f"**** Sell order with stop loss placed successfully. {limit_sell_price=}, {sl_trigger_price=}, {sl_limit_price=}.")
                    else:
                        aggregated_logs = _file_logger.write_aggregated_logs(aggregated_logs, last_timestamp)
                        print(f"Decision is False. {symbol=}, {mode=}.")


        except Exception as e:
            traceback.print_exception(e)
            _file_logger.write_stream(f"---- {repr(e)}")
            aggregated_logs = _file_logger.read_aggregated_logs()
            aggregated_logs.add_error()
            _file_logger.write_aggregated_logs(aggregated_logs)
        finally:
            wait_until_next_interval(interval_mins, min_gap_mins)
            # time.sleep(0.1)

if __name__ == "__main__":
    schedule()