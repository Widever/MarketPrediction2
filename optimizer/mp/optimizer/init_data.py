import datetime as dt
import os
import time

import numpy as np
import pandas as pd

from mp.market.binance_data_provider import BinanceDataProvider
from mp.market.currency_data import CurrencyData
from mp.optimizer import mark

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

deviation_pairs = (
    ("BTCUSDT", "ETHUSDT"),
    ("BTCUSDT", "DOGEUSDT"),
    ("BTCUSDT", "ADAUSDT"),
    ("BTCUSDT", "AVAXUSDT"),
    ("BTCUSDT", "XRPUSDT"),
    ("ETHUSDT", "DOGEUSDT"),
    ("ETHUSDT", "AVAXUSDT"),
    ("ETHUSDT", "ADAUSDT"),
    ("ETHUSDT", "XRPUSDT"),
    ("DOGEUSDT", "ADAUSDT"),
    ("DOGEUSDT", "XRPUSDT"),
    ("DOGEUSDT", "SUIUSDT"),
)

CURRENCY_DATA_DICT = {}
DEVIATION_K_DICT = {}
AMPL_RATIO_DICT = {}
TREND_DICT = {}
PEAKS_AND_TREND_DICT = {}

def init_currency_data_dict(interval, lower_bound_timestamp=None, upper_bound_timestamp=None, save_to_file=True):

    if lower_bound_timestamp is None:
        lower_bound_timestamp = int(dt.datetime(2025, 1, 5).timestamp() * 1000)
    if upper_bound_timestamp is None:
        upper_bound_timestamp = int(dt.datetime.now().timestamp() * 1000)

    binance_data_provider = BinanceDataProvider(interval)

    cache_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(cache_dir, f"currency_data_{interval}")

    os.makedirs(cache_dir, exist_ok=True)

    for symbol in symbols:
        currency_data = CurrencyData()
        currency_data.binance_data_provider = binance_data_provider
        currency_data.symbol = symbol
        currency_data.lower_bound_timestamp = lower_bound_timestamp
        currency_data.upper_bound_timestamp = upper_bound_timestamp
        CURRENCY_DATA_DICT[symbol] = currency_data
        currency_data.update()
        if save_to_file:
            file_path = os.path.join(cache_dir, f"{symbol}.csv")
            currency_data.ohlcv_df.to_csv(file_path, index=False)


def init_currency_data_dict_from_cache(interval):
    cache_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(cache_dir, f"currency_data_{interval}")

    for symbol in symbols:
        currency_data = CurrencyData()
        currency_data.symbol = symbol
        CURRENCY_DATA_DICT[symbol] = currency_data
        file_path = os.path.join(cache_dir, f"{symbol}.csv")
        currency_data.ohlcv_df = pd.read_csv(file_path)


def init_deviation_k_dict():

    for symbol_1, symbol_2 in deviation_pairs:

        if symbol_1 not in CURRENCY_DATA_DICT:
            raise RuntimeError(f"Symbol {symbol_1} not found in CURRENCY_DATA_DICT.")
        if symbol_2 not in CURRENCY_DATA_DICT:
            raise RuntimeError(f"Symbol {symbol_2} not found in CURRENCY_DATA_DICT.")

        df_symbol_1 = CURRENCY_DATA_DICT[symbol_1].ohlcv_df
        df_symbol_2 = CURRENCY_DATA_DICT[symbol_2].ohlcv_df

        if not df_symbol_1["timestamp"].equals(df_symbol_2["timestamp"]):
            raise RuntimeError(f"Df for {symbol_1} and {symbol_2} have different timestamps.")

        log_return_symbol_1 = np.log(df_symbol_1["close"] / df_symbol_1["open"])
        log_return_symbol_2 = np.log(df_symbol_2["close"] / df_symbol_2["open"])

        deviation_k = log_return_symbol_1 / log_return_symbol_2

        deviation_df = pd.DataFrame()
        deviation_df["timestamp"] = df_symbol_1["timestamp"]
        deviation_df["deviation_k"] = deviation_k

        DEVIATION_K_DICT[(symbol_1, symbol_2)] = deviation_df

def init_ampl_ratio_dict():
    for symbol_1, symbol_2 in deviation_pairs:

        if symbol_1 not in CURRENCY_DATA_DICT:
            raise RuntimeError(f"Symbol {symbol_1} not found in CURRENCY_DATA_DICT.")
        if symbol_2 not in CURRENCY_DATA_DICT:
            raise RuntimeError(f"Symbol {symbol_2} not found in CURRENCY_DATA_DICT.")

        df_symbol_1 = CURRENCY_DATA_DICT[symbol_1].ohlcv_df
        df_symbol_2 = CURRENCY_DATA_DICT[symbol_2].ohlcv_df

        if not df_symbol_1["timestamp"].equals(df_symbol_2["timestamp"]):
            raise RuntimeError(f"Df for {symbol_1} and {symbol_2} have different timestamps.")

        ampl_symbol_1 = df_symbol_1["high"] / df_symbol_1["low"] - 1
        ampl_symbol_2 = df_symbol_2["high"] / df_symbol_2["low"] - 1

        ampl_ratio = ampl_symbol_1 / ampl_symbol_2

        ampl_ratio_df = pd.DataFrame()
        ampl_ratio_df["timestamp"] = df_symbol_1["timestamp"]
        ampl_ratio_df["ampl_ratio"] = ampl_ratio

        AMPL_RATIO_DICT[(symbol_1, symbol_2)] = ampl_ratio_df

def init_trend_dict(save_to_file=True):

    cache_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(cache_dir, f"trend_data")

    os.makedirs(cache_dir, exist_ok=True)

    for symbol in symbols:
        start = time.time()
        df_data = []
        print(f">>> init trend dict for {symbol}...")
        for timestamp in CURRENCY_DATA_DICT[symbol].ohlcv_df["timestamp"]:
            price_trend = mark.get_price_trend(symbol, timestamp)
            price_trend_dict = {
                "timestamp": timestamp,
                "trend_kind": price_trend.trend_kind.name,
                "trend_value": price_trend.trend_value,
                "trend_len": price_trend.trend_len,
            }
            df_data.append(price_trend_dict)

        df = pd.DataFrame(df_data)
        TREND_DICT[symbol] = df
        end = time.time()
        print(f">>> elapsed time: {end - start} seconds.")

        if save_to_file:
            file_path = os.path.join(cache_dir, f"{symbol}.csv")
            df.to_csv(file_path, index=False)


def init_trend_dict_from_cache():
    cache_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(cache_dir, f"trend_data")

    for symbol in symbols:
        file_path = os.path.join(cache_dir, f"{symbol}.csv")
        TREND_DICT[symbol] = pd.read_csv(file_path)

def init_peaks_and_trend_dict():
    print(">>> init peaks and trend dict...")
    start = time.time()
    for symbol in symbols:
        symbol_df = CURRENCY_DATA_DICT[symbol].ohlcv_df
        peaks_and_trend_dict = mark.detect_peaks(symbol_df)
        PEAKS_AND_TREND_DICT[symbol] = peaks_and_trend_dict
    end = time.time()
    print(f">>> elapsed time: {end - start} seconds.")
