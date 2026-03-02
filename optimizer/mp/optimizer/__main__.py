import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mp.optimizer.init_data as data
import mp.optimizer.mark as mark
import mp.optimizer.comb as comb
import mp.optimizer.benchmark as benchmark


def check_currency_data_dict():
    if not data.CURRENCY_DATA_DICT:
        raise RuntimeError("Currency data dict not initialized.")

def check_deviation_k_dict():
    if not data.DEVIATION_K_DICT:
        raise RuntimeError("Deviation k dict not initialized.")

def check_ampl_ratio_dict():
    if not data.AMPL_RATIO_DICT:
        raise RuntimeError("Ampl ratio dict not initialized.")

def timestamp_to_human_date(timestamp):
    date = pd.to_datetime(timestamp, unit="ms", utc=True)
    date = date.dt.tz_convert("Europe/Kyiv")
    return date


def plot_series_histogram(
    series: pd.Series, bins: int = 10, title: str = "Histogram", xlabel: str = "Value", ylabel: str = "Count"
):
    clean = (
        series
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if clean.empty:
        print("Немає даних для побудови гістограми.")
        return

    counts, bin_edges = np.histogram(clean, bins=bins)

    left_bounds = bin_edges[:-1]
    widths = np.diff(bin_edges)

    # округлення підписів до 3 знаків
    xtick_labels = [f"{v:.3f}" for v in left_bounds]

    plt.figure(figsize=(8, 5))
    plt.bar(
        left_bounds,
        counts,
        width=widths,
        align="edge",
        edgecolor="black",
    )

    plt.xticks(left_bounds, xtick_labels, rotation=45)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    data.init_currency_data_dict_from_cache("5m")
    # data.init_currency_data_dict("5m")
    data.init_deviation_k_dict()
    data.init_ampl_ratio_dict()

    # data.init_trend_dict()
    # data.init_trend_dict_from_cache()
    data.init_peaks_and_trend_dict()

    check_currency_data_dict()
    check_deviation_k_dict()
    check_ampl_ratio_dict()

    # series = pd.Series(mark._trend_value_ratio("DOGEUSDT", "XRPUSDT", timestamp) for timestamp in data.CURRENCY_DATA_DICT["BTCUSDT"].ohlcv_df["timestamp"])
    # pd.set_option("display.max_rows", 500)
    # print(series.sample(n=100, random_state=42))
    # print(series.min(), series.max())
    # def f_tren_value(symbol):
    #
    #     def f(x):
    #         return mark.get_price_trend(symbol, x).trend_value
    #
    #     return f
    #
    # trend_value_series_1 = data.CURRENCY_DATA_DICT["BTCUSDT"].ohlcv_df["timestamp"].apply(f_tren_value("BTCUSDT"))
    # trend_value_series_2 = data.CURRENCY_DATA_DICT["ETHUSDT"].ohlcv_df["timestamp"].apply(f_tren_value("ETHUSDT"))
    #
    # series = trend_value_series_1 / trend_value_series_2

    # series = data.AMPL_RATIO_DICT[("ETHUSDT", "DOGEUSDT")]["ampl_ratio"]
    # series = data.DEVIATION_K_DICT[("ETHUSDT", "DOGEUSDT")]["deviation_k"]
    # series = series[series >= -2]
    # series = series[series <= 3]
    # plot_series_histogram(series, bins=20)

    # mark.mark_data()
    # mark.split_marked_data()
    comb.optimal_combs(limit_comb_n=1)
    # benchmark.super_benchmark()