import random
from pathlib import Path

import numpy as np
import datetime as dt
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

def write_stat(f_name, option, optimal_combs, intervals_stat, total_count, total_sl_count, total_k):
    path = Path(f_name)

    # створює файл, якщо не існує
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    # запис у кінець файлу
    with path.open("a", encoding="utf-8") as f:
        timestamp = dt.datetime.now().isoformat()
        f.write(f"[{timestamp}] {option}\n")
        f.write(f"{optimal_combs}\n")
        f.write("\n")
        f.write(f"{intervals_stat}\n")
        f.write("\n")
        f.write(f"Total: {total_count=}, {total_sl_count=}, {total_k=}\n")
        f.write("\n")
        f.write("\n")
        f.write("===================================================\n")
        f.write("===================================================\n")
        f.write("===================================================\n")

def generate_grid_options(
    l_values=None,
    end_c_range=(11, 200),
    end_step=10,
    start_step=10,
):
    options = []

    if l_values is None:
        l_values = list(range(5, 16))  # 5..15

    for l in l_values:
        min_diff = 5 * l

        for end_c in range(end_c_range[0], end_c_range[1], end_step):
            start_min = end_c + min_diff + 1

            if start_min >= 400:
                continue

            # округляємо до найближчого step
            start_min = ((start_min + start_step - 1) // start_step) * start_step

            for start_c in range(start_min, 300, start_step):
                options.append({
                    "start_c": start_c,
                    "end_c": end_c,
                    "l": l
                })

    return options

if __name__ == '__main__':

    data.init_currency_data_dict_from_cache("5m")
    # data.init_currency_data_dict("5m")
    data.init_deviation_k_dict()
    data.init_ampl_ratio_dict()

    # data.init_trend_dict()
    # data.init_trend_dict_from_cache()
    # data.init_peaks_and_trend_dict()

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

    f_name = "options_stats/exp_18-04_1.txt"
    # options = generate_grid_options(
    #     l_values=[25],
    #     end_step=50,  # грубіший grid
    #     start_step=20
    # )
    options = [
        {
            "start_c": 40,
            "end_c": 10,
            "l": 10
        },
        {
            "start_c": 50,
            "end_c": 20,
            "l": 10
        },
        {
            "start_c": 80,
            "end_c": 50,
            "l": 10
        },
    ]
    # options = random.sample(options, 50)

    print(str(options).replace("},", "},\n"))
    for option in options:
        optimal_combs = comb.optimal_combs(limit_comb_n=1, **option)

        if optimal_combs:
            intervals_stat, total_count, total_sl_count, total_k = benchmark.super_benchmark(optimal_combs)
        else:
            intervals_stat, total_count, total_sl_count, total_k = ("empty", 0, 0, 0)

        write_stat(f_name, option, optimal_combs, intervals_stat, total_count, total_sl_count, total_k)

    # mark.adjust_point_values()