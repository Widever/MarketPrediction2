import importlib

import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data.runtime_data import CURRENCY_DATAS

import dispersion as dsp


TOOL_NAME: str = "change"
CURRENT_LINES: list = []
INFO_ANNOTATION = None
ALL_PLOTS = []
OHLCV_DATA_DFS = tuple()
CURRENT_START_INDEX = 0

def set_date_index_from_timestamp(df):
    if "timestamp" not in set(df.columns):
        raise RuntimeError("Can not set date index to timestamp column if it does not exists.")

    df["Date"] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df["Date"] = df['Date'].dt.tz_convert('Europe/Kyiv')
    df.set_index("Date", inplace=True)

def ohlcv_df_to_chart_data(df):
    chart_data_df = pd.DataFrame()

    chart_data_df["Open"] = df["open"]
    chart_data_df["High"] = df["high"]
    chart_data_df["Low"] = df["low"]
    chart_data_df["Close"] = df["close"]

    chart_data_df["Date"] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    chart_data_df["Date"] = chart_data_df['Date'].dt.tz_convert('Europe/Kyiv')
    chart_data_df.set_index("Date", inplace=True)
    return chart_data_df


show_klines_for_currencies = (
    "BTCUSDT",
    "ETHUSDT",
    "ADAUSDT",
)
def _get_start_end_indexes(start_index: int, data_len: int, sample_len: int, end_limit: int | None):
    end_index = min(end_limit, start_index + sample_len) if end_limit is not None else start_index + sample_len
    start_index = min(start_index, data_len - 2)
    return start_index, end_index

def run(start_i: int = 0, sample_len: int = 200, end_limit: int | None = None):
    if start_i < 0:
        raise RuntimeError("start_i should be >= 0")

    def _quit():
        root.quit()
        root.destroy()

    # Create the main Tkinter window
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", _quit)
    root.title("Custom Toolbar Example")


    ohlcv_charts_count = len(show_klines_for_currencies)
    additional_charts_count = 2
    fig, axs = plt.subplots(
        ohlcv_charts_count + additional_charts_count,
        1,
        figsize=(12, 8),
        sharex=True,
    )
    # Adjust figure padding
    fig.subplots_adjust(left=0.03, right=0.87, top=0.85, bottom=0.2, hspace=0.4)
    importlib.reload(dsp)

    def configure_data(start_i_=start_i):
        ohlcv_datas = {}
        global OHLCV_DATA_DFS
        for symbol in show_klines_for_currencies:
            currency_data = CURRENCY_DATAS.get(symbol)
            if currency_data is None:
                raise RuntimeError(f"No currency data for {symbol}.")

            start_index, end_index = _get_start_end_indexes(start_i_, len(currency_data.ohlcv_df), sample_len, end_limit)
            ohlcv_datas[symbol] = currency_data.ohlcv_df[start_index:end_index]

            if len(ohlcv_datas[symbol]) < sample_len:
                add_len = sample_len - len(ohlcv_datas[symbol])
                interval = ohlcv_datas[symbol]["timestamp"].iat[1] - ohlcv_datas[symbol]["timestamp"].iat[0]
                last_timestamp = ohlcv_datas[symbol]["timestamp"].iat[-1]
                last_close = ohlcv_datas[symbol]["close"].iat[-1]
                empty_ohlcv_df = pd.DataFrame(
                    {
                        "timestamp": [last_timestamp + interval*t for t in range(1, add_len+1)],
                        "open": [last_close] * add_len,
                        "high": [last_close] * add_len,
                        "low": [last_close] * add_len,
                        "close": [last_close] * add_len,
                        "volume": [last_close] * add_len,
                    }
                )
                ohlcv_datas[symbol] = pd.concat([ohlcv_datas[symbol], empty_ohlcv_df])
                ohlcv_datas[symbol] = ohlcv_datas[symbol].reset_index(drop=True)
            if len(ohlcv_datas[symbol]) != sample_len:
                raise RuntimeError(f"Incorrect ohlcv_data len after processing {len(ohlcv_datas[symbol])=}, {symbol=}.")


        OHLCV_DATA_DFS = (
            *ohlcv_datas.values(),
            *([None]*additional_charts_count)
        )

        if len(OHLCV_DATA_DFS) != len(axs):
            RuntimeError(f"ohlc_data_dfs should have the same len as axs. {len(OHLCV_DATA_DFS)=}, {len(axs)=}.")


        for ax_i_, (symbol_, ohlcv_df_) in enumerate(ohlcv_datas.items()):
            # Plot BTC candlestick chart
            y_lim = (ohlcv_df_["low"].min() * 0.98, ohlcv_df_["high"].max() * 1.02)
            mpf_plot = mpf.plot(
                ohlcv_df_to_chart_data(ohlcv_df_), type='candle', ax=axs[ax_i_], style='yahoo', ylabel="", ylim=y_lim
            )
            ALL_PLOTS.append(mpf_plot)

            axs[ax_i_].set_title(symbol_)

        # Add disp 1 lower
        disp_1_lower = dsp.get_disp_1_lower()
        start_index, end_index = _get_start_end_indexes(start_i_, len(disp_1_lower), sample_len, end_limit)
        print(f"{start_index=}")
        print(f"{end_index=}")
        disp_1_lower = disp_1_lower[start_index:end_index]
        interval = disp_1_lower["timestamp"].iat[1] - disp_1_lower["timestamp"].iat[0]
        last_timestamp = disp_1_lower["timestamp"].iat[-1]
        last_disp = disp_1_lower["disp"].iat[-1]
        if len(disp_1_lower) < sample_len:
            add_len = sample_len - len(disp_1_lower)
            add_empty_disp = pd.DataFrame(
                {
                    "timestamp": [last_timestamp + interval * t for t in range(1, add_len + 1)],
                    "disp": [last_disp]*add_len,
                }
            )
            disp_1_lower = pd.concat([disp_1_lower, add_empty_disp])

        if len(disp_1_lower) != sample_len:
            raise RuntimeError("Incorrect disp len.")
        # set_date_index_from_timestamp(disp_1)
        y_lim = (disp_1_lower["disp"].min() * 0.98, disp_1_lower["disp"].max() * 1.02)
        ax_i_ = ohlcv_charts_count - 1 + 1 # + i is ax index
        # mpf.plot(disp_1["disp"], type='line', ax=axs[ax_i_], ylabel="", ylim=y_lim)
        if (
            disp_1_lower["timestamp"].iat[0] != ohlcv_datas[show_klines_for_currencies[0]]["timestamp"].iat[0] or
            disp_1_lower["timestamp"].iat[-1] != ohlcv_datas[show_klines_for_currencies[0]]["timestamp"].iat[-1]
        ):
            raise RuntimeError("Incorrect disp after sampling.")

        axs[ax_i_].plot(disp_1_lower["disp"].reset_index(drop=True), color='blue', linestyle='-')
        axs[ax_i_].set_title("disp_1_lower")


        # Add disp 1 upper

        disp_1_upper = dsp.get_disp_1_upper()
        start_index, end_index = _get_start_end_indexes(start_i_, len(disp_1_upper), sample_len, end_limit)
        print(f"{start_index=}")
        print(f"{end_index=}")
        disp_1_upper = disp_1_upper[start_index:end_index]
        interval = disp_1_upper["timestamp"].iat[1] - disp_1_upper["timestamp"].iat[0]
        last_timestamp = disp_1_upper["timestamp"].iat[-1]
        last_disp = disp_1_upper["disp"].iat[-1]
        if len(disp_1_upper) < sample_len:
            add_len = sample_len - len(disp_1_upper)
            add_empty_disp = pd.DataFrame(
                {
                    "timestamp": [last_timestamp + interval * t for t in range(1, add_len + 1)],
                    "disp": [last_disp] * add_len,
                }
            )
            disp_1_upper = pd.concat([disp_1_upper, add_empty_disp])

        if len(disp_1_upper) != sample_len:
            raise RuntimeError("Incorrect disp len.")
        # set_date_index_from_timestamp(disp_1)
        y_lim = (disp_1_upper["disp"].min() * 0.98, disp_1_upper["disp"].max() * 1.02)
        ax_i_ = ohlcv_charts_count - 1 + 2  # + i is ax index
        # mpf.plot(disp_1["disp"], type='line', ax=axs[ax_i_], ylabel="", ylim=y_lim)
        if (
                disp_1_upper["timestamp"].iat[0] != ohlcv_datas[show_klines_for_currencies[0]]["timestamp"].iat[0] or
                disp_1_upper["timestamp"].iat[-1] != ohlcv_datas[show_klines_for_currencies[0]]["timestamp"].iat[-1]
        ):
            raise RuntimeError("Incorrect disp after sampling.")

        axs[ax_i_].plot(disp_1_upper["disp"].reset_index(drop=True), color='blue', linestyle='-')
        axs[ax_i_].set_title("disp_1_upper")



        global CURRENT_START_INDEX
        CURRENT_START_INDEX = start_i_

    configure_data()

    # Integrate the matplotlib figure with Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # Add a MultiCursor
    multi_cursor = MultiCursor(
        canvas.figure.canvas,
        axs,
        color='red',
        lw=1,
        linestyle='--',
        horizOn=True,
        vertOn=True,
        useblit=True
    )

    def proper_round(num, dec=0):
        num = str(num)[:str(num).index('.') + dec + 2]
        if num[-1] >= '5':
            return float(num[:-2 - (not dec)] + str(int(num[-2 - (not dec)]) + 1))
        return float(num[:-1])

    # On-click event handler
    def on_click(event):
        global CURRENT_LINES
        global INFO_ANNOTATION
        global OHLCV_DATA_DFS

        # Ensure the click is inside an axis
        for ax_i, ax in enumerate(axs):
            if event.inaxes == ax:
                xdata = event.xdata
                ydata = event.ydata
                if xdata and ydata:

                    if TOOL_NAME == "change":

                        if INFO_ANNOTATION is not None:
                            INFO_ANNOTATION.remove()
                            INFO_ANNOTATION = None

                        if len(CURRENT_LINES) == 2:
                            for l, price, annot in CURRENT_LINES:
                                l.remove()
                                annot.remove()
                            CURRENT_LINES = []
                        elif len(CURRENT_LINES) == 0:
                            hline = ax.axhline(y=ydata, color="green", linestyle="--", linewidth=1.5)
                            annot = ax.annotate("", xy=(0.9, ydata), xytext=(-10, 10), textcoords="offset points",
                                                bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                                                arrowprops=dict(arrowstyle='-|>'))

                            CURRENT_LINES.append((hline, ydata, annot))

                            annot_text = f"from {round(CURRENT_LINES[0][1], 3)}"
                            annot.set_text(annot_text)
                        elif len(CURRENT_LINES) == 1:
                            hline = ax.axhline(y=ydata, color="green", linestyle="--", linewidth=1.5)
                            annot = ax.annotate("", xy=(0.9, ydata), xytext=(-10, 10), textcoords="offset points",
                                                       bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                                                       arrowprops=dict(arrowstyle='-|>'))

                            CURRENT_LINES.append((hline, ydata, annot))

                            lines_y = (CURRENT_LINES[0][1], CURRENT_LINES[1][1])
                            change = (lines_y[1] / lines_y[0] - 1) * 100
                            annot_text = f"to {round(lines_y[1], 3)}, {'+' if change > 0 else ''}{round(change, 3)}%"
                            annot.set_text(annot_text)
                    elif TOOL_NAME == "info":
                        for l, price, annot in CURRENT_LINES:
                            l.remove()
                            annot.remove()
                        CURRENT_LINES = []

                        if INFO_ANNOTATION is not None:
                            INFO_ANNOTATION.remove()
                            INFO_ANNOTATION = None

                        annot = ax.annotate("", xy=(xdata, ydata), xytext=(-40, 40), textcoords="offset points",
                                            bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                                            arrowprops=dict(arrowstyle='-|>'))

                        clicked_ohlcv_data = OHLCV_DATA_DFS[ax_i]
                        if clicked_ohlcv_data is None:
                            annot.set_text("Clicked chart is not ohlvc data.")
                        else:
                            index = int(proper_round(xdata))
                            date_column = pd.to_datetime(clicked_ohlcv_data['timestamp'], unit='ms', utc=True)
                            date_column = date_column.dt.tz_convert('Europe/Kyiv')

                            if index >= len(date_column) or index < 0:
                                annot_text = "Out of range x."
                            else:
                                open_ = clicked_ohlcv_data['open'].iat[index]
                                high_ = clicked_ohlcv_data['high'].iat[index]
                                low_ = clicked_ohlcv_data['low'].iat[index]
                                close_ = clicked_ohlcv_data['close'].iat[index]

                                change_ = (close_/open_ - 1) * 100
                                amplitude_ = (high_/low_ - 1) * 100

                                annot_text = (
                                    f"Date: {date_column.iat[index]}\n"
                                    f"Open: {round(open_, 3)} | High: {round(high_, 3)}\n"
                                    f"Low: {round(low_, 3)} | Close: {round(close_, 3)}\n"
                                    f"Change: {'+' if change_>0 else ''}{round(change_, 3)}% | Ampl: {round(amplitude_, 3)}%\n"
                                    f"Volume: {round(clicked_ohlcv_data['volume'].iat[index], 3)}\n"
                                )
                            annot.set_text(annot_text)

                        INFO_ANNOTATION = annot
                    else:
                        RuntimeError(f"Unknown TOOL_NAME: {TOOL_NAME}.")

                    # Update the canvas
                    canvas.draw()
                else:
                    print("OUT")

    canvas.figure.canvas.mpl_connect("button_press_event", on_click)

    def change():
        global TOOL_NAME
        TOOL_NAME = "change"

    def info():
        global TOOL_NAME
        TOOL_NAME = "info"

    def clear():
        global CURRENT_LINES
        global INFO_ANNOTATION

        for l, price, annot in CURRENT_LINES:
            l.remove()
            annot.remove()
        CURRENT_LINES = []

        if INFO_ANNOTATION is not None:
            INFO_ANNOTATION.remove()
            INFO_ANNOTATION = None

        canvas.draw()

    def reset():
        for ax in axs:
            ax.clear()
        configure_data(start_i)
        canvas.draw()

    def go_back():
        for ax in axs:
            ax.clear()

        new_i = max(0, CURRENT_START_INDEX - 50)
        configure_data(new_i)
        canvas.draw()

    def go_forward():
        for ax in axs:
            ax.clear()

        new_i = CURRENT_START_INDEX + 50
        configure_data(new_i)
        canvas.draw()


    # Create the custom toolbar
    toolbar_frame = ttk.Frame(root)
    toolbar_frame.pack(side=tk.TOP, anchor="ne")

    clear_button = ttk.Button(toolbar_frame, text="Clear", command=clear)
    clear_button.grid(row=0, column=0)

    change_button = ttk.Button(toolbar_frame, text="Change", command=change)
    change_button.grid(row=0, column=1)

    info_button = ttk.Button(toolbar_frame, text="Info", command=info)
    info_button.grid(row=0, column=2)


    reset_button = ttk.Button(toolbar_frame, text="Reset", command=reset)
    reset_button.grid(row=1, column=0)

    go_back_button = ttk.Button(toolbar_frame, text="<--", command=go_back)
    go_back_button.grid(row=1, column=1)

    go_forward_button = ttk.Button(toolbar_frame, text="-->", command=go_forward)
    go_forward_button.grid(row=1, column=2)
    # plt.tight_layout()
    root.mainloop()

# plt.show()