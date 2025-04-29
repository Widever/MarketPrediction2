import importlib
import time
from collections import defaultdict
from html.parser import endtagfind
from turtledemo.penrose import start

import chart_builder as chb
import runtime_data as rd
import trading_bot as tb
import trading_analyzer as ta
import trading_simulator as ts
import datetime as dt


def _reload_all():
    importlib.reload(chb)
    importlib.reload(tb)
    importlib.reload(ts)
    importlib.reload(ta)


def update():
    upper_bound_timestamp = int(dt.datetime.now().timestamp() * 1000)
    for currency_data in rd.CURRENCY_DATAS.values():
        currency_data.upper_bound_timestamp = upper_bound_timestamp
        currency_data.update()


def command1():
    _reload_all()

    n = 200
    from_i = max(0, rd.VARS.simulator.current_index - (n-1))
    sample_len = min(rd.VARS.simulator.current_index, n)
    print(f"Chart last {n} for current index {rd.VARS.simulator.current_index}. Max index: {len(rd.VARS.simulator.ohlcv_df)-1}.")
    chb.run(from_i, sample_len=sample_len)


def command2():
    _reload_all()
    print("Info...")
    rd.VARS.simulator.info()

def command3():
    _reload_all()

    print("Observe...")
    tb.TradingBot().trade(200, observe=True)


def command4():
    _reload_all()

    print("Perform...")
    tb.TradingBot().perform()

def command5():
    _reload_all()

    rd.VARS.simulator.skip()
    print(f"Skipped 1. current_index: {rd.VARS.simulator.current_index}")

def command6():
    _reload_all()

    analyzer = ta.TradingAnalyzer()
    analyzer.print_analyze()


def command7():
    _reload_all()
    n = 200
    for _ in range(n):
        rd.VARS.simulator.next()

    print(f"Skipped {n}. current_index: {rd.VARS.simulator.current_index}")

def command8():
    _reload_all()

    if rd.VARS.simulator is None:
        simulator = ts.TradingSimulator()
        simulator.balance_stable = 1000
        simulator.ohlcv_df = rd.CURRENCY_DATAS["ADAUSDT"].ohlcv_df

        rd.VARS.simulator = simulator
        print("New simulator instance has been set.")

    rd.VARS.simulator.reset()
    rd.VARS.simulator.ohlcv_df = rd.CURRENCY_DATAS["ADAUSDT"].ohlcv_df
    print("Reset simulator.")

def command9():
    _reload_all()

    print("Update...")
    update()

def command10():
    _reload_all()
    print("Trade auto!!!")
    to_index = min(rd.VARS.simulator.current_index + 2000, len(rd.VARS.simulator.ohlcv_df)-1)

    while rd.VARS.simulator.current_index < to_index:
        events = tb.TradingBot().trade(200, observe=True)

        if events and any(x.trigger == "stop_loss" for x in events):
            break

        analyzer = ta.TradingAnalyzer()
        decision, reason = analyzer.print_analyze()
        if decision:
            print("Perform...")
            tb.TradingBot().perform()
        else:
            print(f"Skipped 1. current_index: {rd.VARS.simulator.current_index}")
            rd.VARS.simulator.skip()

    print("Trade auto finished.")


def command11():
    _reload_all()
    print("Trade auto!!!")
    start_time = time.time()
    to_index = min(rd.VARS.simulator.current_index + 10000, len(rd.VARS.simulator.ohlcv_df)-1)
    interval_results = []

    true_reason_dict = defaultdict(int)
    false_reason_dict = defaultdict(int)
    sl_reason_dict = defaultdict(int)

    prev_info = None
    last_cutoff = 0
    last_true_decision_reason = None
    while rd.VARS.simulator.current_index < to_index:

        if rd.VARS.simulator.current_index - last_cutoff > 2000:
            simulator_info: dict = rd.VARS.simulator.info()

            sell_orders = simulator_info.get("sell_orders")
            sl_orders = simulator_info.get("stop_loss_orders")
            balance = simulator_info.get("balance")

            if prev_info is not None:
                prev_sell_orders = prev_info.get("sell_orders")
                prev_sl_orders = prev_info.get("stop_loss_orders")
                interval_results.append((sell_orders-prev_sell_orders, sl_orders-prev_sl_orders, balance, rd.VARS.simulator.current_index))
            else:
                interval_results.append((sell_orders, sl_orders, balance, rd.VARS.simulator.current_index))

            prev_info = simulator_info
            last_cutoff = rd.VARS.simulator.current_index

        events = tb.TradingBot().trade(200, observe=True)

        if events and any(x.trigger == "stop_loss" for x in events):
            assert last_true_decision_reason is not None
            sl_reason_dict[last_true_decision_reason] += 1
            continue

        analyzer = ta.TradingAnalyzer()
        decision, reason = analyzer.print_analyze()

        if decision:
            true_reason_dict[reason] += 1
            last_true_decision_reason = reason
            print("Perform...")
            tb.TradingBot().perform()
        else:
            false_reason_dict[reason] += 1
            print(f"Skipped 1. current_index: {rd.VARS.simulator.current_index}")
            rd.VARS.simulator.skip()

    print("Intervals info:")
    for interval_sell_orders, interval_sl_orders, interval_balance, index in interval_results:
        print(f"\t- index: {index}, sell: {interval_sell_orders}, sl: {interval_sl_orders}, balance: {interval_balance}.")

    print("TOTAL:")
    rd.VARS.simulator.info()

    print("True reasons stat:")
    for reason_, count_ in sorted(true_reason_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"\t{reason_}: {count_}")

    print("False reasons stat:")
    for reason_, count_ in sorted(false_reason_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"\t{reason_}: {count_}")

    print("Stop loss reasons stat:")
    for reason_, count_ in sorted(sl_reason_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"\t{reason_}: {count_}")

    print("Benchmark finished.")

    end_time = time.time()
    print(f"Elapsed: {(end_time-start_time)/60}min.")

def run():
    _reload_all()

    # simulator = ts.TradingSimulator()
    # simulator.balance_stable = 1000
    # simulator.ohlcv_df = rd.CURRENCY_DATAS["ADAUSDT"].ohlcv_df
    #
    # rd.VARS.simulator = simulator
    # print("Simulator instance updated.")

    # rd.VARS.simulator.reset()
    # rd.VARS.simulator.ohlcv_df = rd.CURRENCY_DATAS["ADAUSDT"].ohlcv_df

    # tb.TradingBot().trade(200)
    # rd.VARS.simulator.info()

    if True or 1:
        pass
        # tb.TradingBot().perform()
        # rd.VARS.simulator.next()
        tb.TradingBot().trade(200, observe=True)
        chb.run(max(0, rd.VARS.simulator.current_index - 199), sample_len=min(rd.VARS.simulator.current_index, 200))
    else:
        rd.VARS.simulator.info()
        tb.TradingBot().perform()


    # update()
    # chb.run(6600)
    # print(rd.CURRENCY_DATAS["ADAUSDT"].ohlcv_df)
