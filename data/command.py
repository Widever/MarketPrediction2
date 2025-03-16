import importlib
import chart_builder as chb
import runtime_data as rd
import trading_bot as tb
import trading_simulator as ts
import datetime as dt


def update():
    upper_bound_timestamp = int(dt.datetime.now().timestamp() * 1000)
    for currency_data in rd.CURRENCY_DATAS.values():
        currency_data.upper_bound_timestamp = upper_bound_timestamp
        currency_data.update()

def command1():
    importlib.reload(chb)
    importlib.reload(tb)
    importlib.reload(ts)

    n = 200
    from_i = max(0, rd.VARS.simulator.current_index - (n-1))
    sample_len = min(rd.VARS.simulator.current_index, n)
    print(f"Chart last {n} for current index {rd.VARS.simulator.current_index}. Max index: {len(rd.VARS.simulator.ohlcv_df)-1}.")
    chb.run(from_i, sample_len=sample_len)


def command2():
    importlib.reload(chb)
    importlib.reload(tb)
    importlib.reload(ts)
    print("Info...")
    rd.VARS.simulator.info()

def command3():
    importlib.reload(chb)
    importlib.reload(tb)
    importlib.reload(ts)

    print("Observe...")
    tb.TradingBot().trade(200, observe=True)


def command4():
    importlib.reload(chb)
    importlib.reload(tb)
    importlib.reload(ts)

    print("Perform...")
    tb.TradingBot().perform()

def command5():
    importlib.reload(chb)
    importlib.reload(tb)
    importlib.reload(ts)

    rd.VARS.simulator.next()
    print(f"Skipped 1. current_index: {rd.VARS.simulator.current_index}")

def command6():
    importlib.reload(chb)
    importlib.reload(tb)
    importlib.reload(ts)

    print("Command6")

def command7():
    importlib.reload(chb)
    importlib.reload(tb)
    importlib.reload(ts)
    n = 200
    for _ in range(n):
        rd.VARS.simulator.next()

    print(f"Skipped {n}. current_index: {rd.VARS.simulator.current_index}")

def command8():
    importlib.reload(chb)
    importlib.reload(tb)
    importlib.reload(ts)

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
    importlib.reload(chb)
    importlib.reload(tb)
    importlib.reload(ts)

    print("Update...")
    update()


def run():
    importlib.reload(chb)
    importlib.reload(tb)
    importlib.reload(ts)

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
