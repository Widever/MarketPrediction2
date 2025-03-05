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
