import importlib
import chart_builder as chb
import runtime_data as rd
import trading_bot as tb
import trading_simulator as ts


def update():
    for currency_data in rd.CURRENCY_DATAS:
        currency_data.update()

def run():
    importlib.reload(chb)
    importlib.reload(tb)
    importlib.reload(ts)
    #
    # simulator = ts.TradingSimulator()
    # simulator.balance_stable = 1000
    # simulator.ohlcv_df = rd.CURRENCY_DATAS["ADAUSDT"].ohlcv_df
    #
    # rd.VARS.simulator = simulator
    # print("Simulator instance updated.")

    tb.TradingBot().trade(5000)
    # rd.VARS.simulator.reset()
    # rd.VARS.simulator.ohlcv_df = rd.CURRENCY_DATAS["ADAUSDT"].ohlcv_df
    rd.VARS.simulator.info()
    # chb.run(0)

