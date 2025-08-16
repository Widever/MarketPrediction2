import importlib
import chart_builder as chb
import runtime_data as rd
import trading_bot as tb
import trading_analyzer as ta
import trading_simulator as ts
import control_panel_core as cpc
import datetime as dt


def _reload_all():
    importlib.reload(chb)
    importlib.reload(tb)
    importlib.reload(ts)
    importlib.reload(ta)
    # importlib.reload(cpc)


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
    cpc.ControlPanelCore.simulator_info()

def command3():
    _reload_all()
    cpc.ControlPanelCore.observe(200)


def command4():
    _reload_all()
    cpc.ControlPanelCore.perform()

def command5():
    _reload_all()
    cpc.ControlPanelCore.skip()

def command6():
    _reload_all()
    cpc.ControlPanelCore.analyze()

def command7():
    _reload_all()
    cpc.ControlPanelCore.skip_n()

def command8():
    _reload_all()

    if rd.VARS.simulator is None:
        simulator = ts.TradingSimulator()
        simulator.balance_stable = 1000
        simulator.ohlcv_df = rd.CURRENCY_DATAS["ADAUSDT"].ohlcv_df

        rd.VARS.simulator = simulator
        print("New simulator instance has been set.")

    cpc.ControlPanelCore.reset()
    rd.VARS.simulator.ohlcv_df = rd.CURRENCY_DATAS["ADAUSDT"].ohlcv_df
    print("Reset simulator.")

def command9():
    _reload_all()

    print("Update...")
    update()

def command10():
    _reload_all()
    cpc.ControlPanelCore.benchmark_small()


def command11():
    _reload_all()
    cpc.ControlPanelCore.benchmark_big()

def command12():
    _reload_all()
    cpc.ControlPanelCore.go_to_next_sl()

def command13():
    _reload_all()
    cpc.ControlPanelCore.go_to_prev_sl()

def command14():
    _reload_all()
    cpc.ControlPanelCore.go_to_next_perform()

def command15():
    _reload_all()
    cpc.ControlPanelCore.go_to_prev_perform()

def command16():
    _reload_all()
    cpc.ControlPanelCore.go_to_start()

def command17():
    _reload_all()
    cpc.ControlPanelCore.jump_n()

def command18():
    _reload_all()
    cpc.ControlPanelCore.get_event_stats()

def command19():
    _reload_all()
    cpc.ControlPanelCore.optimize()

