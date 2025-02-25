Start simulation:

1. Run main_controller.py - should start downloading market data, and wait for command
2. Command is in command.py.

Command tutorial at 25.02.2025:

- Set up simulator:
```
simulator = ts.TradingSimulator()
simulator.balance_stable = 1000
simulator.ohlcv_df = rd.CURRENCY_DATAS["ADAUSDT"].ohlcv_df

rd.VARS.simulator = simulator
print("Simulator instance updated.")
```

- Start bot trading:
```
rd.VARS.simulator.reset()
rd.VARS.simulator.ohlcv_df = rd.CURRENCY_DATAS["ADAUSDT"].ohlcv_df
tb.TradingBot().trade(10000)
rd.VARS.simulator.info()
```

- Show trading info:
```
rd.VARS.simulator.info()
```

- Show chart:
```
chb.run(0)
```