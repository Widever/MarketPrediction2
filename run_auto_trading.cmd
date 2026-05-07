@echo off
call :launch ADAUSDT ADA 30.0
call :launch BTCUSDT BTC 20.0
call :launch ETHUSDT ETH 25.0
goto :eof

:launch
start "Trade %~1" cmd /k _auto_trade_job.cmd %1 %2 %3
goto :eof