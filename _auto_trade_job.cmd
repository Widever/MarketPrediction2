@echo off
call .venv\Scripts\activate
set PYTHONPATH=%PYTHONPATH%;.\market;.\optimizer;.\trade
python -m mp.trade.auto_trade -s %1 -a %2 -usdt %3