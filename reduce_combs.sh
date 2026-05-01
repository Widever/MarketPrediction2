#!/usr/bin/env bash

# Activate virtual environment
source wsl_venv/bin/activate

# Extend PYTHONPATH (append)
export PYTHONPATH="${PYTHONPATH}:D:/AAAProjects/MarketPrediction2/market:D:/AAAProjects/MarketPrediction2/optimizer:D:/AAAProjects/MarketPrediction2/trade"

# Run module
python -m mp.optimizer.comb_v2
