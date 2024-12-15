import importlib

import numpy
import pandas as pd

import dispersion as dsp
import runtime_data as rd


class TradingBot:

    def __init__(self):
        self.n_cumulative = 1
        self.sell_k = 1.01
        self.stop_loss_k = 0.98
        self.disp_lower_limit = 0.003
        self.historical_stats = []


    def get_disp_changes(self, disp) -> pd.DataFrame:
        lower_limit_disp = sorted(disp["disp"])[int(len(disp)*0.7)]
        print(f"{lower_limit_disp=}")
        prev_disp_col = disp["disp"].reset_index(drop=True)[:len(disp)-1].reset_index(drop=True)
        prev_disp_col = numpy.maximum(lower_limit_disp, prev_disp_col)
        changes = (disp["disp"].reset_index(drop=True)[1:].reset_index(drop=True) / prev_disp_col - 1) * 100
        changes = pd.concat([pd.Series([None]), changes], ignore_index=True)
        if len(changes) != len(disp):
            raise RuntimeError(f"{len(changes)=} != {len(disp)=}")

        result = pd.DataFrame()
        result["timestamp"] = disp["timestamp"]
        result["changes"] = changes
        return result

    def get_cumulative_disp_change(self, disp_changes):
        cumulative_change = None
        for disp_change in disp_changes:
            if not pd.isna(disp_change) and disp_change > 0:
                if cumulative_change is None:
                    cumulative_change = disp_change
                else:
                    cumulative_change += disp_change
            else:
                if cumulative_change is None:
                    return disp_change
                else:
                    break

        return cumulative_change

    def get_last_monotone_by_lower(self, index_):
        monotone_by_lower = []
        min_change = 0.2
        last_low_v = None
        for i in range(20):
            if i > index_:
                break

            v = rd.VARS.simulator.ohlcv_df["low"].at[index_ - i]
            if last_low_v is None:
                last_low_v = v
            else:
                v_change = (last_low_v / v - 1) * 100
                last_low_v = v
                last_change = monotone_by_lower[-1] if monotone_by_lower else None
                if abs(v_change) < min_change:
                    deviant_v = min_change if last_change is None or last_change > 0 else -min_change
                    monotone_by_lower.append(deviant_v/2)
                elif last_change is None or ((v_change < 0) == (last_change < 0)):
                    monotone_by_lower.append(v_change)
                else:
                    break
        return monotone_by_lower

    def trade(self, n: int):
        importlib.reload(dsp)
        lower_disp_1 = dsp.get_disp_1_lower()
        print("disp1")
        lower_disp_1_changes = self.get_disp_changes(lower_disp_1)
        lower_disp_1_changes.set_index("timestamp", inplace=True)
        lower_disp_1.set_index("timestamp", inplace=True)

        lower_disp_2 = dsp.get_disp_2_lower()
        lower_disp_2_changes = self.get_disp_changes(lower_disp_2)
        lower_disp_2_changes.set_index("timestamp", inplace=True)

        lower_disp_3 = dsp.get_disp_3_lower()
        lower_disp_3_changes = self.get_disp_changes(lower_disp_3)
        lower_disp_3_changes.set_index("timestamp", inplace=True)


        for _ in range(n):
            for _ in range(1):
                if rd.VARS.simulator.balance > 0.01:

                    current_open = rd.VARS.simulator.ohlcv_df["open"].at[rd.VARS.simulator.current_index]
                    current_close = rd.VARS.simulator.ohlcv_df["close"].at[rd.VARS.simulator.current_index]
                    current_high = rd.VARS.simulator.ohlcv_df["high"].at[rd.VARS.simulator.current_index]

                    current_change = (current_close/current_open - 1) * 100

                    simulator_timestamp = rd.VARS.simulator.ohlcv_df["timestamp"].at[rd.VARS.simulator.current_index]
                    n_cumulative = 1
                    if rd.VARS.simulator.current_index < n_cumulative:
                        break

                    last_monotone = self.get_last_monotone_by_lower(rd.VARS.simulator.current_index)
                    last_monotone_sum = sum(last_monotone)

                    if last_monotone_sum > 0.9:
                        break

                    last_n_timestamps = [
                        rd.VARS.simulator.ohlcv_df["timestamp"].at[rd.VARS.simulator.current_index-i] for i in range(n_cumulative)
                    ]

                    last_n_disp_1_changes = [lower_disp_1_changes["changes"][t] for t in last_n_timestamps]
                    last_n_disp_2_changes = [lower_disp_2_changes["changes"][t] for t in last_n_timestamps]
                    last_n_disp_3_changes = [lower_disp_3_changes["changes"][t] for t in last_n_timestamps]

                    disp_1_change_cum = self.get_cumulative_disp_change(last_n_disp_1_changes)
                    disp_2_change_cum = self.get_cumulative_disp_change(last_n_disp_2_changes)
                    disp_3_change_cum = self.get_cumulative_disp_change(last_n_disp_3_changes)

                    if disp_1_change_cum < 100:
                        break

                    # if len(last_monotone) > 2 and last_monotone_sum < -1. and disp_1_change_cum < 200:
                    #     break

                    if disp_1_change_cum > 100 and disp_2_change_cum > 100 and disp_3_change_cum > 100:
                        break

                    current_price = rd.VARS.simulator.ohlcv_df["close"].at[rd.VARS.simulator.current_index]
                    price_to_sell = current_price * 1.01
                    stop_loss = current_price * 0.98
                    rd.VARS.simulator.buy_all_instantly()
                    rd.VARS.simulator.order_sell_for_all(price_to_sell, stop_loss)

            rd.VARS.simulator.next()

