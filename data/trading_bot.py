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
        # lower_limit_disp = sorted(disp["disp"])[int(len(disp)*0.5)]
        # print(f"{lower_limit_disp=}")
        lower_limit_disp = 0.07
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

    def get_range_ampl(self, index_, len_):
        start_index = max(0, index_ - len_)
        high = max([rd.VARS.simulator.ohlcv_df["high"].at[i] for i in range(start_index, index_+1)])
        low = min([rd.VARS.simulator.ohlcv_df["low"].at[i] for i in range(start_index, index_+1)])

        ampl = (high / low - 1) * 100
        return ampl

    def normalize_monotone_sum(self, x):
        min_limit = -6
        max_limit = 6

        if x > max_limit:
           return 1
        elif x < min_limit:
            return 0
        else:
            return (x - min_limit) / (max_limit - min_limit)

    def get_f_disp(self, disp_1, disp_2, disp_3, index):
        last_monotone = self.get_last_monotone_by_lower(index)
        last_monotone_sum = self.normalize_monotone_sum(sum(last_monotone))

        current_disp_1 = disp_1["disp"].reset_index(drop=True).at[index]
        current_disp_2 = disp_2["disp"].reset_index(drop=True).at[index]
        current_disp_3 = disp_3["disp"].reset_index(drop=True).at[index]

        disp_sum = current_disp_1 * 1.5 + current_disp_2 * 0.8 + current_disp_3 * 0.8
        f_disp = (1 - last_monotone_sum) * 3 - disp_sum
        return f_disp

    @classmethod
    def perform(cls):
        if rd.VARS.simulator.balance < 0.01:
            print(f"Cant perform because balance < 0.01. index = {rd.VARS.simulator.current_index}.")
            return

        current_price = rd.VARS.simulator.ohlcv_df["close"].at[rd.VARS.simulator.current_index]
        price_to_sell = current_price * 1.01
        stop_loss = current_price * 0.98
        rd.VARS.simulator.buy_all_instantly()
        rd.VARS.simulator.order_sell_for_all(price_to_sell, stop_loss)

    def trade(self, n: int, observe: bool = False):
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
                    n_cumulative = 3
                    if rd.VARS.simulator.current_index < n_cumulative:
                        break

                    last_monotone = self.get_last_monotone_by_lower(rd.VARS.simulator.current_index)
                    last_range_ampl = self.get_range_ampl(rd.VARS.simulator.current_index, len(last_monotone))

                    last_monotone_sum = self.normalize_monotone_sum(sum(last_monotone))

                    current_disp_1 = lower_disp_1["disp"].reset_index(drop=True).at[rd.VARS.simulator.current_index]
                    current_disp_2 = lower_disp_2["disp"].reset_index(drop=True).at[rd.VARS.simulator.current_index]
                    current_disp_3 = lower_disp_3["disp"].reset_index(drop=True).at[rd.VARS.simulator.current_index]

                    disp_sum = current_disp_1 * 1.5 + current_disp_2 * 0.8 + current_disp_3 * 0.8

                    last_n_timestamps = [
                        rd.VARS.simulator.ohlcv_df["timestamp"].at[rd.VARS.simulator.current_index-i] for i in range(n_cumulative)
                    ]

                    last_n_disp_1_changes = [lower_disp_1_changes["changes"][t] for t in last_n_timestamps]
                    last_n_disp_2_changes = [lower_disp_2_changes["changes"][t] for t in last_n_timestamps]
                    last_n_disp_3_changes = [lower_disp_3_changes["changes"][t] for t in last_n_timestamps]

                    disp_1_change_cum = self.get_cumulative_disp_change(last_n_disp_1_changes)
                    disp_2_change_cum = self.get_cumulative_disp_change(last_n_disp_2_changes)
                    disp_3_change_cum = self.get_cumulative_disp_change(last_n_disp_3_changes)

                    # disp_gt_results = [disp_1_change_cum > 300, disp_2_change_cum > 300, disp_3_change_cum > 500]
                    disp_gt_results = [current_disp_1 > 0.21, current_disp_2 > 1110.3, current_disp_3 > 1110.4]
                    disp_gt_results = [x for x in disp_gt_results if x]

                    # f_disps_n = min(rd.VARS.simulator.current_index, 10)
                    # last_n_f_disp = [self.get_f_disp(lower_disp_1, lower_disp_2, lower_disp_3, rd.VARS.simulator.current_index-i) for i in range(f_disps_n)]
                    #
                    # if min(last_n_f_disp) < 0.55:
                    #     break
                    #
                    # if last_range_ampl < 1.2:
                    #     break

                    if len(disp_gt_results) > 0:
                        pass
                    else:
                        break

                    if observe:
                        print(f"Observe: potential perform. index = {rd.VARS.simulator.current_index}")
                        return

                    self.perform()

            events = rd.VARS.simulator.next()

            if observe and len(events) > 0:
                print(f"Events occurred, index = {rd.VARS.simulator.current_index}:")
                for e in events:
                    print("\t" + str(e))
                return

            if rd.VARS.simulator.current_index == len(rd.VARS.simulator.ohlcv_df) - 1:
                return
