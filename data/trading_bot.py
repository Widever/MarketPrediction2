import pandas as pd

import dispersion as dsp
import runtime_data as rd


class TradingBot:

    def __init__(self):
        pass

    def get_disp_changes(self, disp) -> pd.DataFrame:
        changes = (disp["disp"].reset_index(drop=True)[1:].reset_index(drop=True) / disp["disp"].reset_index(drop=True)[:len(disp)-1].reset_index(drop=True) - 1) * 100
        changes = pd.concat([pd.Series([None]), changes], ignore_index=True)
        if len(changes) != len(disp):
            raise RuntimeError(f"{len(changes)=} != {len(disp)=}")

        result = pd.DataFrame()
        result["timestamp"] = disp["timestamp"]
        result["changes"] = changes
        return result

    def trade(self, n: int):
        lower_disp_1 = dsp.get_disp_1_lower()
        lower_disp_1_changes = self.get_disp_changes(lower_disp_1)
        lower_disp_1_changes.set_index("timestamp", inplace=True)

        lower_disp_2 = dsp.get_disp_2_lower()
        lower_disp_2_changes = self.get_disp_changes(lower_disp_2)
        lower_disp_2_changes.set_index("timestamp", inplace=True)

        lower_disp_3 = dsp.get_disp_3_lower()
        lower_disp_3_changes = self.get_disp_changes(lower_disp_3)
        lower_disp_3_changes.set_index("timestamp", inplace=True)

        for _ in range(n):
            if rd.VARS.simulator.balance > 0.01:
                rd.VARS.simulator_timestamp = rd.VARS.simulator.ohlcv_df["timestamp"].at[rd.VARS.simulator.current_index]
                disp_1_change_at = lower_disp_1_changes["changes"][rd.VARS.simulator_timestamp]
                disp_2_change_at = lower_disp_2_changes["changes"][rd.VARS.simulator_timestamp]
                disp_3_change_at = lower_disp_3_changes["changes"][rd.VARS.simulator_timestamp]

                buy_condition = False
                if not pd.isna(disp_1_change_at):
                    buy_condition = disp_1_change_at > disp_2_change_at > disp_3_change_at > 100

                if buy_condition:
                    current_price = rd.VARS.simulator.ohlcv_df["close"].at[rd.VARS.simulator.current_index]
                    price_to_sell = current_price * 1.01
                    stop_loss = current_price * 0.98
                    rd.VARS.simulator.buy_all_instantly()
                    rd.VARS.simulator.order_sell_for_all(price_to_sell, stop_loss)

            rd.VARS.simulator.next()

