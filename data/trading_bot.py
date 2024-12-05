from data.dispersion import get_disp_1_lower
import runtime_data as rd


class TradingBot:

    def __init__(self):
        pass

    def trade(self, n: int):
        lower_disp_1 = get_disp_1_lower()
        lower_disp_1 = lower_disp_1.set_index("timestamp")
        for _ in range(n):
            if rd.VARS.simulator.balance > 0.01:
                rd.VARS.simulator_timestamp = rd.VARS.simulator.ohlcv_df["timestamp"].at[rd.VARS.simulator.current_index]
                disp_at = lower_disp_1["disp"][rd.VARS.simulator_timestamp]
                if disp_at > 0.05:
                    current_price = rd.VARS.simulator.ohlcv_df["close"].at[rd.VARS.simulator.current_index]
                    price_to_sell = current_price * 1.01
                    stop_loss = current_price * 0.98
                    rd.VARS.simulator.buy_all_instantly()
                    rd.VARS.simulator.order_sell_for_all(price_to_sell, stop_loss)

            rd.VARS.simulator.next()

