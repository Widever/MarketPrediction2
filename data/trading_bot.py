
import runtime_data as rd


class TradingBot:

    def __init__(self):
        self.sl_history = []

    def perform(self):
        if rd.VARS.simulator.balance < 0.01:
            print(f"Cant perform because balance < 0.01. index = {rd.VARS.simulator.current_index}.")
            return

        current_price = rd.VARS.simulator.ohlcv_df["close"].at[rd.VARS.simulator.current_index]
        price_to_sell = current_price * 1.01
        stop_loss = current_price * 0.98
        rd.VARS.simulator.buy_all_instantly()
        rd.VARS.simulator.order_sell_for_all(price_to_sell, stop_loss)

    def reset(self):
        self.sl_history = []