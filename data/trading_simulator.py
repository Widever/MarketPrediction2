import pandas as pd

from data.order import Order, ClosedOrder, BuyOrder, SellOrder
from data.validation import validate_ohlc_df


class TradingSimulator:

    def __init__(self):
        # Balance usdt
        self.balance: float = 0.

        # Available tokens
        self.tokens: float = 0.

        self.open_orders: list[Order] = []
        self.closed_orders: list[ClosedOrder] = []

        self.ohlcv_df: pd.DataFrame | None = None
        self.current_index: int = 0

    def execute_cancel_order(self, order: Order) -> None:
        if isinstance(order, BuyOrder):
            self.balance += order.quantity * order.price
        elif isinstance(order, SellOrder):
            self.tokens += order.quantity
        else:
            raise RuntimeError(f"Unknown order type: {type(order)}, {order=}")

    def execute_fill_order(self, order: Order) -> None:
        if isinstance(order, BuyOrder):
            self.tokens += order.quantity
        elif isinstance(order, SellOrder):
            self.balance += order.quantity * order.price
        else:
            raise RuntimeError(f"Unknown order type: {type(order)}, {order=}")

    def execute_stop_loss_order(self, order: Order) -> None:
        if not isinstance(order, SellOrder):
            raise RuntimeError("Stop loss supported only for SellOrder.")

        if order.stop_loss_price is None:
            raise RuntimeError("Can't execute stop loss when it is None.")

        self.balance += order.quantity * order.stop_loss_price

    def next(self):
        validate_ohlc_df(self.ohlcv_df)
        self.ohlcv_df: pd.DataFrame

        if self.current_index == len(self.ohlcv_df) - 1:
            print(">>> next(): Last ohlc data reached.")
            return

        next_index = self.current_index + 1
        next_timestamp = self.ohlcv_df.at[next_index, "timestamp"]
        next_low_price = self.ohlcv_df.at[next_index, "low"]
        next_high_price = self.ohlcv_df.at[next_index, "high"]

        if not isinstance(next_timestamp, int):
            raise RuntimeError("timestamp should be int.")

        remained_open_orders: list[Order] = []

        for order in self.open_orders:
            if order.expired_timestamp >= next_timestamp:
                if order.expired_timestamp != next_timestamp:
                    print(
                        f">>> warning: order {order.id} is expired but expired_timestamp "
                        f"not equal to next timestamp."
                    )

                self.execute_cancel_order(order)
                self.closed_orders.append(ClosedOrder(order, next_timestamp, "expired"))
                continue

            if isinstance(order, BuyOrder) and order.price > next_low_price:
                self.execute_fill_order(order)
                self.closed_orders.append(ClosedOrder(order, next_timestamp, "filled"))
                continue

            if isinstance(order, SellOrder):
                if order.stop_loss_price is not None and order.stop_loss_price > next_low_price:
                    self.execute_stop_loss_order(order)
                    self.closed_orders.append(ClosedOrder(order, next_timestamp, "stop_loss"))
                    continue

                if order.price < next_high_price:
                    self.execute_fill_order(order)
                    self.closed_orders.append(ClosedOrder(order, next_timestamp, "filled"))
                    continue

            remained_open_orders.append(order)

        self.open_orders = remained_open_orders
        self.current_index = next_index
