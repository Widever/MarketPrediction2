import numpy
import pandas as pd

from data.order import Order, ClosedOrder, BuyOrder, SellOrder
from data.validation import validate_ohlcv_df


class TradingSimulator:

    def __init__(self):
        self.balance_stable: float = 1000.
        self.start_index: int = 0

        # Balance usdt
        self.balance: float = self.balance_stable

        # Available tokens
        self.tokens: float = 0.

        self.open_orders: list[Order] = []
        self.closed_orders: list[ClosedOrder] = []

        self.ohlcv_df: pd.DataFrame | None = None
        self.current_index: int = self.start_index
        self.last_order_index = 0

    def reset(self):
        # Balance usdt
        self.balance: float = self.balance_stable

        # Available tokens
        self.tokens: float = 0.

        self.open_orders: list[Order] = []
        self.closed_orders: list[ClosedOrder] = []

        self.ohlcv_df: pd.DataFrame | None = None
        self.current_index: int = self.start_index

    def execute_cancel_order(self, order: Order) -> None:
        print(f">>> Execute cancel order: {type(order)=}, {order.open_timestamp=}, {order.price=}.")
        if isinstance(order, BuyOrder):
            self.balance += order.quantity * order.price
        elif isinstance(order, SellOrder):
            self.tokens += order.quantity
        else:
            raise RuntimeError(f"Unknown order type: {type(order)}, {order=}")

    def execute_fill_order(self, order: Order) -> None:
        print(f">>> Execute order: {type(order)=}, {order.open_timestamp=}, {order.price=}.")
        if isinstance(order, BuyOrder):
            self.tokens += order.quantity
        elif isinstance(order, SellOrder):
            self.balance += order.quantity * order.price
        else:
            raise RuntimeError(f"Unknown order type: {type(order)}, {order=}")

    def execute_stop_loss_order(self, order: Order) -> None:
        print(f">>> Execute stop loss order: {type(order)=}, {order.open_timestamp=}, {order.price=}.")
        if not isinstance(order, SellOrder):
            raise RuntimeError("Stop loss supported only for SellOrder.")

        if order.stop_loss_price is None:
            raise RuntimeError("Can't execute stop loss when it is None.")

        self.balance += order.quantity * order.stop_loss_price

    def next(self) -> list:
        events = []

        validate_ohlcv_df(self.ohlcv_df)
        self.ohlcv_df: pd.DataFrame

        if self.current_index == len(self.ohlcv_df) - 1:
            print(">>> next(): Last ohlc data reached.")
            return events

        next_index = self.current_index + 1
        next_timestamp = self.ohlcv_df.at[next_index, "timestamp"]
        next_low_price = self.ohlcv_df.at[next_index, "low"]
        next_high_price = self.ohlcv_df.at[next_index, "high"]

        if not isinstance(next_timestamp, (int, numpy.int64)):
            raise RuntimeError(f"timestamp should be int. {type(next_timestamp)=}")

        remained_open_orders: list[Order] = []

        for order in self.open_orders:
            if order.expired_timestamp is not None and order.expired_timestamp >= next_timestamp:
                if order.expired_timestamp != next_timestamp:
                    print(
                        f">>> warning: order {order.id} is expired but expired_timestamp "
                        f"not equal to next timestamp."
                    )

                self.execute_cancel_order(order)
                event_order = ClosedOrder(order, next_timestamp, "expired")
                self.closed_orders.append(event_order)
                events.append(event_order)
                continue

            if isinstance(order, BuyOrder) and order.price > next_low_price:
                self.execute_fill_order(order)
                event_order = ClosedOrder(order, next_timestamp, "filled")
                self.closed_orders.append(event_order)
                events.append(event_order)
                continue

            if isinstance(order, SellOrder):
                if order.stop_loss_price is not None and order.stop_loss_price > next_low_price:
                    self.execute_stop_loss_order(order)
                    event_order = ClosedOrder(order, next_timestamp, "stop_loss")
                    self.closed_orders.append(event_order)
                    events.append(event_order)
                    continue

                if order.price < next_high_price:
                    self.execute_fill_order(order)
                    event_order = ClosedOrder(order, next_timestamp, "filled")
                    self.closed_orders.append(event_order)
                    events.append(event_order)
                    continue

            remained_open_orders.append(order)

        self.open_orders = remained_open_orders
        self.current_index = next_index

        return events

    def buy_all_instantly(self) -> str:
        current_price = self.ohlcv_df.at[self.current_index, "close"]
        current_timestamp = self.ohlcv_df.at[self.current_index, "timestamp"]
        quantity = self.balance / current_price
        order_id = self.last_order_index + 1
        order = BuyOrder(
            id_=f"#buy_{order_id}",
            price=current_price,
            quantity=quantity,
            open_timestamp=current_timestamp
        )
        self.balance -= quantity * current_price
        self.execute_fill_order(order)
        self.closed_orders.append(ClosedOrder(order, current_timestamp, "filled"))
        self.last_order_index = order_id
        return order.id

    def order_buy_for_all(self, price: float, expire_steps: int = None):
        current_price = self.ohlcv_df.at[self.current_index, "close"]

        if price >= current_price:
            print(">>> warning buy with price >= current price. buy instantly.")
            return self.buy_all_instantly()

        current_timestamp = self.ohlcv_df.at[self.current_index, "timestamp"]
        if expire_steps is not None:
            interval = current_timestamp - self.ohlcv_df.at[self.current_index - 1, "timestamp"]
            expired_timestamp = current_timestamp + interval * expire_steps
        else:
            expired_timestamp = None

        quantity = self.balance / price
        order_id = self.last_order_index + 1
        order = BuyOrder(
            id_=f"#buy_{order_id}",
            price=price,
            quantity=quantity,
            open_timestamp=current_timestamp,
            expired_timestamp=expired_timestamp
        )
        self.balance -= quantity * price
        self.open_orders.append(order)
        self.last_order_index = order_id
        return order.id

    def sell_all_instantly(self):
        current_price = self.ohlcv_df.at[self.current_index, "close"]
        current_timestamp = self.ohlcv_df.at[self.current_index, "timestamp"]
        quantity = self.tokens
        order_id = self.last_order_index + 1

        order = SellOrder(
            id_=f"#sell_{order_id}",
            price=current_price,
            quantity=quantity,
            open_timestamp=current_timestamp
        )
        self.tokens -= quantity
        self.execute_fill_order(order)
        self.closed_orders.append(ClosedOrder(order, current_timestamp, "filled"))
        self.last_order_index = order_id
        return order.id

    def order_sell_for_all(self, price: float, stop_loss: float = None, expire_steps: int = None):
        current_price = self.ohlcv_df.at[self.current_index, "close"]

        if price <= current_price:
            print(">>> warning sell with price <= current price. sell instantly.")
            return self.sell_all_instantly()

        current_timestamp = self.ohlcv_df.at[self.current_index, "timestamp"]
        if expire_steps is not None:
            interval = current_timestamp - self.ohlcv_df.at[self.current_index - 1, "timestamp"]
            expired_timestamp = current_timestamp + interval * expire_steps
        else:
            expired_timestamp = None

        quantity = self.tokens
        order_id = self.last_order_index + 1

        order = SellOrder(
            id_=f"#sell_{order_id}",
            price=price,
            quantity=quantity,
            open_timestamp=current_timestamp,
            expired_timestamp=expired_timestamp,
            stop_loss_price=stop_loss
        )
        self.tokens -= quantity
        self.open_orders.append(order)
        self.last_order_index = order_id
        return order.id

    def cancel_all(self):
        current_timestamp = self.ohlcv_df.at[self.current_index, "timestamp"]
        for order in self.open_orders:
            self.execute_cancel_order(order)
            self.closed_orders.append(ClosedOrder(order, current_timestamp, "canceled"))
        self.open_orders = []

    def info(self):
        buy_orders_data = []
        sell_orders_data = []
        sl_orders_data = []

        for closed_order in self.closed_orders:
            if closed_order.trigger == "filled":
                if isinstance(closed_order.order, BuyOrder):
                    buy_orders_data.append(closed_order)
                elif isinstance(closed_order.order, SellOrder):
                    sell_orders_data.append(closed_order)
            elif closed_order.trigger == "stop_loss":
                sl_orders_data.append(closed_order)

        msg = (
            f"balance: {self.balance}\n"
            f"current_index: {self.current_index}\n"
            f"df_len: {len(self.ohlcv_df)}\n"
            f"open_orders: {len(self.open_orders)}\n"
            f"closed_orders: {len(self.closed_orders)}\n"
            f"- buy_orders: {len(buy_orders_data)}\n"
            f"- sell_orders: {len(sell_orders_data)}\n"
            f"- stop_loss_orders: {len(sl_orders_data)}\n"
        )

        print(msg)