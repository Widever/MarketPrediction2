class Order:
    def __init__(
            self,
            id_: str = "uniq",
            price: float = 0.,
            quantity: float = 0.,
            open_timestamp: int = 0,
            stop_loss_price: float | None = None,
            expired_timestamp: int | None = None,
    ):
        self.id: str = id_
        self.price: float = price
        self.quantity: float = quantity
        self.open_timestamp: int = open_timestamp
        self.stop_loss_price: float | None = stop_loss_price
        self.expired_timestamp : int | None = expired_timestamp


class BuyOrder(Order):

    def __init__(self, **kwargs):
        if kwargs.get("stop_loss_price") is not None:
            raise RuntimeError("Can't create BuyOrder with stop loss.")
        super().__init__(**kwargs)


class SellOrder(Order):
    pass

class ClosedOrder:

    def __init__(self, order: Order, close_timestamp: int, trigger: str):
        self.order: Order = order
        self.trigger: str = trigger
        self.close_timestamp: int = close_timestamp