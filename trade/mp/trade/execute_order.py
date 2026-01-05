import time

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from decimal import Decimal


def buy_market_and_wait(client: Client,
                        symbol: str,
                        quantity: float = None,
                        quote_order_qty: float = None,
                        max_wait_seconds: int = 30,
                        poll_interval: float = 0.5,
                        backoff_factor: float = 1.5) -> dict:

    if (quantity is None) and (quote_order_qty is None):
        raise ValueError("Specify quantity or quote_order_qty")

    try:
        # Create buy order
        if quote_order_qty is not None:
            order = client.create_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quoteOrderQty=str(quote_order_qty)
            )
        else:
            order = client.create_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quantity=str(quantity)
            )
    except (BinanceAPIException, BinanceRequestException) as e:
        raise

    order_id = order.get("orderId") or order.get("clientOrderId")
    if not order_id:
        return order

    # Polling: wait finished status (FILLED / CANCELED / EXPIRED)
    waited = 0.0
    interval = float(poll_interval)
    terminal_statuses = {"FILLED", "CANCELED", "EXPIRED", "REJECTED"}  # REJECTED possible
    last = order

    while waited < max_wait_seconds:
        try:
            # Get status of specific order
            current = client.get_order(symbol=symbol, orderId=order_id)
        except BinanceAPIException as e:
            current = None

        if current:
            status = current.get("status")
            if status in terminal_statuses:
                return current
            last = current

        time.sleep(interval)
        waited += interval
        interval = min(interval * backoff_factor, 5.0)

    # timeout
    return last

def sell_market_and_wait(client: Client,
                         symbol: str,
                         quantity: float = None,
                         max_wait_seconds: int = 30,
                         poll_interval: float = 0.5,
                         backoff_factor: float = 1.5) -> dict:

    if quantity < 0.1:
        return {}

    if quantity is None:
        raise ValueError("Quantity is required")

    try:
        # Create sell order
        order = client.create_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_MARKET,
            quantity=str(quantity)
        )
    except (BinanceAPIException, BinanceRequestException) as e:
        raise

    order_id = order.get("orderId") or order.get("clientOrderId")
    if not order_id:
        return order

    # Polling of status
    waited = 0.0
    interval = float(poll_interval)
    terminal_statuses = {"FILLED", "CANCELED", "EXPIRED", "REJECTED"}
    last = order

    while waited < max_wait_seconds:
        try:
            current = client.get_order(symbol=symbol, orderId=order_id)
        except BinanceAPIException:
            current = None

        if current:
            status = current.get("status")
            if status in terminal_statuses:
                return current
            last = current

        time.sleep(interval)
        waited += interval
        interval = min(interval * backoff_factor, 5.0)

    return last

def _get_sell_all_qty(client: Client, symbol: str) -> float:
    # 1. Отримуємо інформацію по символу
    info = client.get_symbol_info(symbol)

    lot_size = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")
    notional = next(
        f for f in info["filters"]
        if f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL")
    )

    step_size = Decimal(lot_size["stepSize"])
    min_qty = Decimal(lot_size["minQty"])
    min_notional = Decimal(notional["minNotional"])

    base_asset = info["baseAsset"]

    # 2. Баланс базового asset
    balances = client.get_account()["balances"]
    balance = next(b for b in balances if b["asset"] == base_asset)

    free_qty = Decimal(balance["free"])

    if free_qty <= 0:
        raise RuntimeError(f"No free balance for {base_asset}")

    # 3. Залишаємо 1% запасу + округляємо по stepSize
    qty = (free_qty * Decimal("0.99") // step_size) * step_size

    if qty < min_qty:
        raise RuntimeError(
            f"Quantity {qty} < minQty {min_qty} for {symbol}"
        )

    # 4. Перевірка NOTIONAL (для MARKET потрібна поточна ціна)
    price = Decimal(client.get_symbol_ticker(symbol=symbol)["price"])

    if qty * price < min_notional:
        raise RuntimeError(
            f"Notional {qty * price} < minNotional {min_notional} for {symbol}"
        )

    return qty

def order_sell_all_available(client: Client, symbol: str):
    """
    Продає всі доступні монети (free balance) для symbol через MARKET SELL.
    Коректно обробляє LOT_SIZE та NOTIONAL.
    """
    qty = _get_sell_all_qty(client, symbol)

    # 5. MARKET SELL
    return client.create_order(
        symbol=symbol,
        side=Client.SIDE_SELL,
        type=Client.ORDER_TYPE_MARKET,
        quantity=str(qty),
    )

def sell_all_market_and_wait(
    client: Client,
    symbol: str,
    max_wait_seconds: int = 30,
    poll_interval: float = 0.5,
    backoff_factor: float = 1.5
) -> dict:

    try:
        # Create sell order
        order = order_sell_all_available(client, symbol)
    except (BinanceAPIException, BinanceRequestException) as e:
        raise

    order_id = order.get("orderId") or order.get("clientOrderId")
    if not order_id:
        return order

    # Polling of status
    waited = 0.0
    interval = float(poll_interval)
    terminal_statuses = {"FILLED", "CANCELED", "EXPIRED", "REJECTED"}
    last = order

    while waited < max_wait_seconds:
        try:
            current = client.get_order(symbol=symbol, orderId=order_id)
        except BinanceAPIException:
            current = None

        if current:
            status = current.get("status")
            if status in terminal_statuses:
                return current
            last = current

        time.sleep(interval)
        waited += interval
        interval = min(interval * backoff_factor, 5.0)

    return last

def place_sell_all_with_sl_tp(client, symbol: str, avg_buy_price) -> dict:

    qty = _get_sell_all_qty(client, symbol)

    current_price = get_current_price(client, symbol)
    tp_limit_price = avg_buy_price * 1.01

    if tp_limit_price <= current_price:
        tp_limit_price = current_price * 1.01

    tp_limit_price = round(tp_limit_price, 4)

    sl_trigger_price = avg_buy_price * 0.98

    if sl_trigger_price >= current_price:
        sl_trigger_price = current_price * 0.98

    sl_trigger_price = round(sl_trigger_price, 4)

    sl_limit_price = sl_trigger_price * 0.998
    sl_limit_price = round(sl_limit_price, 4)

    try:
        oco = client.create_oco_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            quantity=qty,
            abovePrice=tp_limit_price,
            belowStopPrice=sl_trigger_price,
            belowPrice=sl_limit_price,
            aboveType="LIMIT_MAKER",
            belowType="STOP_LOSS_LIMIT",
            belowTimeInForce=Client.TIME_IN_FORCE_GTC
        )

        return oco
    except Exception as e:
        raise e


def get_open_orders(client: Client, symbol: str | None = None) -> list[dict]:
    try:
        if symbol:
            return client.get_open_orders(symbol=symbol)
        else:
            return client.get_open_orders()
    except Exception as e:
        raise


def get_available_quote_balance(client: Client, asset: str = "USDT") -> Decimal:
    try:
        balance_info = client.get_asset_balance(asset=asset)
        if not balance_info:
            raise RuntimeError(f"Asset {asset} not found in account.")
        return Decimal(balance_info["free"])
    except Exception as e:
        raise


def get_current_price(client: Client, symbol: str) -> float:
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker["price"])

def cancel_all_orders(client: Client, symbol: str) -> list[dict]:
    if not get_open_orders(client, symbol):
        return []

    try:
        canceled_orders = client.cancel_all_open_orders(symbol=symbol)
        return canceled_orders
    except Exception as e:
        raise e

def create_test_client() -> Client:
    # TEST
    # api_key = "uzbh3mlEit1dySPyv7VV1jnqLJT7jBluK9cwaqOLDn0ARjbP3OFxof2gwVjNe6hS"
    # api_secret = "EkTioiLRUF04qT6y3Hfs0U0kIzYP7ZgLvs8hV7mUhlQm8HDFek3rdo1dQGaTzVok"

    # REAL
    api_key = "***"
    api_secret = "***"

    client = Client(api_key, api_secret, testnet=False)
    # testnet URL
    # client.API_URL = 'https://testnet.binance.vision/api'
    return client

def decision(client):
    symbol = "ADAUSDT"
    asset = "ADA"
    print("!!!! Decision is True. Place order...")

    # Buy asset
    usdt_balance = get_available_quote_balance(client, "USDT")
    usdt_balance = 15.0
    buy_crypto_response = buy_market_and_wait(client, symbol, quote_order_qty=float(usdt_balance))
    buy_order_avg_price = float(
        Decimal(buy_crypto_response["cummulativeQuoteQty"]) / Decimal(buy_crypto_response["executedQty"]))
    print(f"Buy successfully, avg_price: {buy_order_avg_price}.")

    # Sell or stop loss
    sell_crypto = place_sell_all_with_sl_tp(client, symbol, buy_order_avg_price)
    limit_order = next(x for x in sell_crypto.get("orderReports") if x.get("type") == "LIMIT_MAKER")
    limit_sell_price = limit_order.get("price")
    sl_order = next(x for x in sell_crypto.get("orderReports") if x.get("type") == "STOP_LOSS_LIMIT")
    sl_trigger_price = sl_order.get("stopPrice")
    sl_limit_price = sl_order.get("price")

if __name__ == "__main__":

    client = create_test_client()
    # decision(client)

    # cancel_all_orders(client, "ADAUSDT")
    sell_crypto = sell_all_market_and_wait(client, "ADAUSDT")
    # sell_crypto = sell_market_and_wait(client, "USDCUSDT", quantity=100)

    usdt_balance = get_available_quote_balance(client, "USDT")
    ada_balance = get_available_quote_balance(client, "ADA")
    open_orders = get_open_orders(client)
    v = client.get_account()

    print(">>>>")
    print(usdt_balance)
    print(ada_balance)
    print(open_orders)
    print(v)

    # buy_crypto = buy_market_and_wait(client, "ADAUSDT", quote_order_qty=float(usdt_balance))
    # buy_order_avg_price = float(Decimal(buy_crypto["cummulativeQuoteQty"]) / Decimal(buy_crypto["executedQty"]))
    #
    # # # sell_crypto = sell_market_and_wait(client, "ADAUSDT", quantity=float(ada_balance))
    # # # buy_crypto = {'symbol': 'ADAUSDT', 'orderId': 1160743, 'orderListId': -1, 'clientOrderId': 'x-HNA2TXFJ86d24987c389b5a4fe6db6', 'price': '0.00000000', 'origQty': '13455.30000000', 'executedQty': '13455.30000000', 'cummulativeQuoteQty': '10487.06082000', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'MARKET', 'side': 'BUY', 'stopPrice': '0.00000000', 'icebergQty': '0.00000000', 'time': 1759072100223, 'updateTime': 1759072100223, 'isWorking': True, 'workingTime': 1759072100223, 'origQuoteOrderQty': '10487.11692000', 'selfTradePreventionMode': 'EXPIRE_MAKER'}
    # #
    # usdt_balance = get_available_quote_balance(client, "USDT")
    # ada_balance = get_available_quote_balance(client, "ADA")
    # open_orders = get_open_orders(client)
    # print(">>>> after buy")
    # print(usdt_balance)
    # print(ada_balance)
    # print(open_orders)
    #
    # sell_crypto = place_sell_with_sl_tp(client, "ADAUSDT", float(ada_balance), buy_order_avg_price)
    # print(sell_crypto)
    #
    # usdt_balance = get_available_quote_balance(client, "USDT")
    # ada_balance = get_available_quote_balance(client, "ADA")
    # open_orders = get_open_orders(client)
    # print(">>>> after sell")
    # print(usdt_balance)
    # print(ada_balance)
    # print(open_orders)