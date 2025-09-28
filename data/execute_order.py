import time
import math
from turtledemo.penrose import start

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from decimal import Decimal, ROUND_DOWN

# Параметри: API_KEY, API_SECRET
# РЕКОМЕНДАЦІЯ: для розробки використовуйте testnet (https://testnet.binance.vision/) або sandbox

def buy_market_and_wait(client: Client,
                        symbol: str,
                        quantity: float = None,
                        quote_order_qty: float = None,
                        max_wait_seconds: int = 30,
                        poll_interval: float = 0.5,
                        backoff_factor: float = 1.5) -> dict:
    """
    Виконує spot market buy і чекає поки ордер завершиться.

    Аргументи:
      - client: binance.client.Client (заповнений API ключем)
      - symbol: торговий символ, напр. "BTCUSDT"
      - quantity: кількість базової валюти (наприклад BTC) для купівлі
      - quote_order_qty: кількість quote валюти, яку витратити (наприклад USDT). Якщо вказаний, quantity ігнорується.
      - max_wait_seconds: максимальний час очікування виконання
      - poll_interval: початковий інтервал між запитами (сек)
      - backoff_factor: множник для експоненційного backoff

    Повертає: dict з фінальними даними ордеру (як повертає Binance API).
    Викидає виключення при помилці.
    """

    if (quantity is None) and (quote_order_qty is None):
        raise ValueError("Потрібно вказати quantity або quote_order_qty")

    try:
        # Поставити MARKET ордер
        if quote_order_qty is not None:
            # Купити на суму quote_order_qty (спрацює як "spend X USDT")
            order = client.create_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quoteOrderQty=str(quote_order_qty)  # рядком інколи надійніше
            )
        else:
            # Купити quantity одиниць базової валюти
            # Зауваж: треба привести quantity до дозволеного precision (stepSize) для символу
            order = client.create_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quantity=str(quantity)
            )
    except (BinanceAPIException, BinanceRequestException) as e:
        # помилка від API (наприклад insufficient balance, bad symbol тощо)
        raise

    # order містить початкову відповідь; якщо market — може повернути частково заповнені дані
    order_id = order.get("orderId") or order.get("clientOrderId")
    if not order_id:
        # Якщо від API надійшло повне виконання у відповіді — повернути її
        return order

    # Polling: чекати поки статус перейде у кінцевий (FILLED / CANCELED / EXPIRED)
    waited = 0.0
    interval = float(poll_interval)
    terminal_statuses = {"FILLED", "CANCELED", "EXPIRED", "REJECTED"}  # REJECTED можливий
    last = order

    while waited < max_wait_seconds:
        try:
            # Отримати статус конкретного ордеру
            current = client.get_order(symbol=symbol, orderId=order_id)
        except BinanceAPIException as e:
            # іноді API повертає помилку, пробуємо ще раз через інтервал
            current = None

        if current:
            status = current.get("status")
            # Якщо ордер повністю виконано або інша кінцева подія — повертаємо
            if status in terminal_statuses:
                return current
            # Для market ордерів іноді status == 'PARTIALLY_FILLED' або поле executedQty змінюється
            last = current

        time.sleep(interval)
        waited += interval
        interval = min(interval * backoff_factor, 5.0)  # обмежимо backoff до 5 сек

    # таймаут — повертаємо останній відомий стан, або викидаємо помилку
    return last

def sell_market_and_wait(client: Client,
                         symbol: str,
                         quantity: float = None,
                         max_wait_seconds: int = 30,
                         poll_interval: float = 0.5,
                         backoff_factor: float = 1.5) -> dict:
    """
    Виконує spot market sell і чекає поки ордер завершиться.

    Args:
        client: binance.client.Client
        symbol: торговий символ, напр. "BTCUSDT"
        quantity: кількість базової валюти для продажу (наприклад BTC)
        max_wait_seconds: максимальний час очікування виконання
        poll_interval: початковий інтервал між запитами (сек)
        backoff_factor: множник для експоненційного backoff

    Returns:
        dict: фінальний стан ордеру
    """

    if quantity < 0.1:
        return {}

    if quantity is None:
        raise ValueError("Потрібно вказати quantity для продажу")

    try:
        # Поставити MARKET ордер на продаж
        order = client.create_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_MARKET,
            quantity=str(quantity)
        )
    except (BinanceAPIException, BinanceRequestException) as e:
        raise RuntimeError(f"Помилка при створенні sell ордеру: {e}")

    order_id = order.get("orderId") or order.get("clientOrderId")
    if not order_id:
        # Якщо market ордер повністю виконався і info вже є — повертаємо
        return order

    # Polling статусу ордеру
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

def place_sell_with_sl_tp(client, symbol: str, quantity, avg_buy_price) -> dict:
    """
    Створює OCO ордер (take profit + stop loss) після успішного buy.
    """

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
            quantity=quantity,
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
    """
    Повертає список відкритих ордерів користувача.

    Args:
        client: binance.client.Client
        symbol: (опц.) конкретний торговий символ, напр. "BTCUSDT".
                Якщо None — поверне всі відкриті ордери для акаунта.

    Returns:
        list[dict]: список відкритих ордерів (як у Binance API)
    """
    try:
        if symbol:
            return client.get_open_orders(symbol=symbol)
        else:
            return client.get_open_orders()
    except Exception as e:
        raise RuntimeError(f"Не вдалося отримати відкриті ордери: {e}")


def get_available_quote_balance(client: Client, asset: str = "USDT") -> Decimal:
    """
    Повертає кількість доступного (free) quote asset на акаунті.

    Args:
        client: binance.client.Client
        asset: тикер активу (наприклад "USDT", "BUSD", "USDC")

    Returns:
        Decimal: доступна кількість asset
    """
    try:
        balance_info = client.get_asset_balance(asset=asset)
        if not balance_info:
            raise RuntimeError(f"Актив {asset} не знайдено у балансі акаунта")
        return Decimal(balance_info["free"])
    except Exception as e:
        raise RuntimeError(f"Не вдалося отримати баланс {asset}: {e}")


def get_current_price(client: Client, symbol: str) -> float:
    """
    Повертає останню ринкову ціну для symbol (наприклад BTCUSDT)
    """
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker["price"])

def cancel_all_orders(client: Client, symbol: str) -> list[dict]:
    """
    Скасовує всі відкриті ордери для заданого символу (наприклад BTCUSDT).
    Повертає список скасованих ордерів.
    """
    if not get_open_orders(client, symbol):
        return []

    try:
        canceled_orders = client.cancel_all_open_orders(symbol=symbol)
        return canceled_orders
    except Exception as e:
        raise e

def create_test_client() -> Client:
    """
    Повертає тестовий Binance Client (Testnet / sandbox).

    Returns:
        binance.client.Client
    """

    api_key = "5xGX7qgQZ9FhKtr5TcKYpo7DjWXOPd1MZ1pjwRSaHlSDZPue9bYnRI47AA1GY7MW"
    api_secret = "Xn9jcBisUBUBHAoynkH6xM0p5WnLsXCs3zoY0S3y23jHaKwNpCevmLWvsJWpYFp2"

    client = Client(api_key, api_secret, testnet=True)
    # testnet URL
    client.API_URL = 'https://testnet.binance.vision/api'
    return client


if __name__ == "__main__":
    client = create_test_client()
    cancel_all_orders(client, "ADAUSDT")
    sell_crypto = sell_market_and_wait(client, "ADAUSDT", quantity=float(get_available_quote_balance(client, "ADA")))

    usdt_balance = get_available_quote_balance(client, "USDT")
    ada_balance = get_available_quote_balance(client, "ADA")
    open_orders = get_open_orders(client)

    print(">>>> before buy")
    print(usdt_balance)
    print(ada_balance)
    print(open_orders)

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
    #
    # usdt_balance = get_available_quote_balance(client, "USDT")
    # ada_balance = get_available_quote_balance(client, "ADA")
    # open_orders = get_open_orders(client)
    # print(">>>> after sell")
    # print(usdt_balance)
    # print(ada_balance)
    # print(open_orders)