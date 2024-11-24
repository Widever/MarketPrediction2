import datetime as dt
import pandas as pd


class BinanceDataProvider:

    def __init__(self):
        pass

    @classmethod
    def get_currency_ohlc_data(cls, currency: str, start_datetime: dt.datetime) -> pd.DataFrame:
        pass
