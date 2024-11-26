import importlib
import chart_builder as chb
from data.runtime_data import CURRENCY_DATAS


def update():
    for currency_data in CURRENCY_DATAS:
        currency_data.update()

def run():
    importlib.reload(chb)
    chb.run()

