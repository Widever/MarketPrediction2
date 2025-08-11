import importlib
import time
from collections import defaultdict
from dataclasses import dataclass
from dispersion import reset_disp_cache

import runtime_data as rd
import trading_analyzer as ta
import trading_bot as tb
import chart_builder as chb
import trading_simulator as ts
from data.order import ClosedOrder, SellOrder

all_events = defaultdict(list)

@dataclass
class Event:
    pass

@dataclass
class TriggerEvent(Event):
    trigger: bool | None = None
    trigger_reason: str | None = None

@dataclass
class DecisionEvent(Event):
    decision: bool | None = None
    decision_reason: str | None = None

@dataclass
class MarketEvent(Event):
    type: str | None = None
    closed_order: ClosedOrder | None = None
    balance: float | None = None


class ControlPanelCore:

    @classmethod
    def _reload_all(cls):
        # importlib.reload(chb)
        # importlib.reload(tb)
        # importlib.reload(ts)
        # importlib.reload(ta)
        pass

    @classmethod
    def init_runtime_vars(cls):
        if rd.VARS.simulator is None or 1:
            simulator = ts.TradingSimulator()
            simulator.balance_stable = 1000
            simulator.ohlcv_df = rd.CURRENCY_DATAS["ADAUSDT"].ohlcv_df

            rd.VARS.simulator = simulator
            print("New simulator instance has been set.")

        if rd.VARS.bot is None or 1:
            rd.VARS.bot = tb.TradingBot()

    @classmethod
    def observe(cls, limit: int) -> Event | None:
        cls._reload_all()

        to_index = min(rd.VARS.simulator.current_index + limit, len(rd.VARS.simulator.ohlcv_df)-1)
        while rd.VARS.simulator.current_index < to_index:
            if rd.VARS.simulator.balance > 0.01:
                trigger, trigger_reason = ta.TradingAnalyzer().check_trigger()

                if trigger:
                    trigger_event = TriggerEvent(trigger=trigger, trigger_reason=trigger_reason)
                    all_events[rd.VARS.simulator.current_index].append(trigger_event)

                    print(f">> observed Trigger event. i={rd.VARS.simulator.current_index}")
                    return trigger_event

            events = rd.VARS.simulator.next()
            if events:
                is_stop_loss = events and any(x.trigger == "stop_loss" for x in events)
                if is_stop_loss:
                    rd.VARS.bot.sl_history.append(rd.VARS.simulator.current_index)
                is_sell = events and any(isinstance(x.order, SellOrder) and x.trigger == "filled" for x in events)
                market_event_type = "stop_loss" if is_stop_loss else "sell" if is_sell else None
                market_event = MarketEvent(type=market_event_type, balance=rd.VARS.simulator.balance)
                all_events[rd.VARS.simulator.current_index].append(market_event)

                print(f">> observed Market event. {is_stop_loss=}. i={rd.VARS.simulator.current_index}")
                return market_event

        print(f">> observe limit reached. i={rd.VARS.simulator.current_index}")
        return None

    @classmethod
    def go_to_next_perform(cls):
        cls._reload_all()
        for index, events in sorted(all_events.items(), key=lambda x: x[0]):
            if index <= rd.VARS.simulator.current_index:
                continue

            for event in events:
                if isinstance(event, MarketEvent) and event.type == "perform":
                    rd.VARS.simulator.current_index = index
                    print(f">> go_to_next_perform done. To index {index}.")
                    return

        print(f">> go_to_next_perform done. Not found next perform event.")

    @classmethod
    def go_to_prev_perform(cls):
        cls._reload_all()
        print(f"{len(all_events)=}")
        for index, events in sorted(all_events.items(), key=lambda x: x[0], reverse=True):
            if index >= rd.VARS.simulator.current_index:
                continue

            for event in events:
                if isinstance(event, MarketEvent) and event.type == "perform":
                    rd.VARS.simulator.current_index = index
                    print(f">> go_to_prev_perform done. To index {index}.")
                    return

        print(f">> go_to_prev_perform done. Not found prev perform event.")

    @classmethod
    def go_to_next_sl(cls):
        cls._reload_all()
        for index, events in sorted(all_events.items(), key=lambda x: x[0]):
            if index <= rd.VARS.simulator.current_index:
                continue

            for event in events:
                if isinstance(event, MarketEvent) and event.type == "stop_loss":
                    rd.VARS.simulator.current_index = index
                    print(f">> go_to_next_sl done. To index {index}.")
                    return

        print(f">> go_to_next_sl done. Not found next stop loss event.")

    @classmethod
    def go_to_prev_sl(cls):
        cls._reload_all()
        for index, events in sorted(all_events.items(), key=lambda x: x[0], reverse=True):
            if index >= rd.VARS.simulator.current_index:
                continue

            for event in events:
                if isinstance(event, MarketEvent) and event.type == "stop_loss":
                    rd.VARS.simulator.current_index = index
                    print(f">> go_to_prev_sl done. To index {index}.")
                    return

        print(f">> go_to_prev_sl done. Not found prev stop loss event.")

    @classmethod
    def go_to_start(cls):
        cls._reload_all()
        start_i = 200
        rd.VARS.simulator.current_index = start_i
        print(f">> go_to_start done. i={rd.VARS.simulator.current_index}")

    @classmethod
    def jump_n(cls):
        cls._reload_all()
        n = 100
        to_index = min(rd.VARS.simulator.current_index + n, len(rd.VARS.simulator.ohlcv_df) - 1)
        rd.VARS.simulator.current_index = to_index
        print(f">> jump_n done. i={rd.VARS.simulator.current_index}")


    @classmethod
    def skip_n(cls):
        cls._reload_all()
        n = 200
        to_index = min(rd.VARS.simulator.current_index + n, len(rd.VARS.simulator.ohlcv_df) - 1)
        while rd.VARS.simulator.current_index < to_index:
            cls.skip()

        print(f">> skip_n done. i={rd.VARS.simulator.current_index}")

    @classmethod
    def analyze(cls):
        cls._reload_all()
        return ta.TradingAnalyzer().analyze()

    @classmethod
    def benchmark_small(cls):
        cls._reload_all()
        return cls._benchmark(8000)

    @classmethod
    def benchmark_big(cls):
        cls._reload_all()
        return cls._benchmark(20000)

    @classmethod
    def _benchmark(cls, limit: int):
        start_time = time.time()
        to_index = min(rd.VARS.simulator.current_index + limit, len(rd.VARS.simulator.ohlcv_df) - 1)
        print(f">> start benchmark to index {to_index}. i={rd.VARS.simulator.current_index}")
        while rd.VARS.simulator.current_index < to_index:
            observed = cls.observe(100)
            if observed is None:
                continue
            elif isinstance(observed, TriggerEvent):
                decision, reason = ta.TradingAnalyzer().analyze()
                decision_event = DecisionEvent(decision=decision, decision_reason=reason)
                all_events[rd.VARS.simulator.current_index].append(decision_event)

                if decision:
                    cls.perform()
                else:
                    cls.skip()
            else:
                continue
        print(f">> benchmark done. i={rd.VARS.simulator.current_index}")
        cls.get_event_stats()
        end_time = time.time()
        print(f"Elapsed: {(end_time - start_time) / 60}min.")

    @classmethod
    def reset(cls):
        cls._reload_all()
        cls.init_runtime_vars()

        global all_events
        all_events = defaultdict(list)

        rd.VARS.simulator.reset()
        reset_disp_cache()
        rd.VARS.bot.reset()
        print(">> reset done.")

    @classmethod
    def perform(cls):
        cls._reload_all()
        rd.VARS.bot.perform()
        perform_event = MarketEvent(type="perform")
        all_events[rd.VARS.simulator.current_index].append(perform_event)
        print(f">> perform done. i={rd.VARS.simulator.current_index}")

    @classmethod
    def skip(cls):
        cls._reload_all()
        events = rd.VARS.simulator.next()
        if events:
            is_stop_loss = events and any(x.trigger == "stop_loss" for x in events)
            if is_stop_loss:
                rd.VARS.bot.sl_history.append(rd.VARS.simulator.current_index)
            is_sell = events and any(isinstance(x.order, SellOrder) and x.trigger == "filled" for x in events)
            market_event_type = "stop_loss" if is_stop_loss else "sell" if is_sell else None
            market_event = MarketEvent(type=market_event_type, balance=rd.VARS.simulator.balance)
            all_events[rd.VARS.simulator.current_index].append(market_event)
            print(f">>>> {is_stop_loss=}. {is_sell=}")

        print(f">> skip done. i={rd.VARS.simulator.current_index}")

    @classmethod
    def simulator_info(cls):
        cls._reload_all()
        return rd.VARS.simulator.info()

    @classmethod
    def get_event_stats(cls):
        cls._reload_all()
        last_interval_start_index = None

        intervals_data = []

        interval_sl_events = 0
        interval_sell_events = 0

        total_sl_events = 0
        total_sell_events = 0

        true_decision_stat = defaultdict(int)
        false_decision_stat = defaultdict(int)
        sl_decision_stat = defaultdict(int)

        trigger_stat = defaultdict(int)
        sl_trigger_stat = defaultdict(int)

        last_true_decision_event: DecisionEvent | None = None
        last_true_trigger_event: TriggerEvent | None = None
        last_balance = None
        for index, events in sorted(all_events.items(), key=lambda x: x[0]):
            if last_interval_start_index is None:
                last_interval_start_index = index

            if index - last_interval_start_index > 2000:
                intervals_data.append((last_interval_start_index, index, interval_sl_events, interval_sell_events, last_balance))

                interval_sl_events = 0
                interval_sell_events = 0
                last_interval_start_index = index

            for event in events:
                if isinstance(event, DecisionEvent):
                    if event.decision:
                        last_true_decision_event = event
                        true_decision_stat[event.decision_reason] += 1
                    else:
                        false_decision_stat[event.decision_reason] += 1

                elif isinstance(event, MarketEvent) and event.type == "sell":
                    last_balance = event.balance
                    interval_sell_events += 1
                    total_sell_events += 1
                elif isinstance(event, MarketEvent) and event.type == "stop_loss":
                    last_balance = event.balance
                    interval_sl_events += 1
                    total_sl_events += 1

                    assert last_true_decision_event is not None
                    assert last_true_trigger_event is not None

                    sl_decision_stat[last_true_decision_event.decision_reason] += 1
                    sl_trigger_stat[last_true_trigger_event.trigger_reason] += 1
                elif isinstance(event, TriggerEvent):
                    trigger_stat[event.trigger_reason] += 1
                    last_true_trigger_event = event

        intervals_data.append(
            (last_interval_start_index, index, interval_sl_events, interval_sell_events, last_balance)
        )

        print("Intervals stats:")
        for start_index_, end_index_, sl_events_, sell_events_, balance_ in intervals_data:
            print(f"\t- i<{end_index_}, {sl_events_}/{sell_events_}, {balance_=}")

        true_decision_stat_by_tag = defaultdict(lambda: [0, 0])
        print("True decision stat:")
        for reason_, count_ in sorted(true_decision_stat.items(), key=lambda x: x[1], reverse=True):
            sl_count = sl_decision_stat[reason_]
            tags = reason_.split(";")
            for tag in tags:
                true_decision_stat_by_tag[tag][0] += count_
                true_decision_stat_by_tag[tag][1] += sl_count

            sell_div_sl_stat = (count_ - sl_count) / sl_count if sl_count > 0 else sl_count
            print(f"\t{reason_}: {count_} | sell/sl = {sell_div_sl_stat}")

        print("True decision stat by tag:")
        for tag, (count_, sl_count) in sorted(true_decision_stat_by_tag.items(), key=lambda x: x[1][0], reverse=True):
            sell_div_sl_stat = (count_ - sl_count) / sl_count if sl_count > 0 else sl_count
            print(f"\t{tag}: {count_} | sell/sl = {sell_div_sl_stat}")


        print("False decision stat:")
        for reason_, count_ in sorted(false_decision_stat.items(), key=lambda x: x[1], reverse=True):
            print(f"\t{reason_}: {count_}")

        print("Stop loss decision stat:")
        for reason_, count_ in sorted(sl_decision_stat.items(), key=lambda x: x[1], reverse=True):
            print(f"\t{reason_}: {count_}")

        print("Triggers stat:")
        for reason_, count_ in sorted(trigger_stat.items(), key=lambda x: x[1], reverse=True):
            print(f"\t{reason_}: {count_}")

        print("Stop loss triggers stat:")
        for reason_, count_ in sorted(sl_trigger_stat.items(), key=lambda x: x[1], reverse=True):
            print(f"\t{reason_}: {count_}")

        print(f"TOTAL: {total_sl_events}/{total_sell_events}")
