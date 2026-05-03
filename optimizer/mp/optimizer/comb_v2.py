import dataclasses
import gc
import math
import os
import time
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Self, Iterable, Generator

import itertools
import numpy as np
import pandas as pd
import random
from mp.optimizer.mark import PointValues
from mp.optimizer.parallel_reduce_combs import grade_combs_parallel_cpu

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"optimize_main_dir")

_train_df = None

@dataclass(slots=True, frozen=True)
class CombGrade:
    comb: tuple[str, ...] | None = None
    bayesian_winrate: float = 0.0
    profit: int = 0
    total: int = 0

    def __hash__(self):
        return hash(self.comb)

    def __eq__(self, other):
        if not isinstance(other, CombGrade):
            return NotImplemented
        return self.comb == other.comb

def get_select_combs_mask(marked_points_df: pd.DataFrame, combs: list[tuple[str, ...]]) -> pd.Series:
    n = len(marked_points_df)

    combs_masks = [
        marked_points_df[list(comb)].astype(bool).all(axis=1)
        for comb in combs
    ]

    any_comb = np.logical_or.reduce(combs_masks) if combs_masks else np.zeros(n, dtype=bool)

    mask = np.zeros(n, dtype=bool)
    skip = 0
    scopes = marked_points_df["scope"].to_numpy()

    for i in range(n):
        if skip > 0:
            skip -= 1
            continue

        if any_comb[i]:
            mask[i] = True
            skip = int(scopes[i])

    return pd.Series(mask, index=marked_points_df.index)

def get_select_combs_mask_ignore_scope(marked_points_df: pd.DataFrame, combs: list[tuple[str, ...]]) -> pd.Series:
    if not combs:
        return pd.Series(False, index=marked_points_df.index)

    combs_masks = [
        marked_points_df[list(comb)].all(axis=1)
        for comb in combs
    ]

    any_comb = np.logical_or.reduce(combs_masks)

    return pd.Series(any_comb, index=marked_points_df.index)

def profit_loss_in_df(comb_df):
    # Peak down
    profit = len(
        comb_df[
            (comb_df["peak_down"]) &
            (~comb_df["peak_up"])
            # (comb_df["change_from_last_peak"] < -0.025)
            # (comb_df["len_from_last_peak"] > 30)
            # (comb_df["avg_log_return"] < -0.00)
        ]
    )

    # Middle rising
    # profit = len(
    #     comb_df[
    #         (comb_df["ada_rise_from_low"] > 0.02) &
    #         # (comb_df["rising"]) &
    #         # (~comb_df["peak_up"]) &
    #         # (comb_df["change_from_last_peak"] < 0.03) &
    #         # (comb_df["len_from_last_peak"] < 20)
    #         (comb_df["avg_log_return_ratio"] < 0)
    #     ]
    # )
    loss = len(comb_df) - profit
    return profit, loss

def reduce_combs(combs: list[tuple[str, ...]], min_profit_n: int) -> list[tuple[str, ...]]:
    new_combs = []

    for comb in combs:
        select_mask = get_select_combs_mask_ignore_scope(_train_df, [comb])
        comb_df = _train_df[select_mask]
        profit, loss = profit_loss_in_df(comb_df)
        if profit > min_profit_n:
            new_combs.append(comb)

    return new_combs

def tag_to_field(tag: str) -> str:
    f_names = [f.name for f in dataclasses.fields(PointValues)]

    value = tag.removeprefix("#tag_")
    parts = value.split("_")

    res = []
    for part in parts:
        res.append(part)
        if (joined_res := "_".join(res)) in f_names:
            return joined_res

    raise RuntimeError(f"Tag {tag} is unknown field.")

# def tag_to_field(tag: str) -> str:
#     return tag.split("_")[0]

def build_deeper_combs(combs: list[tuple[str, ...]], tags: list[str]) -> list[tuple[str, ...]]:

    if not combs:
        return [(tag,) for tag in tags]

    # Precompute fields for all tags once
    tag_fields = [(tag, tag_to_field(tag)) for tag in tags]

    result = []
    for comb in combs:
        used_fields = {tag_to_field(tag) for tag in comb}
        for tag, field in tag_fields:
            if field not in used_fields:
                result.append(comb + (tag,))
    return result


def count_deeper_combs(combs: list[tuple[str, ...]], tags: list[str]) -> int:
    field_tag_counts = Counter(tag_to_field(tag) for tag in tags)

    total = 0
    for c in combs:
        used_fields = {tag_to_field(tag) for tag in c}
        for field, count in field_tag_counts.items():
            if field not in used_fields:
                total += count
    return total

def calc_number_of_combs(tags, l):
    field_to_tags_dict = defaultdict(list)
    for tag in tags:
        field_to_tags_dict[tag_to_field(tag)].append(tag)

    counts = [len(v) for v in field_to_tags_dict.values()]
    n = len(counts)

    if l > n:
        return 0

    dp = [0] * (l + 1)
    dp[0] = 1

    for i, c in enumerate(counts):
        for j in range(min(i + 1, l), 0, -1):
            dp[j] += dp[j - 1] * c

    return dp[l]

def bayesian_winrate(profit: int, total: int) -> float:
    alpha = 10
    beta = 10
    loss = total - profit
    return (profit + alpha) / (profit + loss + alpha + beta)

def get_combs_of_len(
    l: int,
    min_profit_n_values: tuple[int, ...] = (0,),
    get_by_profit_n: int = 1000,
    get_by_bayesian_winrate_n: int = 1000,
    profit_signal_mode: int = 1,
) -> list[CombGrade]:

    if len(min_profit_n_values) < l+1:
        raise ValueError(f"{len(min_profit_n_values)=} < {l+1=}.")

    # from mp.optimizer.parallel_reduce_combs import reduce_combs_parallel
    from mp.optimizer.parallel_reduce_combs_gpu import grade_combs_parallel_gpu
    global _train_df
    if _train_df is None:
        _train_df = pd.read_csv(f"{data_dir}/marked_points_train.csv")

    tags = [x for x in _train_df.columns if x.startswith("#tag_")]
    print(f"Number of tags: {len(tags)}.")

    combs = []
    selected_graded_combs = None
    for i in range(1, l+1):
        print(f"Input combs for get deeper count: {len(combs)}.")
        start = time.perf_counter()
        combs = build_deeper_combs(combs, tags)
        initial_len = len(combs)
        end = time.perf_counter()
        print(f"Deeper combs count: {initial_len}, {i=}, time: {end - start}s.")

        print(f"Grade combs...")
        start = time.perf_counter()
        graded_combs = grade_combs_parallel_gpu(combs, profit_signal_mode)
        end = time.perf_counter()
        # graded_combs = grade_combs_parallel_cpu(combs)
        end = time.perf_counter()
        print(f"Grade combs time: {end - start}s.")

        start = time.perf_counter()
        graded_combs = [
            CombGrade(comb=comb, profit=profit, total=total, bayesian_winrate=bayesian_winrate(profit, total))
            for comb, profit, total in graded_combs if profit > min_profit_n_values[i]
        ]
        end = time.perf_counter()
        print(f"Filter combs by min_profit_n_value: {min_profit_n_values[i]}, count: {len(graded_combs)}/{initial_len}, time: {end - start}s.")

        start = time.perf_counter()
        graded_combs_sorted = list(sorted(graded_combs, key=lambda x: x.bayesian_winrate, reverse=True))
        graded_combs_selected_by_bayesian_winrate = graded_combs_sorted[:get_by_bayesian_winrate_n]
        graded_combs_selected_by_bayesian_winrate_worst = set(graded_combs_sorted[-get_by_bayesian_winrate_n:])
        end = time.perf_counter()
        print(f"Select graded combs by bayesian winrate, len: {len(graded_combs_selected_by_bayesian_winrate)}/{initial_len}, time: {end - start}s.")

        start = time.perf_counter()
        graded_combs_exclude_worst = [x for x in graded_combs if x not in graded_combs_selected_by_bayesian_winrate_worst]
        graded_combs_sorted = list(sorted(graded_combs_exclude_worst, key=lambda x: x.profit, reverse=True))
        graded_combs_selected_by_profit = graded_combs_sorted[:get_by_profit_n]
        end = time.perf_counter()
        print(f"Select graded combs by profit, len: {len(graded_combs_selected_by_profit)}/{initial_len}, time: {end - start}s.")


        start = time.perf_counter()
        selected_graded_combs = graded_combs_selected_by_profit + graded_combs_selected_by_bayesian_winrate
        combs = [x.comb for x in selected_graded_combs]
        end = time.perf_counter()
        print(f"Total selected combs: {len(combs)}/{initial_len}, {i=}, time: {end - start}s.")

    return selected_graded_combs

def optimize_combs(
    l,
    get_by_profit_n,
    get_by_bayesian_winrate_n,
    m,
    profit_signal_mode: int,
):
    min_profit_n_values = (m,)*(l+1)
    combs_ = get_combs_of_len(l, min_profit_n_values, get_by_profit_n, get_by_bayesian_winrate_n, profit_signal_mode)
    graded_combs_sorted = list(sorted(combs_, key=lambda x: x.bayesian_winrate, reverse=True))
    result = [graded_combs_sorted[0]]
    return result

def memory_leak_test():
    from mp.optimizer.parallel_reduce_combs_gpu import grade_combs_parallel_gpu
    global _train_df
    if _train_df is None:
        _train_df = pd.read_csv(f"{data_dir}/marked_points_train.csv")
    tags = [x for x in _train_df.columns if x.startswith("#tag_")]

    combs_full = []
    combs_full = build_deeper_combs(combs_full, tags)
    combs_full = random.sample(combs_full, 100)

    combs_to_deep = combs_full
    for i in range(15):
        print(f">>>>>>>>>>> ITERATION {i}")
        combs_ = build_deeper_combs(combs_to_deep, tags)
        grade_combs_parallel_gpu(combs_)
        combs_to_deep = [tuple(x) for x in random.sample(combs_, 100)]


if __name__ == "__main__":
    # min_profit_n_values = (0, 100, 100, 100, 100, 100, 100, 100, 100)
    # # combs_ = get_combs_of_len(5, min_profit_n_values, 1000, 1000)
    # tags_ = ["a_1", "a_2", "a_3", "b_1", "b_2", "b_3", "c_1", "c_2", "c_3", "d_1", "d_2", "d_3"]
    # input_combs_ = []
    # combs_ = build_deeper_combs(input_combs_, tags_)
    # for comb_ in combs_:
    #     print(comb_)

    memory_leak_test()