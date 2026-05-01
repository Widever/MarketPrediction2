import dataclasses
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Self, Iterable, Generator

import itertools
import numpy as np
import pandas as pd

from mp.optimizer.mark import PointValues

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"optimize_main_dir")

_train_df = None

@dataclass(slots=True)
class CombGrade:
    comb: tuple[str, ...] | None = None
    score: float = 0.0
    props: dict = dataclasses.field(default_factory=dict)
    verify_grade: Self | None = None

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

def get_deeper_combs(combs: Generator[list[tuple[str, ...]], None, None], tags: list[str], batch_size: int) -> Generator[list[tuple[str, ...]], None, None]:
    tag_index = {tag: i for i, tag in enumerate(tags)}
    tag_to_field_dict = {tag: tag_to_field(tag) for tag in tags}

    deeper_combs = []
    for combs_batch in combs:
        for comb in combs_batch:
            filled_fields = {tag_to_field_dict[x] for x in comb}
            last_index = max(tag_index[x] for x in comb)  # <-- key change
            remaining_tags = [
                x for x in tags
                if tag_to_field_dict[x] not in filled_fields and tag_index[x] > last_index
            ]
            deeper_combs.extend((*comb, x) for x in remaining_tags)
            if len(deeper_combs) > batch_size:
                yield deeper_combs
                deeper_combs = []

    yield deeper_combs
    return None

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

def get_combs_of_len(l: int, min_profit_n: int = 100) -> list[tuple[str, ...]]:
    # from mp.optimizer.parallel_reduce_combs import reduce_combs_parallel
    from mp.optimizer.parallel_reduce_combs_gpu import reduce_combs_parallel_gpu
    global _train_df
    _train_df = pd.read_csv(f"{data_dir}/marked_points_train.csv")

    tags = [x for x in _train_df.columns if x.startswith("#tag_")]
    print(f"Number of tags: {len(tags)}.")

    combs = []
    min_profit_n_values = [0, 500, 400, 300, 200, 100]
    for i in range(1, l+1):
        start = time.perf_counter()
        combs = get_deeper_combs(combs, tags)
        end = time.perf_counter()
        print(f"Get {len(combs)} combs of len {i}.")
        print(f"Get deeper combs time: {end-start}s.")

        start = time.perf_counter()
        # combs = reduce_combs_parallel(combs, min_profit_n_values[i])
        combs = reduce_combs_parallel_gpu(combs, min_profit_n_values[i])
        end = time.perf_counter()
        print(f"Get {len(combs)} combs of len {i} after reduce.")
        print(f"Reduce time: {end-start}s.")

    return combs


if __name__ == "__main__":
    combs_ = get_combs_of_len(5)

    # for comb_ in combs_:
    #     print(comb_)