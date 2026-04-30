import dataclasses
import itertools
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd

from mp.optimizer.mark import PointValues
from mp.optimizer.parallel_grade_comb import grade_combs_parallel

min_comb_k = 2
min_comb_count = 70
min_comb_verify_count = 30
min_verify_comb_k = 2
print_combs_or_look = False

data_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(data_dir, f"optimize_main_dir")

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


def get_exclude_combs_mask(marked_points_df: pd.DataFrame, combs: list[tuple[str, ...]]) -> pd.Series:
    n = len(marked_points_df)

    combs_masks = [
        marked_points_df[list(comb)].astype(bool).all(axis=1)
        for comb in combs
    ]

    any_comb = np.logical_or.reduce(combs_masks) if combs_masks else np.zeros(n, dtype=bool)

    mask = np.ones(n, dtype=bool)
    skip = 0
    scopes = marked_points_df["scope"].to_numpy()

    for i in range(n):
        if skip > 0:
            skip -= 1
            mask[i] = False
            continue

        if any_comb[i]:
            mask[i] = False
            skip = int(scopes[i])

    return pd.Series(mask, index=marked_points_df.index)


def _comb_uniformity(comb_df, interval_bins) -> float:
    comb_df_ = pd.DataFrame()
    comb_df_["index"] = comb_df["index"]
    comb_df_["sl"] = comb_df["sl"]
    comb_df_["interval"] = pd.cut(comb_df_["index"], bins=interval_bins)

    intervals_stat = comb_df_.groupby("interval", observed=False).agg(
        count_=("index", "size"),
        sl_count_=("sl", "sum"),
    ).reset_index()

    total_count = intervals_stat["count_"].sum()
    intervals_stat["prop_count"] = intervals_stat["count_"] / total_count
    max_prop_count = intervals_stat["prop_count"].max()

    return float(1 - max_prop_count)


def _comb_k(comb_df) -> float:
    count_ = len(comb_df)
    sl_count = int(comb_df["sl"].sum())
    k = (count_ - sl_count) / sl_count if sl_count > 0 else count_
    return k

def _comb_k2(comb_df) -> float:
    count_ = len(comb_df)
    sl_count = int(comb_df["sl"].sum())
    k = (count_ - sl_count) - sl_count
    if k < 0:
        k = 0
    return k

def _peak_down_n(comb_df) -> int:
    peak_down_n = int(comb_df["peak_down"].sum())
    return peak_down_n

def _peak_down_k(comb_df) -> float:
    count_ = len(comb_df)
    peak_down_count = _peak_down_n(comb_df)
    k = (count_ - peak_down_count) / peak_down_count if peak_down_count > 0 else count_
    if k < 0:
        k = 0
    return k

def _comb_uniformity_2(comb_df, timestamp_range):
    values = comb_df["timestamp"].to_numpy()
    n = len(values)

    if n == 0:
        return 0.0

    k = math.ceil(n * 0.5)

    min_dist = float("inf")

    for i in range(n - k + 1):
        dist = values[i + k - 1] - values[i]
        min_dist = min(min_dist, dist)

    return min_dist / timestamp_range

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

def bayesian_peak_down_winrate(comb_df, alpha=10, beta=10, props=None):
    profit, loss = profit_loss_in_df(comb_df)

    if props is not None:
        props["loss"] = loss
        props["profit"] = profit
        props["total"] = len(comb_df)
    # return profit
    # if profit <= 500:
    #     return 0.0

    return (profit + alpha) / (profit + loss + alpha + beta)

def grade_comb(comb_df: pd.DataFrame, comb: tuple[str, ...], interval_bins=None, timestamp_range=None) -> CombGrade:
    props = {}
    score = bayesian_peak_down_winrate(comb_df, alpha=50, beta=50, props=props)

    return CombGrade(
        comb=comb,
        score=score,
        props=props
    )

def f(x: float) -> float:
    a = 300
    if x <= 0:
        x = 0
    elif x >= a:
        x = a
    return 1 + 2 * x / a

def choose_comb(train_marked_points_df: pd.DataFrame, verify_marked_points_df: pd.DataFrame,
                all_combs: list[tuple[str, ...]], selected_combs: list[CombGrade], interval_bins, start_c, end_c, l) -> CombGrade:
    print(f"Selected combs count: {len(selected_combs)}.")
    if selected_combs:
        exclude_mask = get_exclude_combs_mask(train_marked_points_df,
                                                   [selected_comb.comb for selected_comb in selected_combs])
        train_marked_points_df = train_marked_points_df[exclude_mask].reset_index(drop=True)

    print(f"Df len after exclude selected: {len(train_marked_points_df)}.")

    # comb_grades = []
    # for comb in all_combs:
    #     select_mask = get_select_combs_mask(train_marked_points_df, [comb])
    #     comb_df = train_marked_points_df[select_mask]
    #
    #     comb_grade = grade_comb(comb_df, comb, interval_bins)
    #     comb_grades.append(comb_grade)

    comb_grades = grade_combs_parallel(all_combs, selected_combs)

    print("Grading finished. Sorting comb_grades...")
    comb_grades_sorted: list[CombGrade] = list(sorted(comb_grades, key=lambda x: x.score , reverse=True))

    i = 0

    for comb_grade in comb_grades_sorted:

        # if comb_grade.k < min_comb_k:
        #     continue
        #

        x = len(comb_grade.comb)

        # start_c = 150
        # end_c = 10
        # l = 15
        # min_comb_count_ = 2000 * math.exp(-0.5 * x) + 0
        min_comb_count_ = start_c - (start_c - end_c) / l * x

        if comb_grade.props["profit"] < end_c:
            continue

        # if comb_grade.uniformity2 < 0.0:
        #     # print(comb_grade.uniformity2)
        #     continue

        return comb_grade

        #
        # if comb_grade.uniformity < 0.6:
        #     continue

        # Check overlearning
        if selected_combs:
            verify_exclude_mask = get_exclude_combs_mask(verify_marked_points_df,
                                                              [selected_comb.comb for selected_comb in selected_combs])
            verify_marked_points_df = verify_marked_points_df[verify_exclude_mask].reset_index(drop=True)

        verify_select_mask = get_select_combs_mask(verify_marked_points_df, [comb_grade.comb])
        verify_comb_df = verify_marked_points_df[verify_select_mask]

        verify_comb_grade = grade_comb(verify_comb_df, comb_grade.comb, interval_bins)

        if verify_comb_grade.count_ < min_comb_verify_count:
            continue

        if verify_comb_grade.k < comb_grade.k * 0.8:
            continue
        #
        # if verify_comb_grade.count_ < (comb_grade.count_ / 2.7):
        #     continue

        verify_comb_grade.comb = None
        comb_grade.verify_grade = verify_comb_grade
        if print_combs_or_look:
            print(comb_grade)
            if i > 1000:
                break
            i += 1
        else:
            return comb_grade

    raise RuntimeError("Not found comb.")

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


def generate_combinations(attr_lists, m):
    result = []

    list_combinations = list(itertools.combinations(attr_lists, m))

    if len(list_combinations) > 50:
        list_combinations = random.sample(list_combinations, 50)

    print(f"List combinations len: {len(list_combinations)}")

    # вибираємо m списків із n
    for selected_lists in list_combinations:
        # беремо по одному атрибуту з кожного вибраного списку
        for combo in itertools.product(*selected_lists):
            result.append(combo)

    if len(result) > 100000:
        result = random.sample(result, 100000)

    return result

def optimal_combs(limit_comb_n=10, selected_combs=None, start_c=None, end_c=None, l=None) -> list[CombGrade]:
    full_time_start = time.time()

    train_marked_points_df = pd.read_csv(f"{data_dir}/marked_points_train.csv")
    verify_marked_points_df = pd.read_csv(f"{data_dir}/marked_points_verify.csv")

    interval_bins = pd.cut(train_marked_points_df["index"], bins=12).cat.categories
    print(f"Full df len: {len(train_marked_points_df)}")

    tags = [x for x in train_marked_points_df.columns if x.startswith("#tag_")]

    tags_by_field = defaultdict(list)
    for tag in tags:
        tag_field = tag_to_field(tag)
        tags_by_field[tag_field].append(tag)

    attr_lists = list(tags_by_field.values())
    print(f"Tags by field len: {len(attr_lists)}")
    combs = generate_combinations(attr_lists, 5)

    print(f"All combs len: {len(combs)}")

    if selected_combs is None:
        selected_combs: list[CombGrade] = []

    while True:
        try:
            if len(selected_combs) > limit_comb_n - 1:
                break

            print("Start choosing comb...")
            start_time = time.time()

            comb_grade = choose_comb(train_marked_points_df, verify_marked_points_df, combs, selected_combs,
                                          interval_bins, start_c, end_c, l)

            selected_combs.append(comb_grade)

            print("Chosen comb:")
            print(comb_grade)

            end_time = time.time()
            print(f"For this comb elapsed {end_time - start_time}s.")

        except RuntimeError as e:
            print(e)
            break

    print(f"Selected {len(selected_combs)} combs:")
    for comb in selected_combs:
        print(f"{str(comb)},")
    print()

    full_time_end = time.time()

    print(f"Total elapsed time: {full_time_end - full_time_start}s.")

    return selected_combs