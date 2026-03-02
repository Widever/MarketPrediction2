import dataclasses
import itertools
import math
import os
import time
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
    comb: tuple[str, ...] | None
    count_: int
    sl_count: int
    uniformity: float
    uniformity2: float
    k: float
    k2: float
    peak_down_k: float
    peak_down_n: int
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


def grade_comb(comb_df: pd.DataFrame, comb: tuple[str, ...], interval_bins=None, timestamp_range=None) -> CombGrade:
    count_ = len(comb_df)
    sl_count = int(comb_df["sl"].sum())

    if interval_bins is None:
        uniformity = 0.0
    else:
        uniformity = _comb_uniformity(comb_df, interval_bins)

    k = _comb_k(comb_df)

    uniformity2 = _comb_uniformity_2(comb_df, timestamp_range)

    return CombGrade(
        comb=comb,
        count_=count_,
        sl_count=sl_count,
        uniformity=uniformity,
        uniformity2=uniformity2,
        k=k,
        k2=_comb_k2(comb_df),
        peak_down_k=_peak_down_k(comb_df),
        peak_down_n=_peak_down_n(comb_df),
    )

def f(x: float) -> float:
    a = 300
    if x <= 0:
        x = 0
    elif x >= a:
        x = a
    return 1 + 2 * x / a

def choose_comb(train_marked_points_df: pd.DataFrame, verify_marked_points_df: pd.DataFrame,
                all_combs: list[tuple[str, ...]], selected_combs: list[CombGrade], interval_bins) -> CombGrade:
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
    comb_grades_sorted: list[CombGrade] = list(sorted(comb_grades, key=lambda x: x.peak_down_k , reverse=False))

    i = 0

    for comb_grade in comb_grades_sorted:

        # if comb_grade.k < min_comb_k:
        #     continue
        #

        x = len(comb_grade.comb)

        start_c = 300
        end_c = 100
        l = 10
        # min_comb_count_ = 2000 * math.exp(-0.5 * x) + 0
        min_comb_count_ = start_c - (start_c - end_c) / l * x

        if comb_grade.count_ < min_comb_count_:
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

def optimal_combs(limit_comb_n=10, selected_combs=None) -> list[CombGrade]:
    full_time_start = time.time()

    train_marked_points_df = pd.read_csv(f"{data_dir}/marked_points_train.csv")
    verify_marked_points_df = pd.read_csv(f"{data_dir}/marked_points_verify.csv")

    interval_bins = pd.cut(train_marked_points_df["index"], bins=12).cat.categories
    print(f"Full df len: {len(train_marked_points_df)}")

    tags = [x for x in train_marked_points_df.columns if x.startswith("#tag_")]
    # combinations = list(itertools.combinations(tags, 1))
    # combinations += list(itertools.combinations(tags, 2))
    # combinations += list(itertools.combinations(tags, 3))
    # combinations += list(itertools.combinations(tags, 4))
    # combinations += list(itertools.combinations(tags, 5))

    # dict_of_field_tags = {}
    # f_names = [f.name for f in dataclasses.fields(PointValues)]
    # f_names = [x for x in f_names if x not in ("btc_price_up", "all_same_price_dir", "ada_price_up")]

    # for f_name in f_names:
    #     field_tags = [x for x in tags if x.startswith(f"#tag_{f_name}")]
    #     dict_of_field_tags[f_name] = field_tags
    #
    # field_combs = []
    # field_combs += list(itertools.combinations(f_names, 4))
    # field_combs += list(itertools.combinations(f_names, 5))
    #
    # combinations = []
    # for field_comb in field_combs:
    #     list_of_field_tags = [dict_of_field_tags[f] for f in field_comb]
    #     combinations += list(itertools.product(*list_of_field_tags))

    start_combs: list[tuple[str, ...]] = list(itertools.combinations(tags, 2))
    combs: list[tuple[str, ...]] = start_combs

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
                                          interval_bins)

            if len(comb_grade.comb) < 10:

                filled_fields = {tag_to_field(x) for x in comb_grade.comb}
                remaining_tags = [x for x in tags if tag_to_field(x) not in filled_fields]
                new_combs = [(*comb_grade.comb, x) for x in remaining_tags]
                combs = new_combs
                continue

            selected_combs.append(comb_grade)
            combs = start_combs

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