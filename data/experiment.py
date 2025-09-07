import numpy as np
from math import sqrt, exp
import itertools

import pandas as pd


def uniformity_sl(wide_df):

    wide_df["interval"] = pd.cut(wide_df["index"], bins=bins)

    intervals_stat = wide_df.groupby("interval", observed=False).agg(
        count_=("index", "size"),
        sl_count_=("sl", "sum"),
    ).reset_index()

    total_count = intervals_stat["count_"].sum()
    intervals_stat["prop_count"] = intervals_stat["count_"] / total_count
    max_prop_count = intervals_stat["prop_count"].max()

    return 1 - max_prop_count

if __name__ == "__main__":
    df = pd.read_csv("optimize/marked_events_40k.csv")

    tags = set(tag for index, row in df.iterrows() for tag in row["reason"].split(";"))
    wide_df_data = []

    for index, row in df.iterrows():
        reason = row["reason"]
        reason_tags = reason.split(";")

        wide_row = {k: 1 if k in reason_tags else 0 for k in tags}
        wide_row["index"] = row["index"]
        wide_row["sl"] = row["sl"]
        wide_df_data.append(wide_row)

    wide_df = pd.DataFrame(wide_df_data)
    interval_bins = pd.cut(wide_df["index"], bins=15).cat.categories

    comb = ('up_strick_0', 'extreme_disp_many', 'max_disp_change<')
    mask = (wide_df[list(comb)] == 1).all(axis=1)
    comb_df = wide_df[mask]

    uniformity_sl(comb_df, interval_bins)
