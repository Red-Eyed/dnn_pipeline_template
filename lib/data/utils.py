#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from expression import curry_flip
from torch import Tensor
import torch
import polars as pl


def dataframe_from_dict(data: dict[str, np.ndarray]) -> pd.DataFrame:
    same_len = len(set(v.shape[0] for v in data.values())) == 1
    assert same_len, "All values have to be the same len: {}".format({k: v.shape[0] for k, v in data.items()})
    new_dict = {}
    for key, value in data.items():
        if value.ndim == 2:
            for i in range(value.shape[1]):
                new_dict[f"{key}_dim[{i}]"] = value[:, i]
        elif value.ndim == 1:
            new_dict[key] = value
        else:
            raise ValueError(f"Invalid data: {value.shape=}")

    df = pd.DataFrame.from_dict(new_dict)
    return df


@curry_flip(1)
def groupby_to_dict(df: pd.DataFrame, group_col: str) -> dict[str, pd.DataFrame]:
    return {str(group): group_df.reset_index(drop=True) for group, group_df in df.groupby(group_col)}


def list_of_dict_to_dict_of_lists(data: list[dict]) -> dict:
    result = defaultdict(list)
    for entry in data:
        for key, value in entry.items():
            result[key].append(value)
    return dict(result)


def to_padded_tensor(tensors: list[Tensor], pad_id: int | float):
    nested = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
    ret = nested.to_padded_tensor(pad_id)
    return ret


def _compute_lengths(total: int, proportions: list[float]) -> list[int]:
    """
    Compute sequential split lengths given total size and list of proportions.
    The last split takes the remainder.
    """
    if not proportions:
        return []
    if any(p < 0 for p in proportions):
        raise ValueError("All split proportions must be non-negative")
    total_prop = sum(proportions)
    if total_prop <= 0:
        raise ValueError("Sum of split proportions must be positive")

    # initial lengths by floor of weighted size
    lengths = [int((p / total_prop) * total) for p in proportions]
    # assign remainder to last split
    remainder = total - sum(lengths)
    lengths[-1] += remainder
    return lengths


def sequential_split(df: pl.DataFrame, proportions: list[float]) -> list[pl.DataFrame]:
    """
    Split a DataFrame sequentially into multiple parts according to proportions.
    Similar to torch.random_split but deterministic and sequential.
    """
    n = df.height
    lengths = _compute_lengths(n, proportions)

    parts: list[pl.DataFrame] = []
    offset = 0
    for length in lengths:
        parts.append(df.slice(offset, length))
        offset += length
    return parts
