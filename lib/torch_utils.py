#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import random_split, Dataset, Subset
import numpy as np


def describe(t: torch.Tensor) -> dict[str, Any]:
    def percentile(t: torch.Tensor, q: float) -> torch.Tensor:
        return t.kthvalue(int(q * (t.numel() - 1) + 1)).values

    shape = t.shape
    t = t.flatten()

    return {
        "shape": shape,
        "count": torch.tensor(t.shape, dtype=torch.float32),
        "mean": torch.mean(t),
        "std": torch.std(t, unbiased=False),
        "min": torch.min(t),
        "25%": percentile(t, 0.25),
        "50% (median)": torch.median(t),
        "75%": percentile(t, 0.75),
        "max": torch.max(t),
    }


def make_batch(*args: Tensor) -> tuple[Tensor, ...] | Tensor:
    def unsqueeze_if_needed(t: Tensor) -> Tensor:
        return t.unsqueeze(0) if len(t.shape) == 2 else t

    if len(args) == 1:
        return unsqueeze_if_needed(args[0])
    return () if not args else (unsqueeze_if_needed(args[0]),) + make_batch(*args[1:])


def assert_same_shape(t1: Tensor, t2: Tensor):
    assert t1.shape == t2.shape, f"Shape mistmatch! {t1.shape} != {t2.shape}"


def split_dataset(dataset: Dataset, lengths: Sequence[float], random: bool = True) -> list[Subset]:
    # Ensure lengths are valid (percentages sum up to 1)
    if not np.isclose(sum(lengths), 1.0):
        raise ValueError("Lengths must sum to 1.")

    total_size = len(dataset)
    absolute_lengths = [int(length * total_size) for length in lengths]

    # Adjust the last split to account for rounding errors
    absolute_lengths[-1] = total_size - sum(absolute_lengths[:-1])

    if random:
        # Random split using torch.utils.data.random_split
        return random_split(dataset, absolute_lengths)
    else:
        # Non-random split (sequential slicing)
        indices = torch.arange(total_size)
        splits = []
        start = 0
        for length in absolute_lengths:
            end = start + length
            splits.append(Subset(dataset, indices[start:end]))
            start = end
        return splits
