#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

from enum import StrEnum, auto
import math
from pathlib import Path

from pydantic import BaseModel, Field, PositiveInt, model_validator


class DataSplit(StrEnum):
    train = auto()
    test = auto()
    val = auto()


class BaseDataConfig(BaseModel, validate_assignment=True):
    batch_size: int = 4
    num_workers: int = 4
    prefetch_factor: int = 1

    # limit dataset size
    # possible values: int [1; inf), or -1 - means unlimited
    dataset_max_size: int = Field(default=int(1e20), ge=-1, le=int(1e20))

    # Limit cache size, usefull for debigging, to not wait to cache all dataset folder/archive
    cache_max_size: PositiveInt | None = None

    cache_dir: Path = Path("~/cache").expanduser()
    overwrite_cache: bool = False
    with_cache: bool = True

    eval_split: DataSplit = DataSplit.train

    # train test val
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05

    @model_validator(mode="after")
    def validate_split(self):
        assert math.isclose(self.train_ratio + self.val_ratio + self.test_ratio, 1.0), (
            f"{self.train_ratio=} + {self.val_ratio=} + {self.test_ratio=}!= 0"
        )
        return self
