#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import os
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from expression import pipe
import lmdb_cache
from lmdb_cache import lmdb_exists, LMDBCacheCompressed as LMDBCacheCLS
from torch.utils.data import Dataset
from typing_extensions import Self
import logging

from lib.utils.functional import prefetch_iter
from lib.utils.progress_bar import ProgressBar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheDataset(Dataset[T]):
    def __init__(self, cache_dir: os.PathLike):
        self.cache_dir = Path(cache_dir)

        assert lmdb_exists(self.cache_dir)

        self.cached_lmdb_dict = LMDBCacheCLS(self.cache_dir)

    def __getitem__(self, item: Any) -> T:
        data = self.cached_lmdb_dict[item]
        return data

    def __len__(self):
        return len(self.cached_lmdb_dict)

    @classmethod
    def from_iterable(cls, iter: Iterable[dict[str, Any]], cache_dir: os.PathLike) -> Self:
        cache_dir = Path(cache_dir)
        size = 0

        def count_size(iterable):
            nonlocal size

            for i in iterable:
                yield i
                size += 1

        LMDBCacheCLS.from_iterable(
            cache_dir,
            pipe(iter, count_size),
            block_size=100 * 1024**2,
            batch_size=128,
        )
        if size == 0:
            shutil.rmtree(cache_dir)
            raise ValueError(f"Dataset '{cache_dir}' size is 0!")

        return cls(cache_dir)


def build_cached_dataset(
    build_iterable_dataset: Callable[[], Iterable],
    cache_dir: Path,
    overwrite_cache: bool = False,
    prefetch_size=100_000,
):
    if overwrite_cache:
        logger.info(f"Removing cache: '{cache_dir}'")
        shutil.rmtree(cache_dir, ignore_errors=True)

    if lmdb_cache.lmdb_cache.lmdb_exists(cache_dir):
        logger.info(f"Reading cache from: '{cache_dir}'")
        cached_dataset = CacheDataset(cache_dir)
    else:
        progress_bar = ProgressBar()
        logger.info(f"Writing cache to: '{cache_dir}'")
        with progress_bar:
            cached_dataset = pipe(
                build_iterable_dataset(),
                progress_bar.capture_progress("Caching strokes dataset"),
                prefetch_iter(prefetch_size),
                lambda x: CacheDataset.from_iterable(x, cache_dir),
            )

    return cached_dataset
