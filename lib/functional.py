#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
from typing import Callable, Generic, Mapping, TypeVar

from expression import curry_flip

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


class _LazyMapHelper(Generic[T1, T3]):
    def __init__(self, obj: Mapping[T1, T2], fn: Callable[[T2], T3]):
        self.obj = obj
        self.fn = fn

    def __getitem__(self, key: T1) -> T3:
        # Apply the transformation lazily when the item is accessed
        return self.fn(self.obj[key])

    def __len__(self):
        return len(self.obj)


def lazy_map(fn: Callable[[T1], T2]):
    """Higher-order function to lazily apply `fn` to the result of `__getitem__`."""

    def apply_transformation(obj):
        return _LazyMapHelper(obj, fn)

    return apply_transformation


def parallel_map(fn: Callable[[T1], T2], it: list[T1]) -> list[T2]:
    """
    Parallel map, usefull to call GIL free functions, such as C-API
    """
    with ThreadPoolExecutor(max_workers=cpu_count() - 1) as ex:
        futures = [ex.submit(fn, i) for i in it]
        return [f.result() for f in futures]


@curry_flip(1)
def apply_to_all(data: list | tuple | dict, fn):
    def apply(d):
        if isinstance(d, (list, tuple)):
            return type(d)(map(apply, d))
        elif isinstance(d, dict):
            return {k: apply(v) for k, v in d.items()}
        else:
            return fn(d)

    return apply(data)
