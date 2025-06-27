#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import collections
import collections.abc
from concurrent.futures import ThreadPoolExecutor
from operator import getitem, setitem
from queue import Queue
from threading import Thread
from typing import Callable, Generic, Iterable, Iterator, Protocol, TypeVar, runtime_checkable

from expression import curry, curry_flip

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def getvalue(obj: T1, key: str) -> T2:
    if hasattr(obj, "__getitem__"):
        get_ = getitem
    else:
        get_ = getattr

    ret = get_(obj, key)
    return ret


def setvalue(obj: T1, key: str, value: T2) -> None:
    if hasattr(obj, "__setitem__"):
        set_ = setitem
    else:
        set_ = setattr

    set_(obj, key, value)


@curry_flip(1)
def change_value_by_key(data: T1, key: T2, fn: Callable[[T3], T3]) -> T1:
    setvalue(data, key, fn(getvalue(data, key)))
    return data


KeyType = TypeVar("KeyType")
ValueType = TypeVar("ValueType")
TransformedType = TypeVar("TransformedType")


@runtime_checkable
class MappingLike(Protocol[KeyType, ValueType]):  # type: ignore
    def __getitem__(self, key: KeyType, /) -> ValueType: ...
    def __len__(self) -> int: ...


class LazyMap(Generic[KeyType, ValueType, TransformedType]):
    def __init__(
        self,
        obj: MappingLike[KeyType, ValueType],
        fn_value: Callable[[ValueType], TransformedType] | None = None,
        fn_key_value: Callable[[KeyType, ValueType], TransformedType] | None = None,
    ):
        self.obj = obj
        self.fn_value = fn_value
        self.fn_key_value = fn_key_value

    def __getitem__(self, key: KeyType) -> TransformedType:
        if self.fn_key_value:
            return self.fn_key_value(key, self.obj[key])
        elif self.fn_value:
            return self.fn_value(self.obj[key])
        else:
            raise RuntimeError("No transformation function provided.")

    def __len__(self) -> int:
        return len(self.obj)


ValueTransform = Callable[[ValueType], TransformedType]
KeyValueTransform = Callable[[KeyType, ValueType], TransformedType]


def lazy_map(
    fn: ValueTransform,
) -> Callable[[MappingLike[KeyType, ValueType]], LazyMap[KeyType, ValueType, TransformedType]]:
    def apply_transformation(obj: MappingLike[KeyType, ValueType]) -> LazyMap[KeyType, ValueType, TransformedType]:
        return LazyMap(obj, fn_value=fn)

    return apply_transformation


def lazy_map2(
    fn: KeyValueTransform,
) -> Callable[[MappingLike[KeyType, ValueType]], LazyMap[KeyType, ValueType, TransformedType]]:
    def apply_transformation(obj: MappingLike[KeyType, ValueType]) -> LazyMap[KeyType, ValueType, TransformedType]:
        return LazyMap(obj, fn_key_value=fn)

    return apply_transformation


class _SizeLimiterMaping(MappingLike[KeyType, ValueType]):
    def __init__(self, collection: MappingLike, size: int):
        # Use object.__setattr__ to avoid triggering __getattr__
        object.__setattr__(self, "col", collection)
        object.__setattr__(self, "size", min(len(collection), size))

    def __getitem__(self, key: KeyType) -> ValueType:
        # Delegate item access to the underlying collection
        return self.col[key]

    def __len__(self) -> int:
        # Return the precomputed limited size
        return object.__getattribute__(self, "size")

    def __getattr__(self, a):
        # Fallback to accessing attributes from the underlying collection
        # Use object.__getattribute__ to prevent recursion
        col = object.__getattribute__(self, "col")
        return getattr(col, a)


class _SizeLimiterIterable(Generic[ValueType]):
    def __init__(self, collection: Iterator, size: int):
        self.col = collection
        self.size = size

    def __iter__(self) -> Iterator[ValueType]:
        for i, v in enumerate(self.col):
            if i < self.size:
                yield v
            else:
                return

    def __getattr__(self, a):
        return getattr(self.col, a)


ColectionT = MappingLike[KeyType, ValueType] | Iterable[ValueType]


@curry_flip(1)
def limit_size(collection: ColectionT, size: int) -> ColectionT:
    match collection:
        case MappingLike():
            return _SizeLimiterMaping(collection, size)
        case collections.abc.Iterable():
            return _SizeLimiterIterable(collection, size)
        case _:
            raise ValueError


@curry_flip(1)
def parallel_map(it: Iterable[T1], fn: Callable[[T1], T2]):
    """
    Parallel map, usefull to call GIL free functions, such as C-API
    """
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fn, i) for i in it]
        for f in futures:
            yield f.result()


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


def cache_once(input_foo):
    ret = None

    def foo(*args, **kw):
        nonlocal ret
        if ret is None:
            ret = input_foo(*args, **kw)

        return ret

    return foo


T = TypeVar("T")


@curry(1)
def prefetch_iter(buffer_size: int, iterable: Iterable[T]) -> Iterator[T]:
    """Returns an iterator that prefetches items from the given iterable into a queue."""
    queue: Queue[T | None] = Queue(maxsize=buffer_size)

    def producer():
        for item in iterable:
            queue.put(item)
        queue.put(None)  # Sentinel to indicate completion

    t = Thread(target=producer, daemon=True)
    t.start()

    while (item := queue.get()) is not None:
        yield item
