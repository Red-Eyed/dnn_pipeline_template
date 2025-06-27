#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

from typing import Any, Callable

import dill


class _Base:
    @staticmethod
    def _dill_loads(*args) -> bytes:
        return dill.loads(args[0])


class SerializableMixIn(_Base):
    def __reduce__(self):
        cls = self.__class__
        state = self.__getstate__() if hasattr(self, "__getstate__") else self.__dict__
        # serialize class and state only (avoid recursion on self)
        return self._dill_loads, (dill.dumps((cls, state)),)

    @staticmethod
    def _dill_loads(serialized: bytes):
        cls, state = dill.loads(serialized)
        obj = cls.__new__(cls)
        if hasattr(obj, "__setstate__"):
            obj.__setstate__(state)
        else:
            obj.__dict__.update(state)
        return obj


class _SerializableObject(_Base):
    def __init__(self, obj: Any):
        self.obj = obj

    def __reduce__(self):
        serialized = dill.dumps(self.obj)
        return self._dill_loads, (serialized, 0)


class _SerializableCallable(_Base):
    def __init__(self, fn: Callable):
        self.fn = fn

    def __reduce__(self):
        serialized = dill.dumps(self.fn)
        return self._dill_loads, (serialized, 0)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.fn(*args, **kwds)


def serialize_obj(obj: Any) -> _SerializableObject:
    return _SerializableObject(obj)


def serialize_callable(callable: Any) -> _SerializableCallable:
    return _SerializableCallable(callable)


def make_pickable(obj: Any):
    if callable(obj):
        return serialize_callable(obj)
    else:
        return serialize_obj(obj)
