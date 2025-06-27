#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import json
from functools import cache
from pathlib import PosixPath, WindowsPath
from typing import Any, Iterable, Type

from pydantic import BaseModel


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np

        if isinstance(obj, np.number):
            return float(obj)
        elif isinstance(obj, PosixPath) or isinstance(obj, WindowsPath):
            return str(obj.as_posix())
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            return json.JSONEncoder.default(self, obj)


def get_params(params):
    if params:
        ret = {}
        for k, v in params.items():
            if k != "self" and not k.startswith("_"):
                ret[k] = v

        return ret
    return params


@cache
def get_cached_session(name):
    import requests_cache

    session = requests_cache.CachedSession(name, backend="filesystem", use_cache_dir=True)
    return session


def get_cached_ticker(t: str):
    import yfinance as yf

    session = get_cached_session("yfinance_cache")
    ticker = yf.Ticker(t, session=session)
    return ticker


def get_property_name(pydantic_obj: BaseModel, pydantic_obj_property):
    for field in pydantic_obj.model_fields:
        if getattr(pydantic_obj, field) == pydantic_obj_property:
            return field
    raise ValueError("Property not found.")


def update_model_inplace(model: BaseModel, updates: dict[str, Any]) -> None:
    def apply_updates(obj: BaseModel, update_dict: dict[str, Any]) -> None:
        for key, value in update_dict.items():
            if isinstance(value, dict) and hasattr(obj, key):
                # Recursive update for nested objects
                nested_obj = getattr(obj, key)
                apply_updates(nested_obj, value)
            else:
                setattr(obj, key, value)

    apply_updates(model, updates)


def extract_model_fields(model: Type[BaseModel]) -> Iterable[tuple[str, Any]]:
    return model.__annotations__.items()


def is_debug():
    import sys

    gettrace = getattr(sys, "gettrace", None)

    if gettrace is None:
        return False
    else:
        v = gettrace()
        if v is None:
            return False
        else:
            return True
