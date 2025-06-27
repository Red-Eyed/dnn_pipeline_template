#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import json
from pathlib import Path
from typing import Any

from expression import pipe
from expression.collections import seq
from pydantic import BaseModel, model_serializer

from lib.utils.argparse_utils import build_parser_from_pydantic


def autocast(d: dict) -> dict:
    for k, v in d.items():
        if isinstance(v, str):
            try:
                d[k] = float(v)
                d[k] = int(v)
            except (TypeError, ValueError):
                pass

    return d


def from_cmd(model: BaseModel):
    parser = build_parser_from_pydantic(model)

    parser_kwargs = parser.parse_args().__dict__
    parser_kwargs = autocast(parser_kwargs)
    config = type(model)(**parser_kwargs)
    return config


def load(obj, cmd=True):
    config = obj.from_cmd() if cmd else obj

    # load from json, but parameters from argparse have highest priority
    if config.load_config_path:
        kwargs_from_file = json.loads(config.load_config_path.read_text())
        kwargs_from_file.update(config.model_dump())
        config = type(obj)(**kwargs_from_file)

    return config


@model_serializer(when_used="json")
def sort_model(obj) -> dict[str, Any]:
    def not_path(p):
        return not isinstance(p[1], Path)

    ret = pipe(
        obj.model_dump().items(),
        seq.filter(not_path),
        sorted,
        dict,
    )
    return ret
