#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import json
from functools import singledispatch
from pathlib import Path

from pydantic import BaseModel

from lib.utils.jsonref import JSONRef


def find_root(path: Path) -> Path:
    if path.is_dir() and {"data", "lib"}.issubset(p.name for p in path.iterdir()):
        return path
    else:
        return find_root(path.parent)


class ConfigFS:
    ROOT_DIR: Path = find_root(Path(__file__))
    HOME_DIR: Path = Path.home()
    CONFIGS_DIR: Path = ROOT_DIR / "cfgs"
    DATA_DIR: Path = ROOT_DIR / "data"
    CMAKE_OUT_DIR: Path = ROOT_DIR / "bin"
    CMAKE_BUILD_DIR: Path = ROOT_DIR / "build"


def _build_json_ref():
    refs = {
        "${PROJECT_DIR}": ConfigFS.ROOT_DIR.resolve().as_posix(),
        "${HOME_DIR}": ConfigFS.HOME_DIR.resolve().as_posix(),
    }

    jsonref = JSONRef(refs)

    return jsonref


jsonref = _build_json_ref()


@singledispatch
def config_save(obj: dict, path: str | Path):
    obj = jsonref.restore(obj)  # type: ignore
    jsonref.save_json(path, obj)


@config_save.register
def _(obj: BaseModel, path: str | Path):
    json_ = obj.model_dump_json()
    obj = json.loads(json_)
    config_save(obj, path)


def config_load(path: str | Path) -> dict:
    obj = jsonref.load_json(path)
    obj = jsonref.resolve(obj)
    return obj  # type: ignore
