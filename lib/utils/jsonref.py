#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import json
from pathlib import Path
from typing import Any, Callable

T = dict | list | str


class JSONRef:
    def __init__(self, refs: dict):
        self.refs = dict(sorted(refs.items(), key=lambda kv: -len(kv[1])))

    def _var2ref(self, value):
        for var, ref in self.refs.items():
            value = value.replace(var, ref)
        return value

    def _ref2var(self, value):
        for var, ref in self.refs.items():
            value = value.replace(ref, var)
        return value

    def _apply(self, data: T, fn: Callable) -> T:
        def foo(value: Any):
            if isinstance(value, dict):
                return {k: foo(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [foo(v) for v in value]
            elif isinstance(value, str):
                return fn(value)
            return value

        return foo(data)

    def resolve(self, data: T) -> T:
        """Replaces placeholders in JSON with actual values."""

        return self._apply(data, self._var2ref)

    def restore(self, data: T) -> T:
        """Restores placeholders from absolute values back to references."""

        return self._apply(data, self._ref2var)

    def save_json(self, filepath: str | Path, data: dict):
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        Path(filepath).write_text(json.dumps(data, indent=4))

    def load_json(self, filepath: str | Path):
        return json.loads(Path(filepath).read_text())
