#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import argparse
from typing import List, Literal, Optional, get_args, get_origin

from pydantic import BaseModel


class SortedHelpFormatter(argparse.HelpFormatter):
    def add_arguments(self, actions: List[argparse.Action]) -> None:
        actions = sorted(actions, key=lambda action: action.option_strings[0] if action.option_strings else action.dest)
        super().add_arguments(actions)

    def _format_action_invocation(self, action: argparse.Action) -> str:
        parts = super()._format_action_invocation(action).split(", ")
        return "\n".join(parts)

    def _format_action(self, action: argparse.Action) -> str:
        parts = super()._format_action(action)
        if action.choices:
            choices_str = ", ".join(map(str, action.choices))
            parts += f" (choices: {choices_str})"
        return parts


def build_parser_from_pydantic(
    model: BaseModel,
    parser: Optional[argparse.ArgumentParser] = None,
    prefix: str = "",
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(formatter_class=SortedHelpFormatter)

    for name, field in model.model_fields.items():
        if isinstance(field, BaseModel):
            parser = build_parser_from_pydantic(field, parser, "name.")
            continue

        kwargs = {}
        if field.default is not None:
            type_ = type(field.default)
            kwargs["type"] = type_
        else:
            type_ = None

        if get_origin(field.annotation) is Literal:
            kwargs["choices"] = get_args(field.annotation)
            kwargs["type"] = str
        elif type_ in (list, tuple):
            kwargs["nargs"] = "+"
        elif type_ is bool:
            del kwargs["type"]
            if field.default:
                kwargs["action"] = "store_false"
            else:
                kwargs["action"] = "store_true"

        parser.add_argument(
            f"--{prefix}{name}",
            help="default: %(default)s",
            default=field.default,
            **kwargs,
        )

    return parser
