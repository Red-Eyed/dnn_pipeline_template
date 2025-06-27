#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import logging
import sys
from pathlib import Path


class LevelFilter(logging.Filter):
    def __init__(self, low, high, filter_path: str):
        self._low = low
        self._high = high
        self._filter_path = Path(filter_path).expanduser().resolve()

        logging.Filter.__init__(self)

    def filter(self, record: logging.LogRecord):
        path = Path(record.pathname)
        path_cond = not path.is_absolute() or self._filter_path.as_posix() in path.as_posix()
        level_cond = self._low <= record.levelno <= self._high
        if level_cond and path_cond:
            return True
        return False


def set_logger(error_file: Path, info_file: Path, min_level, filter_path: Path):
    filter_path = Path(filter_path).expanduser().resolve()
    error_file = Path(error_file).expanduser().resolve()
    error_file.parent.mkdir(exist_ok=True, parents=True)

    info_file = Path(info_file).expanduser().resolve()
    info_file.parent.mkdir(exist_ok=True, parents=True)

    file_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(module)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(module)s]: %(message)s", datefmt="%b, %d %H:%M:%S"
    )

    error_file_handler = logging.FileHandler(filename=error_file.as_posix())
    error_file_handler.addFilter(LevelFilter(logging.WARNING, logging.CRITICAL, filter_path.as_posix()))
    error_file_handler.setFormatter(file_formatter)

    info_file_handler = logging.FileHandler(filename=info_file.as_posix())
    info_file_handler.addFilter(LevelFilter(min_level, logging.CRITICAL, filter_path.as_posix()))
    info_file_handler.setFormatter(file_formatter)

    info_stream_handler = logging.StreamHandler(sys.stdout)
    info_stream_handler.addFilter(LevelFilter(min_level, logging.CRITICAL, filter_path.as_posix()))
    info_stream_handler.setFormatter(stream_formatter)

    logging.basicConfig(
        handlers=[
            info_file_handler,
            error_file_handler,
            info_stream_handler,
        ],
        level=min_level,
        force=True,
    )
