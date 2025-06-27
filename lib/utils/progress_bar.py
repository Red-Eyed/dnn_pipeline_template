#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

from queue import Queue
from threading import Thread
from typing import Iterable

import tqdm


class _CustomDict(dict):
    def __init__(self, factory):
        self.factory = factory

    def __missing__(self, key):
        self[key] = self.factory(key)
        return self[key]


class ProgressBar:
    def __init__(self):
        self._progressq = Queue(-1)
        self._tread = Thread(target=self._wait_with_progress)

    def stop(self):
        self._progressq.put(("", None))

    def capture_progress(self, description: str):
        def foo(iter: Iterable) -> Iterable:
            for i, data in enumerate(iter):
                self._progressq.put((description, i))
                yield data

            self._progressq.put((description, None))

        return foo

    def _wait_with_progress(self):
        def tqdm_init(desc):
            t = tqdm.tqdm(
                desc=desc,
                unit_scale=True,
            )
            t.write("")
            return t

        tqdms = _CustomDict(tqdm_init)

        while True:
            data = self._progressq.get()
            if data is None:
                break

            description, i = data
            if i is None:
                tqdms[description].set_postfix_str("Done ✓")
                tqdms[description].close()
                del tqdms[description]
            else:
                tqdms[description].update()

            if len(tqdms) == 0:
                return

    def __enter__(self):
        self._tread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.stop()
            self._tread.join()
            raise exc_val.with_traceback(exc_tb)
        else:
            self._tread.join()
