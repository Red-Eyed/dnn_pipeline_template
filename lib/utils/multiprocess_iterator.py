#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import multiprocessing as mp
from threading import Thread
from typing import Any, Callable, Iterable, Iterator

from lib.utils.serialization import make_pickable


class MultiprocessIterator:
    def __init__(
        self,
        process_fn: Callable[[Any], Any],
        iterable: Iterable,
        num_workers: int = 8,
        queue_size: int = 100,
    ):
        self.iterable = iter(iterable)
        self.process_fn = make_pickable(process_fn)
        self.num_workers = num_workers
        self.task_queue = mp.Queue(maxsize=queue_size)
        self.result_queue = mp.Queue(maxsize=queue_size)
        self.processes = []
        self.task_queue_finished = False

    def worker(self):
        for item in iter(self.task_queue.get, None):
            self.result_queue.put(self.process_fn(item))

    def feeder(self):
        """Feeds tasks to the queue dynamically."""
        for item in self.iterable:
            self.task_queue.put(item)

        # send stop signal to workers
        for i in range(self.num_workers):
            self.task_queue.put(None)

        self.task_queue_finished = True

    def __iter__(self) -> Iterator:
        # Start worker processes
        for _ in range(self.num_workers):
            p = mp.Process(target=self.worker)
            p.start()
            self.processes.append(p)

        # Start feeder process
        feeder_thread = Thread(target=self.feeder)
        feeder_thread.start()

        while True:
            if self.task_queue_finished and self.task_queue.empty() and self.result_queue.empty():
                break

            result = self.result_queue.get()
            yield result

        feeder_thread.join()

        for p in self.processes:
            p.join()
