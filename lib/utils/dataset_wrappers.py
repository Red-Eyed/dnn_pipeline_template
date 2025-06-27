#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import logging
import traceback
from copy import deepcopy
from typing import Any


from lib.data.base_dataset import GenericDataset
from lib.utils.functional import MappingLike
from lib.utils.serialization import SerializableMixIn

logger = logging.getLogger(__name__)


class SafeDataset(GenericDataset):
    def __init__(self, dataset: MappingLike, num_attempts=10) -> None:
        super().__init__(dataset)
        self.num_attmpts = num_attempts

    def __getitem__(self, index, attempt_n=0) -> Any:
        try:
            data = self.obj[index]
            attempt_n = 0
            return data
        except StopIteration:
            raise
        except Exception:
            if attempt_n >= self.num_attmpts:
                raise

            logger.warning(f"Invalid sample[{index}], skipping. Traceback:\n {traceback.format_exc()}")
            return self.__getitem__(index + 1 % len(self), attempt_n=attempt_n + 1)


class FakeDataset(GenericDataset):
    def __init__(self, dataset: MappingLike) -> None:
        super().__init__(dataset)
        self._sample = deepcopy(self.obj[0])
        self._obj = None

    def __getitem__(self, index) -> Any:
        return self._sample

    def __iter__(self):
        while True:
            yield self._sample


class PickableDataset(GenericDataset, SerializableMixIn):
    pass
