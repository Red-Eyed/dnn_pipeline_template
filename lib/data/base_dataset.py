#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

from typing import Iterator, TypeVar

from torch.utils.data import Dataset

from lib.utils.functional import MappingLike

SampleType = TypeVar("SampleType")


T_co = TypeVar("T_co", covariant=True)


class GenericDataset(Dataset[T_co]):
    def __init__(self, obj: MappingLike) -> None:
        super().__init__()
        self.obj = obj

    def __len__(self) -> int:
        return len(self.obj)

    def __getitem__(self, item: int) -> T_co:
        if item >= len(self):
            raise StopIteration
        return self.obj[item]

    def __iter__(self) -> Iterator[T_co]:
        for i in range(len(self)):
            yield self[i]


class DatasetProto(MappingLike[int, SampleType]):
    pass


def to_dataset(dataset: DatasetProto[SampleType]) -> Dataset[SampleType]:
    return GenericDataset(dataset)
