#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

from typing import Callable, Literal

import lightning as L
import torch
from expression import compose
from lightning.fabric.utilities.seed import pl_worker_init_function
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import default_collate, DataLoader

from lib.data.base_dataset import DatasetProto, to_dataset
from lib.data.dataloader import NodesDataLoader
from lib.utils.dataset_wrappers import PickableDataset, SafeDataset
from lib.utils.serialization import serialize_callable


def worker_init(*args, **kw):
    # set seed for each worker
    pl_worker_init_function(*args, **kw)


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        train: DatasetProto,
        val: DatasetProto,
        test: DatasetProto,
        batch_size=32,
        num_workers=8,
        prefetch_factor=4,
        collate_fn: Callable = default_collate,
        pin_memory=True,
        transfer_batch_to_device_fn: Callable | None = None,
        mp_start_method: Literal["spawn", "fork", "forkserver"] = "fork",
        method: Literal["process", "thread"] = "thread",
        **kw,
    ):
        super().__init__()

        self.method = method
        self.mp_start_method = mp_start_method
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.prefetch_factor = max(1, prefetch_factor)
        self.collate_fn = serialize_callable(collate_fn)
        self.batch_size = batch_size

        if transfer_batch_to_device_fn is None:
            transfer_batch_to_device_fn = super().transfer_batch_to_device
        self.transfer_batch_to_device_fn = transfer_batch_to_device_fn

        make_dataset = compose(to_dataset, SafeDataset, PickableDataset)  # type: ignore

        self._train = make_dataset(train)
        self._test = make_dataset(test)
        self._val = make_dataset(val)

    def get_example_batch(self, batch_size: int | None = None):
        sample = self.train_dataset[0]
        batch_size = self.batch_size if batch_size is None else batch_size
        batch = self.collate_fn([sample] * batch_size)
        return batch

    def build_dataloader(self, *args, **kw):
        return self._build_dataloader_old(*args, **kw)

    def _build_dataloader_new(self, dataset, shuffle=False, drop_last=False):
        dl = NodesDataLoader(
            batch_size=self.batch_size,
            dataset=dataset,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=worker_init,
            shuffle=shuffle,
            drop_last=drop_last,
            multiprocessing_context=None if self.num_workers == 0 else self.mp_start_method,
            persistent_workers=True,
            method="process",
            prebatch=self.num_workers,
        )
        return dl

    def _build_dataloader_old(self, dataset, shuffle=False, drop_last=False):
        dl = DataLoader(
            batch_size=self.batch_size,
            dataset=dataset,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            worker_init_fn=worker_init,
            shuffle=shuffle,
            drop_last=drop_last,
            multiprocessing_context=None if self.num_workers == 0 else self.mp_start_method,
            persistent_workers=True if self.num_workers > 0 else False,
        )
        return dl

    def transfer_batch_to_device(
        self, batch: EVAL_DATALOADERS, device: torch.device, dataloader_idx: int
    ) -> EVAL_DATALOADERS:
        return self.transfer_batch_to_device_fn(batch, device, dataloader_idx)

    @property
    def train_dataset(self):
        return self._train

    @property
    def test_dataset(self):
        return self._test

    @property
    def val_dataset(self):
        return self._val

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.build_dataloader(dataset=self.train_dataset, shuffle=True, drop_last=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.build_dataloader(dataset=self.test_dataset, shuffle=False, drop_last=False)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.build_dataloader(dataset=self.val_dataset, shuffle=False, drop_last=False)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()
