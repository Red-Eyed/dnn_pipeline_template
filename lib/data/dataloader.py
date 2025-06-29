#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

from typing import List, Callable, Literal
import torchdata.nodes as tn
from torch.utils.data import RandomSampler, SequentialSampler, default_collate, Dataset


class MapAndCollate:
    """A simple transform that takes a batch of indices, maps with dataset, and then applies
    collate.
    TODO: make this a standard utility in torchdata.nodes
    """

    def __init__(self, dataset, collate_fn):
        self.dataset = dataset
        self.collate_fn = collate_fn

    def __call__(self, batch_of_indices: List[int]):
        batch = [self.dataset[i] for i in batch_of_indices]
        return self.collate_fn(batch)


# To keep things simple, let's assume that the following args are provided by the caller
def NodesDataLoader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 1,
    collate_fn: Callable = default_collate,
    pin_memory: bool = True,
    drop_last: bool = True,
    method: Literal["process", "thread"] = "thread",
    multiprocessing_context: str | None = None,
    prefetch_factor: int = 1,
    prebatch: int = 1,
    **kw,
):
    # Assume we're working with a map-style dataset
    assert hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__")
    # Start with a sampler, since caller did not provide one
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    # Sampler wrapper converts a Sampler to a BaseNode
    node = tn.SamplerWrapper(sampler)

    # Now let's batch sampler indices together
    node = tn.Batcher(node, batch_size=batch_size, drop_last=drop_last)

    # Create a Map Function that accepts a list of indices, applies getitem to it, and
    # then collates them
    map_and_collate = MapAndCollate(dataset, collate_fn)

    # MapAndCollate is doing most of the heavy lifting, so let's parallelize it. We could
    # choose process or thread workers. Note that if you're not using Free-Threaded
    # Python (eg 3.13t) with -Xgil=0, then multi-threading might result in GIL contention,
    # and slow down training.
    node = tn.ParallelMapper(
        node,
        map_fn=map_and_collate,
        num_workers=num_workers,
        method=method,
        in_order=True,
        multiprocessing_context=multiprocessing_context,
        prebatch=prebatch,
    )

    # Optionally apply pin-memory, and we usually do some pre-fetching
    if pin_memory:
        node = tn.PinMemory(node)

    if prefetch_factor > 1:
        node = tn.Prefetcher(node, prefetch_factor=prefetch_factor)

    # Note that node is an iterator, and once it's exhausted, you'll need to call .reset()
    # on it to start a new Epoch.
    # Insteaad, we wrap the node in a Loader, which is an iterable and handles reset. It
    # also provides state_dict and load_state_dict methods.
    return tn.Loader(node)
