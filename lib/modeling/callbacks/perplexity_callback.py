#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import torch

from typing import Any
from torchmetrics.text import Perplexity
from lib.modeling.callbacks.base_metric_callback import BaseMetricsCallback


class PerplexityCallback(BaseMetricsCallback):
    def __init__(self, pad_idx: int = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_idx = pad_idx

    def build_metric(self):
        return Perplexity(ignore_index=self.pad_idx)

    def get_target(self, batch: Any):
        nested = torch.nested.as_nested_tensor(batch["tokens"], layout=torch.jagged)
        padded = nested.to_padded_tensor(self.pad_idx).to(torch.long)
        return padded

    def get_preds(self, outputs: Any):
        return outputs["logits"]
