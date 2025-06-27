#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

from abc import abstractmethod, ABC
from itertools import product
import logging

import lightning.pytorch as pl

from typing import Dict, Any, Literal
from collections import defaultdict
from lightning.pytorch import Callback
from lightning.pytorch.trainer.states import RunningStage
import torch
from torchmetrics import Metric

logger = logging.getLogger(__name__)

StagesT = set[Literal["predict", "test", "validation", "train", "sanity_check"]]


class BaseMetricsCallback(ABC, Callback):
    def __init__(
        self,
        metric_name,
        stages: StagesT,
        every_n_epochs=1,
    ):
        super().__init__()

        self.metric_name = metric_name
        self.device = "cpu"

        self._map_metric: Dict[str, Metric] = defaultdict(self.build_metric)
        self._every_n_epochs = every_n_epochs

        self._set_callbacks(stages)

    def _generate_callback_names(self):
        template = "on_{stage}_{batch_epoch}_{start_end}"
        stages = "predict test validation train sanity_check".split(" ")
        batch_epoch = "batch epoch".split(" ")
        start_end = "start end".split()

        callback_names = [
            template.format(stage=s, batch_epoch=be, start_end=se)
            for s, be, se in product(stages, batch_epoch, start_end)
        ]

        return callback_names

    def _set_callbacks(self, stages: StagesT):
        def is_in(set_: set[str], str_: str):
            return any(i in str_ for i in set_)

        callbacks = self._generate_callback_names()
        for c in callbacks:
            if not is_in(stages, c) or "start" in c:  # type: ignore
                continue

            if "batch" in c:
                callback = self._on_batch_end
            elif "epoch" in c:
                callback = self._on_epoch_end

            setattr(self, c, callback)
            logger.info(f"Callback '{c}' is set for metric {self.metric_name}")

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage) -> None:
        self.device = pl_module.device

        # move to device
        for s in RunningStage:
            self._map_metric[s] = self._map_metric[s].to(device=self.device)

    @abstractmethod
    def build_metric(self) -> Metric:
        """
        Builds and returns class for calculating metric.
        """

    @abstractmethod
    def get_target(self, batch: Any):
        """
        Preprocesses input batch to make required representation for metrics calculation.

        Args:
            batch: input that model receives on forward step;
                may be dict, list, torch.Tensor or other depending on your pipeline

        Returns:
            Preprocessed batch that will be passed to metrics calculation as "target" parameter.
        """

    @abstractmethod
    def get_preds(self, outputs: Any):
        """
        Preprocesses outputs of the model so that it is ready for metrics calculation.

        Args:
            outputs: output of the model

        Returns:
            Preprocessed model output ready for metrics calculation as "preds" parameter.
        """

    @torch.no_grad()
    def _on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        *args,
        **kw,
    ) -> None:
        stage = trainer.state.stage
        if stage is None:
            return

        self._map_metric[stage].update(preds=self.get_preds(outputs), target=self.get_target(batch))
        result = self._map_metric[stage].compute()

        pl_module.log(
            f"{stage.value}_{self.metric_name}_batch",
            result,
            on_step=True,
            prog_bar=True,
            on_epoch=False,
        )

    @torch.no_grad()
    def _on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        stage = trainer.state.stage
        if stage is None:
            return

        # Check if we should report on this epoch
        if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
            result = self._map_metric[stage].compute()
            pl_module.log(
                f"{stage.value}_{self.metric_name}_epoch",
                result,
                on_epoch=True,
                prog_bar=True,
            )

        self._map_metric[stage].reset()
