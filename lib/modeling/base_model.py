#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import logging
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Literal, Mapping, Self, TypeVar

import dill
import lightning as L
import torch
from pydantic import BaseModel, ConfigDict, computed_field
from torch import Tensor
from torchmetrics import Metric

from lib.config.project_config import ConfigFS

logger = logging.getLogger(__name__)


class BaseModelConfig(BaseModel):
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=True,
    )

    @computed_field
    @property
    def name(self) -> str:
        return self.__class__.__name__


BaseModelConfigT = TypeVar("BaseModelConfigT", bound=BaseModelConfig)


class BaseLightningModule(ABC, L.LightningModule, Generic[BaseModelConfigT]):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.cfg = self.build_default_cfg().model_copy(update=kwargs)
        self._example_input_array = None

    @property
    def example_input_array(self):
        if self._example_input_array is None:
            try:
                self._example_input_array = self.build_example_input_array()
            except Exception:
                # All exceptions, except SystemExit are suppressed and example_input_array attribute is not created
                # throwing SystemExit with traceback to exit and print stracktrace from self.build_example_input_array()
                raise SystemExit(traceback.format_exc())

        return self._example_input_array

    @example_input_array.setter
    def example_input_array(self, example: Tensor | tuple | dict | None) -> None:
        self._example_input_array = example

    @classmethod
    @abstractmethod
    def build_default_cfg(cls) -> BaseModelConfigT: ...

    @classmethod
    def build_from_config(cls, cfg: BaseModelConfigT, **kw):
        return cls(**cfg.model_dump(mode="python"), **kw)

    @classmethod
    def build_default(cls, **kw):
        cfg = cls.build_default_cfg()
        obj = cls.build_from_config(cfg, **kw)
        return obj

    @abstractmethod
    def build_example_input_array(self) -> Tensor | dict[str, Tensor]: ...

    @abstractmethod
    def forward(self, *args, **kw) -> Any: ...

    @abstractmethod
    def process_step(self, *args, **kw) -> Mapping[str, Tensor | Metric]:
        """Should return dict with required 'loss' and optionally any other key to include to log"""
        ...

    @abstractmethod
    def predict_step(self, *args, **kw) -> Any: ...

    def to_onnx(self, file_path: Path, opset_version=18) -> None:
        raise NotImplementedError

    @classmethod
    def load_from_checkpoint(  # type: ignore
        cls,
        checkpoint_path: Path,
        map_location: torch.device | str = "cpu",
        strict: bool = True,
        **kw,
    ) -> Self:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
            pickle_module=dill,
        )
        hparams = checkpoint["hyper_parameters"]
        hparams |= kw
        model = cls(**hparams)
        state_dict = load_chekpoint(model, checkpoint, strict=strict)
        model.load_state_dict(state_dict, strict=strict)
        logger.info(
            f"Loaded checkpoint '{checkpoint_path.relative_to(ConfigFS.ROOT_DIR)}' for model: '{model.__class__.__name__}'"
        )
        return model

    def on_load_checkpoint(self, checkpoint: dict):
        return load_chekpoint(self, checkpoint)

    def _process_step_and_log(self, stage: Literal["train", "test", "validation"], batch):
        assert stage in ("train", "validation", "test")

        returns = self.process_step(**batch)
        assert "loss" in returns

        # in case we have dynamic batch size
        batch_size = returns.get("batch_size", None)

        to_log = {f"{k}_{stage}": v for k, v in returns.items() if k.startswith("loss")}

        self.log_dict(
            to_log,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        ret = returns

        return ret

    def training_step(self, batch, batch_idx):
        return self._process_step_and_log("train", batch)

    def validation_step(self, batch, batch_idx):
        return self._process_step_and_log("validation", batch)

    def test_step(self, batch, **kw):
        return self._process_step_and_log("test", batch)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            cooldown=50,
            patience=50,
            min_lr=1e-6,
            threshold_mode="rel",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "loss_train_epoch",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }


def load_chekpoint(model: BaseLightningModule, checkpoint: dict, strict=True):
    checkpoint_state_dict = checkpoint["state_dict"]

    # add forward to initialize nn.Lazy*() layers
    match model.example_input_array:
        case dict():
            model.forward(**model.example_input_array)
        case Tensor():
            model.forward(model.example_input_array)
        case tuple():
            model.forward(*model.example_input_array)
        case _:
            logger.warning("example_input_array is None")
            model.forward()

    model_state_dict = model.state_dict()

    is_changed = False
    for k in checkpoint_state_dict:
        if k in model_state_dict:
            if checkpoint_state_dict[k].shape != model_state_dict[k].shape:
                if (
                    "embedding" in k
                    and model_state_dict[k].shape[0] > checkpoint_state_dict[k].shape[0]
                    and model_state_dict[k].shape[1] == checkpoint_state_dict[k].shape[1]
                ):
                    model_state_dict[k][: checkpoint_state_dict[k].shape[0]] = checkpoint_state_dict[k]
                    logger.info(
                        f"Load embeddings with different shape: {k} with shape: {checkpoint_state_dict[k].shape} "
                        f"required shape is: {model_state_dict[k].shape}"
                    )
                else:
                    logger.info(
                        f"Skip loading parameter: {k} with shape: {checkpoint_state_dict[k].shape} "
                        f"required shape is: {model_state_dict[k].shape}"
                    )
                is_changed = True
            else:
                model_state_dict[k] = checkpoint_state_dict[k]
        else:
            logger.info(f"Parameter {k} from checkpoint is not present in model. Dropping")
            is_changed = True

    if is_changed:
        logger.info("Skip loading optimizer")
        checkpoint.pop("optimizer_states", None)

        checkpoint.pop("lr_scheduler", None)
        logger.info("Skip loading lr_scheduler")

    if is_changed and strict:
        raise RuntimeError("Model does not match checkpoint")

    return model_state_dict
