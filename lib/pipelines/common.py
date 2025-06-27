#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import OrderedDict

import lightning.pytorch.callbacks as C
import torch
import torch.utils
import torch.utils.data
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.plugins.environments import LightningEnvironment
from torch import Tensor

from lib.config.base_cfg import BasePipelineConfig
from lib.modeling.base_model import BaseLightningModule

logger = logging.getLogger(__name__)


def build_default_callbacks(cfg: BasePipelineConfig) -> list[C.Callback]:
    saver_params = dict(
        dirpath=cfg.checkpoints_dir,
        verbose=True,
        mode="min",
        save_top_k=3,
        save_last=True,
        enable_version_counter=False,
        every_n_epochs=1,
    )

    def get_monitor_and_filename(monitor):
        return dict(monitor=monitor, filename=f"{{epoch:04d}}-{{{monitor}:3.3f}}")

    callbacks = [
        C.RichModelSummary(cfg.summary_depth),
        C.RichProgressBar(leave=True),
        C.ModelCheckpoint(**saver_params | get_monitor_and_filename("loss_train_epoch")),
        C.ModelCheckpoint(**saver_params | get_monitor_and_filename("loss_validation_epoch")),
        C.ModelCheckpoint(
            **saver_params
            | {
                "every_n_train_steps": None,
                "every_n_epochs": None,
                "train_time_interval": timedelta(minutes=1),
                "save_top_k": False,
            }  # type: ignore
        ),
        C.LearningRateMonitor("step", True, True),
    ]

    return callbacks


def build_default_trainer(
    config: BasePipelineConfig,
    callbacks: list[Callback],
    train_logger: Logger,
):
    plugins = None

    if config.use_lightning_env_plugin:
        plugins = [LightningEnvironment()]

    trainer = Trainer(
        default_root_dir=config.base_dir.as_posix(),
        logger=train_logger,
        max_epochs=config.max_epoches,
        benchmark=config.cudnn_benchmark,
        accelerator=config.accelerator,
        strategy=config.strategy,
        sync_batchnorm=False,
        callbacks=callbacks,
        precision=config.precision,
        fast_dev_run=config.fast_dev_run,
        detect_anomaly=config.detect_anomaly,
        enable_model_summary=False,
        reload_dataloaders_every_n_epochs=0,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        overfit_batches=config.overfit_batches,
        num_sanity_val_steps=config.num_sanity_val_steps,
        accumulate_grad_batches=config.accumulate_grad_batches,
        profiler=config.profiler,
        gradient_clip_algorithm=config.gradient_clip_algorithm,
        gradient_clip_val=config.gradient_clip_value,
        plugins=plugins,
    )

    return trainer


def load_model_non_strict(model: BaseLightningModule, state_dict: dict | OrderedDict) -> BaseLightningModule:
    model_state_dict = model.state_dict()

    to_be_loaded_state_dict = {}
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                logger.info(
                    f"Skip loading parameter: {k} with shape: {state_dict[k].shape} "
                    f"required shape is: {model_state_dict[k].shape}"
                )
                to_be_loaded_state_dict[k] = model_state_dict[k]
            else:
                to_be_loaded_state_dict[k] = state_dict[k]

    model.load_state_dict(to_be_loaded_state_dict, strict=False)
    return model


def load_model(model: BaseLightningModule, checkpoint_path: Path | None, strict: bool = False) -> BaseLightningModule:
    if checkpoint_path and checkpoint_path.exists():
        pass
    else:
        logger.warning(f"Skip loading. Checkpoint doesn't exist: '{checkpoint_path}'")
        return model

    # add forward to initialize nn.Lazy*() layers
    match model.example_input_array:
        case dict():
            model.forward(**model.example_input_array)
        case Tensor():
            model.forward(model.example_input_array)
        case tuple():
            model.forward(*model.example_input_array)
        case _:
            model.forward()
            logger.warning("example_input_array is None")

    checkpoint = torch.load(checkpoint_path.as_posix(), map_location="cpu", weights_only=True)
    checkpoint_state_dict = checkpoint["state_dict"]

    if strict:
        model.load_state_dict(checkpoint_state_dict, strict=True)
    else:
        model = load_model_non_strict(model, checkpoint_state_dict)

    logger.info(f"Loaded checkpoint with {strict=}: {checkpoint_path}")

    return model


def export_checkpoint(src_checkpoint: Path, dst_checkpoint: Path):
    ckpt = torch.load(src_checkpoint.as_posix())
    state_dict_ckpt = {"state_dict": ckpt["state_dict"]}
    torch.save(state_dict_ckpt, dst_checkpoint)
    logger.info(f"Exported: {src_checkpoint} -> {dst_checkpoint}")
