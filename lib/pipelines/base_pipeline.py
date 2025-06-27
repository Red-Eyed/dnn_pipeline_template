#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import logging
import multiprocessing
import os
import pprint
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

import torch
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from tqdm import tqdm

from lib.config.base_cfg import BasePipelineConfig
from lib.config.base_data_config import BaseDataConfig
from lib.config.project_config import ConfigFS, config_save
from lib.data.datamodule import DataModule
from lib.modeling.base_model import BaseLightningModule
from lib.pipelines.utils import get_summary_writer
from lib.pipelines.common import build_default_callbacks, build_default_trainer
from lib.utils.log_utils import set_logger
from lib.utils.utils import is_debug

logger = logging.getLogger(__name__)


def format_hparams_md(hparams: dict[str, float]) -> str:
    header = "| Hyperparameter | Value |\n|---|---|\n"
    rows = [f"| {k} | {v} |" for k, v in hparams.items()]
    return header + "\n".join(rows)


class IPipeline(ABC):
    @abstractmethod
    def build_model(self, **kw) -> BaseLightningModule: ...

    @abstractmethod
    def build_datamodule(self) -> DataModule: ...

    @abstractmethod
    def build_trainer(self) -> Trainer: ...

    @abstractmethod
    def handle_action(self, datamodule: DataModule, model: BaseLightningModule, trainer: Trainer, **kw) -> bool: ...

    @abstractmethod
    def run(self): ...


class DefaultPipelineConfig(
    BasePipelineConfig,
    BaseDataConfig,
):
    @property
    def strict_model_load(self) -> bool:
        return self.action == "eval"


class BasePipeline(IPipeline):
    def __init__(self, cfg: DefaultPipelineConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.setup_env()

    def build_callbacks(self) -> list[Callback]:
        return build_default_callbacks(self.cfg)

    @classmethod
    @abstractmethod
    def get_model_class(cls) -> Type[BaseLightningModule]: ...

    def build_model(self, **kw):
        klass = self.get_model_class()
        model = klass.build_from_config(self.cfg, **kw)

        if self.cfg.checkpoint and self.cfg.checkpoint.exists():
            model = klass.load_from_checkpoint(
                checkpoint_path=self.cfg.checkpoint,
                strict=self.cfg.strict_model_load,
                map_location="cpu",
                **model.cfg.model_dump(mode="python"),
                **kw,
            )
            logger.info(f"Checkpoint is loaded: {self.cfg.checkpoint}")
        else:
            logger.info("Checkpoint is not loaded")

        if self.cfg.torch_compile:
            model = torch.compile(model)

        return model

    def build_trainer(self) -> Trainer:
        tensorboard = TensorBoardLogger(
            save_dir=self.cfg.experiment_dir,
            name="tensorboard_events",
            log_graph=False,
            default_hp_metric=False,
        )

        trainer = build_default_trainer(
            self.cfg,
            self.build_callbacks(),
            tensorboard,
        )

        return trainer

    def export(self, trainer: Trainer, model: BaseLightningModule, export_dir: Path):
        shutil.rmtree(export_dir, ignore_errors=True)
        export_dir.mkdir(parents=True)

        model_name = model._get_name()
        dst_model_checkpoint = export_dir / f"{model_name}.ckpt"

        # Hack to save checkpoint with hparams
        # https://lightning.ai/forums/t/saving-a-lightningmodule-without-a-trainer/2217/3
        trainer.strategy.connect(model)
        trainer.save_checkpoint(dst_model_checkpoint, weights_only=True)
        logger.info(f"Exported model checkpoint to: {dst_model_checkpoint}")

        exported_config_path = export_dir / f"{model_name}_config.json"
        config_save(self.cfg, exported_config_path)
        logger.info(f"Exported model config to: {exported_config_path}")

        model_onnx_path = export_dir / f"{model_name}.onnx"
        try:
            model.to_onnx(model_onnx_path, opset_version=self.cfg.onnx_opset_version)
        except NotImplementedError:
            logger.info(f"{model.__class__.__name__}.to_onnx(...) is not implemented. Skipping")
        else:
            logger.info(f"Exported model to onnx: {export_dir}")

    def benchmark_dataloader(self, dl):
        try:
            for _i in tqdm(dl):
                pass
        except KeyboardInterrupt:
            pass

    def setup_env(self):
        cfg = self.cfg

        cfg.base_dir.mkdir(exist_ok=True, parents=True)
        cfg.experiment_dir.mkdir(exist_ok=True, parents=True)

        os.chdir(cfg.base_dir)

        seed_everything(cfg.seed, workers=False)

        log_level = cfg.log_level

        if is_debug():
            logger.info("Running in the DEBUG mode, changing some configuration")
            log_level = logging.DEBUG

            cfg = cfg.model_copy(
                update=dict(
                    accelerator="cpu",
                    num_sanity_val_steps=1,
                    num_workers=0,
                    precision="32-true",
                    multiprocess_start_method="fork",
                )
            )

            self.cfg = cfg

        multiprocessing.set_start_method(cfg.multiprocess_start_method, force=True)

        set_logger(
            info_file=cfg.experiment_dir / "train_info.log",
            error_file=cfg.experiment_dir / "train_error.log",
            min_level=log_level,
            filter_path=ConfigFS.ROOT_DIR,
        )

        logger.info("### CONFIGURATION ###")
        logger.info(pprint.pformat(cfg.model_dump()))

        if cfg.clear_experiment_dir:
            logger.info(f"removing '{cfg.experiment_dir}'")
            shutil.rmtree(cfg.experiment_dir, ignore_errors=True)
            cfg.experiment_dir.mkdir(parents=True)

        config_save(self.cfg, self.cfg.experiment_dir / f"{self.cfg.experiment}_config.json")

    def handle_action(self, datamodule: DataModule, model: BaseLightningModule, trainer: Trainer, **kw):
        match self.cfg.action:
            case "export":
                self.export(trainer, model, self.cfg.experiment_dir / "export")
            case "train":
                ckpt_path = None
                if self.cfg.resume:
                    ckpt_path = self.cfg.checkpoint
                trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            case "eval":
                dataloader = getattr(datamodule, f"{self.cfg.eval_split}_dataloader")()
                trainer.test(model=model, dataloaders=dataloader)
            case "benchmark_dataloader":
                self.benchmark_dataloader(datamodule.train_dataloader())
            case _:
                return False

        return True

    def _log_config(self, trainer: Trainer):
        writer = get_summary_writer(trainer)
        if writer:
            writer.add_text("Config", format_hparams_md(self.cfg.model_dump()), global_step=0)

    def run(self):
        datamodule = self.build_datamodule()

        def len_f(x):
            return tqdm.format_sizeof(len(x))

        logger.info(f"Dataset train size = {len_f(datamodule.train_dataset)}")
        logger.info(f"Dataset test size = {len_f(datamodule.test_dataset)}")
        logger.info(f"Dataset val size = {len_f(datamodule.val_dataset)}")

        model = self.build_model()
        trainer = self.build_trainer()

        self._log_config(trainer)

        ret = self.handle_action(
            datamodule=datamodule,
            model=model,
            trainer=trainer,
        )

        if ret is False:
            logger.warning("Nothing is done")
