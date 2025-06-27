#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import logging
from argparse import ArgumentParser
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from lib.config.project_config import ConfigFS


class BasePipelineConfig(BaseModel, validate_assignment=True):
    log_level: int = logging.INFO
    accumulate_grad_batches: int = 1
    precision: Literal["16-mixed", "bf16-mixed", "32-true", "64-true", "bf16"] = "32-true"
    seed: int = 0
    cudnn_benchmark: bool = True
    strategy: Literal["auto", "fsdp", "ddp"] = "auto"
    gradient_clip_algorithm: Literal["norm", "value"] = "norm"
    gradient_clip_value: float = 1.0
    accelerator: Literal["cpu", "cuda"] = "cuda"
    summary_depth: int = 2
    check_val_every_n_epoch: int = 1
    num_sanity_val_steps: int = 4
    use_lightning_env_plugin: bool = False

    # Set it in the Runner
    base_dir: Path = ConfigFS.ROOT_DIR / "work_dir"

    experiment: str = "default"

    @cached_property
    def experiment_dir(self) -> Path:
        return self.base_dir / self.experiment

    @cached_property
    def export_dir(self) -> Path:
        return self.experiment_dir / "export"

    @cached_property
    def checkpoints_dir(self) -> Path:
        return self.experiment_dir / "checkpoints"

    clear_experiment_dir: bool = False
    checkpoint: Optional[Path] = None
    profiler: Optional[Literal["simple", "advanced", "pytorch"]] = None
    onnx_opset_version: int = 16
    fast_dev_run: bool = False
    torch_compile: bool = False
    detect_anomaly: bool = False
    max_epoches: int = -1
    overfit_batches: int = 0
    action: Literal["train", "eval", "export", "benchmark_dataloader"] = "eval"
    multiprocess_start_method: Literal["fork", "forkserver", "spawn"] = "fork"
    resume: bool = False
    with_visualization: bool = False

    @property
    def visualization_dir(self) -> Path:
        return self.experiment_dir / "visualization"

    @staticmethod
    def parse_args():
        parser = ArgumentParser()
        parser.add_argument("--load_config", type=Path, default=None)
        parser.add_argument("--save_config", type=Path, default=None)

        args = parser.parse_args()

        return parser, args
