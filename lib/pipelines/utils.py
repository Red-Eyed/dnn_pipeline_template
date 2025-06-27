#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

import logging
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def get_summary_writer(trainer: Trainer) -> SummaryWriter | None:
    match trainer.logger:
        case TensorBoardLogger() if isinstance(trainer.logger.experiment, SummaryWriter):
            return trainer.logger.experiment
        case _:
            logger.warning("Can't match tensorboard logger, embedding logging is skipped")
            return None
