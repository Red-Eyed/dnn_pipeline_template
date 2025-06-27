#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:author:     Vadym Stupakov <vadim.stupakov@gmail.com>
:license:    MIT
"""

from expression import pipe
from lightning.pytorch import Callback, Trainer, LightningModule
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict

from lib.pipelines.utils import get_summary_writer
import numpy as np
import cv2


def normalize_attention(attn: torch.Tensor) -> torch.Tensor:
    attn_min, attn_max = attn.min(), attn.max()
    return (attn - attn_min) / (attn_max - attn_min) if attn_max > attn_min else attn


def apply_colormap(tensor: torch.Tensor, colormap: int = cv2.COLORMAP_JET):
    assert tensor.ndim == 3 and tensor.shape[0] == 1, "Tensor must have shape (1, seq_len, seq_len)"

    tensor_np = tensor.squeeze(0).cpu().numpy()
    tensor_uint8 = (tensor_np * 255).astype(np.uint8)

    colored_image = cv2.applyColorMap(tensor_uint8, colormap)
    colored_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)

    return colored_image


class AttentionMapVisualizationCallback(Callback):
    def __init__(self):
        self.attention_maps: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def _register_hooks(self, model: LightningModule):
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.MultiheadAttention):
                self.hooks.append(layer.register_forward_pre_hook(self._hook_fn(name), with_kwargs=True))

    def _hook_fn(self, layer_name: str):
        def hook(module: torch.nn.MultiheadAttention, *args):
            args, kw = args[0], args[1]
            kw["need_weights"] = True
            _, attn_output_weights = module.forward(*args, **kw)
            if attn_output_weights is not None:
                self.attention_maps[layer_name] = attn_output_weights

            return None

        return hook

    def _log_attention_maps(self, writer: SummaryWriter, global_step: int):
        for layer_name, attn_layer in self.attention_maps.items():
            if not isinstance(attn_layer, torch.Tensor):
                continue

            batch_size, l1, l2 = attn_layer.shape
            with writer as w:
                for i in range(batch_size):
                    attn_map = attn_layer[i].unsqueeze(0)  # Shape: (1, seq_len, seq_len)
                    colormap = pipe(
                        attn_map,
                        normalize_attention,
                        lambda x: apply_colormap(x, cv2.COLORMAP_TURBO),
                    )
                    try:
                        w.add_image(
                            f"attention/{layer_name}",
                            colormap,
                            global_step=global_step,
                            dataformats="HWC",
                        )
                    except TypeError:
                        continue

        self.attention_maps.clear()

    def on_predict_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        self._register_hooks(pl_module)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        writer = get_summary_writer(trainer)
        if writer:
            self._log_attention_maps(writer, batch_idx)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        for hook in self.hooks:
            hook.remove()
