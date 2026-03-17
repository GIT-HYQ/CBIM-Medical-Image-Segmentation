# -*- coding: utf-8 -*-
"""
3c validation wrapper:
- Keep original training/validation.py unchanged
- Fix 2D input layout for multi-channel input (e.g. C=2 prior mode)
"""

from typing import Any
import torch
from training import validation as _v
import cv2
import os
import numpy as np

def _fix_2d_input_layout(inputs: torch.Tensor, args) -> torch.Tensor:
    """
    Feed legacy layout to original validation.py:
    original code will do inputs = inputs.permute(1,0,2,3)
    so we pre-convert BCHW -> CBHW here.
    """
    if getattr(args, "dimension", None) != "2d":
        return inputs

    if not torch.is_tensor(inputs):
        return inputs

    if inputs.dim() != 4:
        raise RuntimeError(f"[validation_3c] Unexpected 2D input dim: {tuple(inputs.shape)}")

    in_chan = int(getattr(args, "in_chan", 1))

    # If already BCHW (B,C,H,W), convert to legacy CBHW for old validation.py
    if inputs.shape[1] == in_chan:
        return inputs.permute(1, 0, 2, 3)

    # If already legacy (C,B,H,W), keep it
    if inputs.shape[0] == in_chan:
        return inputs

    raise RuntimeError(
        f"[validation_3c] Cannot infer input layout: shape={tuple(inputs.shape)}, in_chan={in_chan}"
    )


def _fix_batch(batch: Any, args):
    """
    Expected batch forms in this repo:
    - train/val common: (inputs, targets, ...)
    """
    if isinstance(batch, (list, tuple)) and len(batch) >= 1:
        batch = list(batch)
        batch[0] = _fix_2d_input_layout(batch[0], args)
        return tuple(batch)
    return batch


class _FixedLoader:
    def __init__(self, loader, args):
        self._loader = loader
        self._args = args

    def __iter__(self):
        for batch in self._loader:
            yield _fix_batch(batch, self._args)

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        # passthrough: dataset, sampler, batch_size, etc.
        return getattr(self._loader, name)

def save_images2(img, msk, msk_pred, name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 保留第一个通道，训练时叠加了一个通道的分割掩码先验，所以原图的第一个通道才是我们需要保存的内容
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()[:, :, 0] * 255
    img = img.astype(np.uint8)
    msk = msk.permute(1,2,0).detach().cpu().numpy()
    msk = scale_image_max(msk)
    msk_pred = msk_pred.permute(1,2,0).detach().cpu().numpy()
    msk_pred = scale_image_max(msk_pred)
    image_path = os.path.join(save_path, name.replace('.png', '_src.png'))
    mask_path = os.path.join(save_path, name.replace('.png', '_mask.png'))
    pred_path = os.path.join(save_path, name.replace('.png', '_pred.png'))
    cv2.imwrite(image_path, img)
    cv2.imwrite(mask_path, msk)
    cv2.imwrite(pred_path, msk_pred)

def validation(net, valLoader, args, **kwargs):
    fixed_loader = _FixedLoader(valLoader, args)
    return _v.validation(net, fixed_loader, args, **kwargs)


def validation_without_calc(net, valLoader, args, **kwargs):
    if not hasattr(_v, "validation_without_calc"):
        raise AttributeError("[validation_3c] training.validation has no validation_without_calc")
    fixed_loader = _FixedLoader(valLoader, args)
    return _v.validation_without_calc(net, fixed_loader, args, **kwargs)


def validation_ddp(net, valLoader, args, **kwargs):
    if not hasattr(_v, "validation_ddp"):
        raise AttributeError("[validation_3c] training.validation has no validation_ddp")
    fixed_loader = _FixedLoader(valLoader, args)
    return _v.validation_ddp(net, fixed_loader, args, **kwargs)


# Optional re-export helpers for compatibility
if hasattr(_v, "scale_image_max"):
    scale_image_max = _v.scale_image_max
# if hasattr(_v, "save_images2"):
#     save_images2 = _v.save_images2
    _v.save_images2 = save_images2
if hasattr(_v, "save_images"):
    save_images = _v.save_images