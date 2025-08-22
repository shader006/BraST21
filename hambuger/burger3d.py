# -*- coding: utf-8 -*-
"""
Wrapper Hamburger3D_V1
- Chuẩn hoá ham_args (dict/object)
- Ép MD_D = in_c nếu không đặt hoặc <= 0
- Kiểm tra MD_S chia hết C
- In lại cấu hình cuối để tránh nhầm (chỉ rank0)
"""
from __future__ import annotations
from types import SimpleNamespace
from typing import Union

import torch
import torch.nn as nn

from .ham3d import NMF3D, _get_rank_from_args


def _to_namespace(x: Union[dict, object, None]) -> SimpleNamespace:
    if x is None:
        return SimpleNamespace()
    if isinstance(x, dict):
        return SimpleNamespace(**x)
    # đã là object có attribute
    return x  # type: ignore[return-value]


class Hamburger3D_V1(nn.Module):
    def __init__(self, in_c: int, ham_args=None):
        super().__init__()
        self.in_c = int(in_c)
        h = _to_namespace(ham_args)

        # --- điền mặc định & ép kiểu ---
        # Loại ham
        h.HAM_TYPE = getattr(h, "HAM_TYPE", "NMF3D")
        # splitting & spatial
        h.MD_S     = int(getattr(h, "MD_S", 1))
        h.SPATIAL  = bool(getattr(h, "SPATIAL", True))
        # rank & D
        h.MD_R     = int(getattr(h, "MD_R", 64))
        md_d_in    = int(getattr(h, "MD_D", 0))
        h.MD_D     = self.in_c if md_d_in <= 0 else md_d_in  # ép về in_c nếu không đặt
        # các step & ổn định
        h.TRAIN_STEPS = int(getattr(h, "TRAIN_STEPS", 6))
        h.EVAL_STEPS  = int(getattr(h, "EVAL_STEPS", 7))
        h.ETA         = float(getattr(h, "ETA", 0.9))
        h.EPS         = float(getattr(h, "EPS", 1e-6))
        h.RAND_INIT   = bool(getattr(h, "RAND_INIT", True))
        # amp/dtype
        h.COMPUTE_FP32        = bool(getattr(h, "COMPUTE_FP32", True))
        h.CLAMP_INPUT_NONNEG  = bool(getattr(h, "CLAMP_INPUT_NONNEG", True))
        h.NORM_EACH_STEP      = bool(getattr(h, "NORM_EACH_STEP", True))
        h.EARLY_STOP_TOL      = float(getattr(h, "EARLY_STOP_TOL", 1e-4))
        # log
        h.rank = int(getattr(h, "rank", 0))

        # --- kiểm tra hợp lệ ---
        if self.in_c % h.MD_S != 0:
            raise ValueError(f"MD_S={h.MD_S} phải chia hết cho C={self.in_c} (C % S == 0).")
        if h.MD_R <= 0:
            raise ValueError("MD_R (rank) phải > 0.")
        if h.MD_D <= 0:
            raise ValueError("MD_D phải > 0 (đã được ép về in_c nếu không đặt).")

        # --- chọn biến thể ham ---
        ham_type = str(h.HAM_TYPE).upper()
        if ham_type == "NMF3D":
            self.ham = NMF3D(h)
        else:
            raise ValueError(f"HAM_TYPE '{ham_type}' chưa được hỗ trợ.")

        # (tuỳ chọn) pre/post conv nếu muốn “sandwich”
        self.pre  = nn.Identity()
        self.post = nn.Identity()

        # --- in cấu hình cuối ---
        if _get_rank_from_args(h, 0) == 0:
            print(f"[3D MD FINAL] S={h.MD_S} D_cfg={h.MD_D} R={h.MD_R} SPATIAL={h.SPATIAL} "
                  f"fp32MU={h.COMPUTE_FP32} steps={h.TRAIN_STEPS}/{h.EVAL_STEPS}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, D, H, W)
        """
        y = self.pre(x)
        y = self.ham(y)
        y = self.post(y)
        return y
