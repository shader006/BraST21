# -*- coding: utf-8 -*-
"""
3D Hamburger (Matrix Decomposition) for PyTorch
- NMF3D với multiplicative updates (MU) trực tiếp trên tensor 5D (B, C, D, H, W)

Tối ưu & An toàn:
- MU chạy ở fp32 (tùy chọn) trong context tắt autocast (API mới torch.amp.autocast)
- Giảm copy bộ nhớ: bỏ .contiguous() không cần thiết; dùng .reshape thay .view
- Tối ưu traffic: tiền tính Xt = x^T, tái sử dụng Gram BtB, CtC mỗi bước
- Ổn định: clamp không âm + chuẩn hóa L2 cho bases theo dim đặc trưng
- Early stopping (chuẩn Frobenius)
- Bật TF32 cho bmm/matmul (GPU Ampere+) nếu sẵn sàng
"""
from __future__ import annotations
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp


def _get_rank_from_args(a: Union[dict, object], default: int = 0) -> int:
    if isinstance(a, dict):
        return int(a.get("rank", default))
    return int(getattr(a, "rank", default))


class _MatrixDecomposition3DBase(nn.Module):
    """
    Base class cho các phép phân rã ma trận 3D/5D.
    Lớp con cần hiện thực:
      - _build_bases(self, B, S, D, R, device=None, dtype=None)
      - local_step(self, x, bases, coef, *args, **kwargs)
      - compute_coef(self, x, bases, coef)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        if isinstance(args, dict):
            self.spatial     = bool(args.get("SPATIAL", True))
            self.S           = int(args.get("MD_S", 1))
            self.D_cfg       = int(args.get("MD_D", 512))
            self.R           = int(args.get("MD_R", 64))
            self.train_steps = int(args.get("TRAIN_STEPS", 6))
            self.eval_steps  = int(args.get("EVAL_STEPS", 7))
            self.inv_t       = float(args.get("INV_T", 100))
            self.eta         = float(args.get("ETA", 0.9))
            self.rand_init   = bool(args.get("RAND_INIT", True))
        else:
            self.spatial     = bool(getattr(args, "SPATIAL", True))
            self.S           = int(getattr(args, "MD_S", 1))
            self.D_cfg       = int(getattr(args, "MD_D", 512))
            self.R           = int(getattr(args, "MD_R", 64))
            self.train_steps = int(getattr(args, "TRAIN_STEPS", 6))
            self.eval_steps  = int(getattr(args, "EVAL_STEPS", 7))
            self.inv_t       = float(getattr(args, "INV_T", 100))
            self.eta         = float(getattr(args, "ETA", 0.9))
            self.rand_init   = bool(getattr(args, "RAND_INIT", True))

        self._rank0_print(f"[3D MD] spatial={self.spatial}")
        self._rank0_print(f"[3D MD] S={self.S} D_cfg={self.D_cfg} R={self.R}")
        self._rank0_print(f"[3D MD] steps train/eval = {self.train_steps}/{self.eval_steps}")

    # --------- helpers ----------
    def _rank0_print(self, *a, **k):
        if _get_rank_from_args(self.args, default=0) == 0:
            print(*a, **k)

    # --------- abstract hooks ----------
    def _build_bases(self, B, S, D, R, device=None, dtype=None):
        raise NotImplementedError

    def local_step(self, x, bases, coef, *args, **kwargs):
        raise NotImplementedError

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        """
        Mặc định: init coef bằng softmax(inv_t * X^T B) rồi lặp local_step.
        Lớp con (NMF3D) override để thêm Xt & early-stop.
        """
        coef = torch.bmm(x.transpose(1, 2), bases)  # (B*S, N, R)
        coef = F.softmax(self.inv_t * coef, dim=-1)
        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)
        return bases, coef

    def forward(self, x5d: torch.Tensor, return_bases: bool = False):
        """
        x5d: (B, C, D, H, W) -> trả về x_hat cùng shape
        """
        assert x5d.dim() == 5, f"Expect 5D (B,C,D,H,W), got {tuple(x5d.shape)}"
        B, C, Dv, H, W = x5d.shape
        device, dtype = x5d.device, x5d.dtype
        assert C % self.S == 0, f"Channels C={C} must be divisible by S={self.S}"

        # Sắp xếp (không dùng .contiguous() trừ khi thật sự cần)
        if self.spatial:
            D_feat = C // self.S
            N = Dv * H * W
            x = x5d.reshape(B, self.S, D_feat, Dv, H, W).reshape(B * self.S, D_feat, N)
        else:
            D_feat = Dv * H * W
            N = C // self.S
            x = (
                x5d.reshape(B, self.S, C // self.S, Dv, H, W)
                    .reshape(B * self.S, N, D_feat)
                    .transpose(1, 2)  # (B*S, D_feat, N)
            )

        # Cached bases (eval/online update)
        if (not self.rand_init) and (not hasattr(self, "bases")):
            bases0 = self._build_bases(1, self.S, D_feat, self.R, device=device, dtype=dtype)
            # shape (S, D_feat, R)
            self.register_buffer("bases", bases0)

        # Khởi tạo bases cho batch hiện tại
        if self.rand_init:
            bases = self._build_bases(B, self.S, D_feat, self.R, device=device, dtype=dtype)  # (B*S, D_feat, R)
        else:
            bases = (
                self.bases.unsqueeze(0)  # (1, S, D_feat, R)
                .expand(B, -1, -1, -1)  # (B, S, D_feat, R)
                .reshape(B * self.S, D_feat, self.R)
            )

        # Suy luận local (no grad), rồi tinh chỉnh coef có grad
        bases, coef = self.local_inference(x, bases)
        coef = self.compute_coef(x, bases, coef)  # (B*S, N, R)

        # Tái tạo và đưa về 5D
        x_hat = torch.bmm(bases, coef.transpose(1, 2))  # (B*S, D_feat, N)
        if self.spatial:
            x_hat = x_hat.reshape(B, self.S, C // self.S, Dv, H, W).reshape(B, C, Dv, H, W)
        else:
            x_hat = (
                x_hat.transpose(1, 2)
                     .reshape(B, self.S, C // self.S, Dv, H, W)
                     .reshape(B, C, Dv, H, W)
            )

        # Online EMA update cho bases (eval & không trả bases)
        if (not self.rand_init) and (not self.training) and (not return_bases):
            self.online_update(bases.reshape(B, self.S, D_feat, self.R))

        return x_hat

    @torch.no_grad()
    def online_update(self, bases_b: torch.Tensor):
        """
        EMA update self.bases theo (B, S, D_feat, R) -> (S, D_feat, R)
        """
        update = bases_b.mean(dim=0)  # (S, D_feat, R)
        self.bases += self.eta * (update - self.bases)
        self.bases = F.normalize(self.bases, dim=1)


class NMF3D(_MatrixDecomposition3DBase):
    """
    NMF với Multiplicative Updates (MU) cho 3D features.
    - MU ở fp32 (tùy chọn) trong context tắt autocast (API mới torch.amp.autocast)
    - Không dùng .contiguous() khi không cần; bmm hỗ trợ non-contiguous
    - Tránh einsum để không chạm lỗi TorchDynamo, dùng bmm chuẩn
    """
    def __init__(self, args):
        super().__init__(args)

        # Các tuỳ chọn ổn định & hiệu năng
        if isinstance(args, dict):
            self.compute_fp32        = bool(args.get("COMPUTE_FP32", True))  # FP32 cho MU để ổn định với AMP
            self.eps                 = float(args.get("EPS", 1e-6))
            self.clamp_input_nonneg  = bool(args.get("CLAMP_INPUT_NONNEG", True))
            self.normalize_each_step = bool(args.get("NORM_EACH_STEP", True))
            self.early_stop_tol      = float(args.get("EARLY_STOP_TOL", 0.0))  # 0 = tắt
            self.inv_t               = float(args.get("INV_T", 1.0))
        else:
            self.compute_fp32        = bool(getattr(args, "COMPUTE_FP32", True))
            self.eps                 = float(getattr(args, "EPS", 1e-6))
            self.clamp_input_nonneg  = bool(getattr(args, "CLAMP_INPUT_NONNEG", True))
            self.normalize_each_step = bool(getattr(args, "NORM_EACH_STEP", True))
            self.early_stop_tol      = float(getattr(args, "EARLY_STOP_TOL", 0.0))
            self.inv_t               = float(getattr(args, "INV_T", 1.0))

        # Bật TF32 (nếu có)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ---------- helpers ----------
    def _maybe_fp32(self, t: torch.Tensor) -> torch.Tensor:
        return t.float() if self.compute_fp32 else t

    def _autocast_ctx(self, ref: torch.Tensor):
        # Nếu muốn MU chạy fp32: tắt autocast. Ngược lại giữ nguyên.
        device_type = "cuda" if ref.is_cuda else "cpu"
        return amp.autocast(device_type=device_type, enabled=(not self.compute_fp32))

    def _build_bases(self, B, S, D, R, device=None, dtype=None):
        device = torch.device("cpu") if device is None else device
        dtype_eff = torch.float32 if self.compute_fp32 else (dtype or torch.float32)
        bases = torch.rand((B * S, D, R), device=device, dtype=dtype_eff)
        bases = F.normalize(bases, dim=1)
        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef, *, Xt: Optional[torch.Tensor] = None):
        """
        Một bước MU:
          C <- C * (X^T B) / (C (B^T B))
          B <- B * (X C)   / (B (C^T C))

        x:     (B*S, D, N)
        bases: (B*S, D, R)
        coef:  (B*S, N, R)
        Xt:    (B*S, N, D) (tuỳ chọn)
        """
        with self._autocast_ctx(x):
            if self.clamp_input_nonneg:
                x = x.clamp_min_(self.eps)

            x_ = self._maybe_fp32(x)
            b_ = self._maybe_fp32(bases)
            c_ = self._maybe_fp32(coef)

            Xt_ = Xt if Xt is not None else x_.transpose(1, 2)   # (B*S, N, D)
            Bt_ = b_.transpose(1, 2)                              # (B*S, R, D)

            # --- cập nhật C ---
            BtB = torch.bmm(Bt_, b_)                              # (B*S, R, R)
            numC = torch.bmm(Xt_, b_)                             # (B*S, N, R)
            denC = torch.bmm(c_, BtB).add_(self.eps)              # (B*S, N, R)
            c_.mul_(numC).div_(denC)

            # --- cập nhật B ---
            Ct  = c_.transpose(1, 2)                              # (B*S, R, N)
            CtC = torch.bmm(Ct, c_)                               # (B*S, R, R)
            numB = torch.bmm(x_, c_)                              # (B*S, D, R)
            denB = torch.bmm(b_, CtC).add_(self.eps)              # (B*S, D, R)
            b_.mul_(numB).div_(denB)

            # Không âm & chuẩn hoá
            b_.clamp_(min=self.eps)
            c_.clamp_(min=self.eps)
            if self.normalize_each_step:
                b_ = F.normalize(b_, p=2, dim=1)

            # Ghi về dtype ban đầu
            bases.copy_(b_.to(bases.dtype))
            coef.copy_(c_.to(coef.dtype))

        return bases, coef

    def compute_coef(self, x, bases, coef):
        """
        Tinh chỉnh hệ số C cuối cùng (có grad)
        """
        if self.clamp_input_nonneg:
            x = x.clamp_min(self.eps)

        with self._autocast_ctx(x):
            x_ = self._maybe_fp32(x)
            b_ = self._maybe_fp32(bases)
            c_ = self._maybe_fp32(coef)

            Xt_ = x_.transpose(1, 2)   # (B*S, N, D)
            Bt_ = b_.transpose(1, 2)   # (B*S, R, D)
            BtB = torch.bmm(Bt_, b_)   # (B*S, R, R)
            numC = torch.bmm(Xt_, b_)  # (B*S, N, R)
            denC = torch.bmm(c_, BtB).add_(self.eps)  # (B*S, N, R)
            c_new = c_ * (numC / denC)

        return c_new.to(coef.dtype)

    @torch.no_grad()
    def local_inference(self, x, bases):
        """
        Init coef = softmax(inv_t * X^T B); lặp MU với Xt tiền tính.
        Early stop theo ||B_t - B_{t-1}||_F / ||B_{t-1}||_F < tol
        """
        coef = torch.bmm(x.transpose(1, 2), bases)  # (B*S, N, R)
        coef = F.softmax(self.inv_t * coef, dim=-1).to(bases.dtype)

        steps = self.train_steps if self.training else self.eval_steps
        Xt = x.transpose(1, 2)  # (B*S, N, D)

        prev_b = None
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef, Xt=Xt)
            if self.early_stop_tol > 0:
                if prev_b is not None:
                    diff = (bases - prev_b).norm(dim=(1, 2))
                    ref  = prev_b.norm(dim=(1, 2)).clamp_min(self.eps)
                    if torch.all((diff / ref) < self.early_stop_tol):
                        break
                prev_b = bases.detach().clone()

        return bases, coef


# alias cũ (nếu code khác import NMFND)
NMFND = NMF3D
