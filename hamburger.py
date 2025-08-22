# -*- coding: utf-8 -*-
"""
Hamburger V2Plus+ (2D & 3D) — clean, paper-accurate, self-contained.

Features:
- Ham core = Soft-VQ (cosine + softmax + codebook update) OR NMF (MU updates)
- One-step gradient (K-1 steps no_grad, last step with grad)
- Dual ham: spatial + channel (like V2+)
- Cheese bottleneck (1x1 conv → BN → ReLU)
- Learnable mixing: coef_ham (zero-init opt) & coef_shortcut
- BN(H(Z)) BEFORE skip-add (per paper Eq.6)
- Careful contiguity before view/reshape; epsilon guards; dtype/device safe

Author: you
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


# =========================
# Config
# =========================
@dataclass
class HamburgerConfig:
    # common
    in_channels: int
    d_mid: int = 512          # W_l projection channels (paper's d)
    r: int = 64               # rank / codebook size
    steps: int = 3            # inner optimization steps
    ham_mode: str = "vq"      # 'vq' or 'nmf'
    temperature: float = 10.0 # softmax temperature for vq
    groups_S: int = 1         # channel groups for flatten
    dual: bool = True         # use spatial+channel dual hams
    cheese_factor: int = 1    # bottleneck factor (>=1); 1 disables bottleneck
    zero_ham: bool = True     # init coef_ham to 0 for safe start
    # norms/activations
    use_relu_in_lower: bool = True  # safer for NMF; ok for VQ too
    bn_momentum: float = 0.1        # BN momentum


# =========================
# Ham Core (X: (B*, d, n))
# =========================
class HamMD(nn.Module):
    """
    Matrix Decomposition core:
      - mode='vq'  : Soft VQ (cosine + softmax), updates D with XC^T diag(sum)^{-1}
      - mode='nmf' : Nonnegative MF (multiplicative updates for C & D)
    One-step gradient: K-1 steps in no_grad, final step with grad.
    Returns Xbar = D @ C (shape: (B*, d, n))
    """
    def __init__(self, d: int, r: int, steps: int = 3,
                 mode: str = "vq", temperature: float = 10.0):
        super().__init__()
        assert mode in ("vq", "nmf")
        self.mode = mode
        self.r = int(r)
        self.steps = int(steps)
        self.T = float(temperature)
        self.d = int(d)

    @torch.no_grad()
    def _init_D(self, B_, device, dtype):
        D = torch.rand(B_, self.d, self.r, device=device, dtype=dtype)
        if self.mode == "nmf":
            D.clamp_(min=EPS)
        return F.normalize(D, dim=1, eps=EPS)  # normalize along d

    @torch.no_grad()
    def _init_C(self, B_, n, device, dtype):
        C = torch.rand(B_, self.r, n, device=device, dtype=dtype).clamp_(min=EPS)
        C /= (C.sum(dim=1, keepdim=True) + EPS)  # column-wise normalize (optional)
        return C

    # ---- VQ step ----
    def _vq_step(self, X, D):
        # X: (B, d, n), D: (B, d, r)
        Dn = F.normalize(D, dim=1, eps=EPS)
        Xn = F.normalize(X, dim=1, eps=EPS)
        cos = torch.bmm(Dn.transpose(1, 2), Xn)      # (B, r, n)
        C = F.softmax(cos / max(self.T, EPS), dim=1) # soft assignment; sum over r == 1 for each column

        XCt = torch.bmm(X, C.transpose(1, 2))        # (B, d, r)
        sums = C.sum(dim=2).clamp_min(EPS)           # (B, r)
        D = XCt / sums.unsqueeze(1)                  # normalize columns
        D = F.normalize(D, dim=1, eps=EPS)
        Xbar = torch.bmm(D, C)                       # (B, d, n)
        return D, C, Xbar

    # ---- NMF (MU) step ----
    def _nmf_step(self, X_pos, D, C):
        # X_pos: >=0 ; D,C >= 0
        DtX  = torch.bmm(D.transpose(1, 2), X_pos).clamp_min(EPS)       # (B, r, n)
        DtDC = torch.bmm(D.transpose(1, 2), torch.bmm(D, C)).clamp_min(EPS)
        C = (C * (DtX / DtDC)).clamp_min(EPS)

        XCt  = torch.bmm(X_pos, C.transpose(1, 2)).clamp_min(EPS)       # (B, d, r)
        DCCt = torch.bmm(D, torch.bmm(C, C.transpose(1, 2))).clamp_min(EPS)
        D = (D * (XCt / DCCt)).clamp_min(EPS)

        D = F.normalize(D, dim=1, eps=EPS)  # stabilize
        Xbar = torch.bmm(D, C)
        return D, C, Xbar

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B*, d, n)
        """
        B_, d, n = X.shape
        assert d == self.d, f"d mismatch: got {d}, expected {self.d}"
        device, dtype = X.device, X.dtype

        D = self._init_D(B_, device, dtype)
        if self.mode == "nmf":
            C = self._init_C(B_, n, device, dtype)

        K = max(self.steps, 1)
        if self.mode == "vq":
            if K > 1:
                with torch.no_grad():
                    for _ in range(K - 1):
                        D, C, _ = self._vq_step(X, D)
            D_det = D.detach()
            D, C, Xbar = self._vq_step(X, D_det)  # last step with grad
        else:  # nmf
            X_pos = X.relu()  # ensure nonnegativity for NMF
            if K > 1:
                with torch.no_grad():
                    for _ in range(K - 1):
                        D, C, _ = self._nmf_step(X_pos, D, C)
            D_det, C_det = D.detach(), C.detach()
            D, C, Xbar = self._nmf_step(X_pos, D_det, C_det)

        return Xbar


# =========================
# Flatten/Reconstruct helpers
# =========================
# 2D spatial:  (B, C, H, W) -> (B*S, d=C//S, n=H*W)
def _flatten2d_spatial(x: torch.Tensor, S: int):
    B, C, H, W = x.shape
    assert C % S == 0, "C must be divisible by groups_S"
    d = C // S
    n = H * W
    x = x.view(B, S, d, n).reshape(B * S, d, n)
    ctx = (B, C, H, W, d, n, S)
    return x, ctx

def _reconstruct2d_spatial(Xbar: torch.Tensor, ctx):
    B, C, H, W, d, n, S = ctx
    return Xbar.reshape(B, S, d, n).reshape(B, C, H, W)

# 2D channel:  (B, C, H, W) -> (B*S, d=H*W, n=C//S)
def _flatten2d_channel(x: torch.Tensor, S: int):
    B, C, H, W = x.shape
    assert C % S == 0, "C must be divisible by groups_S"
    d = H * W
    n = C // S
    x = x.view(B, S, n, d).permute(0, 1, 3, 2).contiguous().reshape(B * S, d, n)
    ctx = (B, C, H, W, d, n, S)
    return x, ctx

def _reconstruct2d_channel(Xbar: torch.Tensor, ctx):
    B, C, H, W, d, n, S = ctx
    x = Xbar.reshape(B, S, d, n).permute(0, 1, 3, 2).contiguous().reshape(B, C, H, W)
    return x

# 3D spatial:  (B, C, D, H, W) -> (B*S, d=C//S, n=DHW)
def _flatten3d_spatial(x: torch.Tensor, S: int):
    B, C, D, H, W = x.shape
    assert C % S == 0, "C must be divisible by groups_S"
    d = C // S
    n = D * H * W
    x = x.view(B, S, d, n).reshape(B * S, d, n)
    ctx = (B, C, D, H, W, d, n, S)
    return x, ctx

def _reconstruct3d_spatial(Xbar: torch.Tensor, ctx):
    B, C, D, H, W, d, n, S = ctx
    return Xbar.reshape(B, S, d, n).reshape(B, C, D, H, W)

# 3D channel:  (B, C, D, H, W) -> (B*S, d=DHW, n=C//S)
def _flatten3d_channel(x: torch.Tensor, S: int):
    B, C, D, H, W = x.shape
    assert C % S == 0, "C must be divisible by groups_S"
    d = D * H * W
    n = C // S
    x = x.view(B, S, n, d).permute(0, 1, 3, 2).contiguous().reshape(B * S, d, n)
    ctx = (B, C, D, H, W, d, n, S)
    return x, ctx

def _reconstruct3d_channel(Xbar: torch.Tensor, ctx):
    B, C, D, H, W, d, n, S = ctx
    x = Xbar.reshape(B, S, d, n).permute(0, 1, 3, 2).contiguous().reshape(B, C, D, H, W)
    return x


# =========================
# Conv-BN-ReLU helpers (2D/3D)
# =========================
def _bn_nd(num_features: int, dim3d: bool, momentum: float):
    return nn.BatchNorm3d(num_features, momentum=momentum) if dim3d else nn.BatchNorm2d(num_features, momentum=momentum)

def _conv1x1(in_ch: int, out_ch: int, dim3d: bool):
    if dim3d:
        return nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=True)
    else:
        return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

def _conv1x1_nobias(in_ch: int, out_ch: int, dim3d: bool):
    if dim3d:
        return nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
    else:
        return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

def _cheese_block(in_ch: int, out_ch: int, dim3d: bool, momentum: float):
    if dim3d:
        return nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, bias=False),
                             nn.BatchNorm3d(out_ch, momentum=momentum),
                             nn.ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                             nn.BatchNorm2d(out_ch, momentum=momentum),
                             nn.ReLU(inplace=True))


# =========================
# Hamburger 2D / 3D (V2Plus+)
# =========================
class _HamburgerNDPlus(nn.Module):
    """
    Base class to share forward structure between 2D & 3D variants.
    Subclasses must implement:
      - _flatten_spatial, _reconstruct_spatial
      - _flatten_channel, _reconstruct_channel
      - dimension flag (is_3d)
    """
    is_3d: bool = False

    def __init__(self, cfg: HamburgerConfig):
        super().__init__()
        self.cfg = cfg
        C_in = cfg.in_channels
        S = cfg.groups_S
        dual = cfg.dual
        d_mid = cfg.d_mid

        # Lower bread: project to d_mid (dual => twice)
        out_ch = d_mid * (2 if dual else 1)
        lower = [_conv1x1(C_in, out_ch, dim3d=self.is_3d)]
        if cfg.use_relu_in_lower:
            lower += [nn.ReLU(inplace=True)]
        self.lower = nn.Sequential(*lower)

        # Ham(s)
        d_each = d_mid
        self.ham_spatial = HamMD(d=d_each, r=cfg.r, steps=cfg.steps,
                                 mode=cfg.ham_mode, temperature=cfg.temperature)
        if dual:
            self.ham_channel = HamMD(d=d_each, r=cfg.r, steps=cfg.steps,
                                     mode=cfg.ham_mode, temperature=cfg.temperature)
        else:
            self.ham_channel = None

        # Cheese bottleneck
        C_after_ham = out_ch
        C_mid = max(1, C_after_ham // max(1, cfg.cheese_factor))
        self.cheese = _cheese_block(C_after_ham, C_mid, dim3d=self.is_3d, momentum=cfg.bn_momentum)

        # Upper bread + BN(H(Z)) before skip-add
        self.upper = _conv1x1(C_mid, C_in, dim3d=self.is_3d)
        self.bn_h  = _bn_nd(C_in, dim3d=self.is_3d, momentum=cfg.bn_momentum)

        # Learnable mixing (V2+ style)
        self.coef_shortcut = nn.Parameter(torch.tensor(1.0))
        self.coef_ham = nn.Parameter(torch.tensor(0.0 if cfg.zero_ham else 1.0))

    # ---- flatten/reconstruct: to be provided by subclasses ----
    def _flatten_spatial(self, x):  raise NotImplementedError
    def _reconstruct_spatial(self, Xbar, ctx):  raise NotImplementedError
    def _flatten_channel(self, x):  raise NotImplementedError
    def _reconstruct_channel(self, Xbar, ctx):  raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        identity = z
        y = self.lower(z)  # (B, d_mid*(1|2), ...)

        if self.ham_channel is not None:
            # split channels for dual branches
            if self.is_3d:
                B, Ctot, D, H, W = y.shape
            else:
                B, Ctot, H, W = y.shape
            assert Ctot % 2 == 0
            y1, y2 = torch.split(y, Ctot // 2, dim=1)

            # spatial ham
            Xs, ctxs = self._flatten_spatial(y1)
            Xs_bar = self.ham_spatial(Xs)
            y1 = self._reconstruct_spatial(Xs_bar, ctxs)

            # channel ham
            Xc, ctxc = self._flatten_channel(y2)
            Xc_bar = self.ham_channel(Xc)
            y2 = self._reconstruct_channel(Xc_bar, ctxc)

            y = torch.cat([y1, y2], dim=1)
        else:
            # spatial-only ham
            Xs, ctxs = self._flatten_spatial(y)
            Xs_bar = self.ham_spatial(Xs)
            y = self._reconstruct_spatial(Xs_bar, ctxs)

        # Cheese & upper & BN(H) & mix
        y = self.cheese(y)
        h = self.upper(y)
        h = self.bn_h(h)
        out = self.coef_ham * h + self.coef_shortcut * identity
        return F.relu(out, inplace=True)  # keep V2+ style; remove ReLU if you want pure paper form


class Hamburger2DPlus(_HamburgerNDPlus):
    is_3d = False
    def _flatten_spatial(self, x):  return _flatten2d_spatial(x, self.cfg.groups_S)
    def _reconstruct_spatial(self, Xbar, ctx):  return _reconstruct2d_spatial(Xbar, ctx)
    def _flatten_channel(self, x):  return _flatten2d_channel(x, self.cfg.groups_S)
    def _reconstruct_channel(self, Xbar, ctx):  return _reconstruct2d_channel(Xbar, ctx)


class Hamburger3DPlus(_HamburgerNDPlus):
    is_3d = True
    def _flatten_spatial(self, x):  return _flatten3d_spatial(x, self.cfg.groups_S)
    def _reconstruct_spatial(self, Xbar, ctx):  return _reconstruct3d_spatial(Xbar, ctx)
    def _flatten_channel(self, x):  return _flatten3d_channel(x, self.cfg.groups_S)
    def _reconstruct_channel(self, Xbar, ctx):  return _reconstruct3d_channel(Xbar, ctx)


# =========================
# Minimal smoke tests
# =========================
if __name__ == "__main__":
    # 2D
    cfg2d = HamburgerConfig(in_channels=64, d_mid=64, r=16, steps=3,
                            ham_mode="vq", temperature=10.0, groups_S=1,
                            dual=True, cheese_factor=2, zero_ham=True)
    m2d = Hamburger2DPlus(cfg2d)
    x2d = torch.randn(2, 64, 32, 32, requires_grad=True)
    y2d = m2d(x2d)
    assert y2d.shape == x2d.shape
    y2d.mean().backward()
    print("[2D] OK:", x2d.grad is not None)

    # 3D
    cfg3d = HamburgerConfig(in_channels=32, d_mid=64, r=16, steps=3,
                            ham_mode="nmf", groups_S=1, dual=True,
                            cheese_factor=2, zero_ham=True)
    m3d = Hamburger3DPlus(cfg3d)
    x3d = torch.randn(2, 32, 8, 16, 16, requires_grad=True)
    y3d = m3d(x3d)
    assert y3d.shape == x3d.shape
    y3d.mean().backward()
    print("[3D] OK:", x3d.grad is not None)
