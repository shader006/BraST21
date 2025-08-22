# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Union

class LearnableDSUncertaintyLoss(nn.Module):
    """
    Học trọng số cho Deep Supervision theo công thức:
      sum(exp(-s_i) * L_i) + sum(s_i)
    Với head chính (main) cố định s_main = 0 (tức weight_main = 1, không học).
    Mặc định n_heads=3 (aux1, aux2, main) — phù hợp yêu cầu "chỉ w1, w2 học".
    """
    def __init__(
        self,
        base_loss: nn.Module,
        n_heads: int = 3,      # aux1, aux2, main
        main_index: int = -1,  # head cuối là main
        init_value: float = 0.0,
        interp_mode: str = "nearest",  # resample label cho từng head
    ):
        super().__init__()
        assert n_heads >= 1
        self.base_loss = base_loss
        self.n_heads = n_heads
        self.main_index = main_index if main_index >= 0 else (n_heads - 1)
        self.interp_mode = interp_mode

        # chỉ học cho (n_heads - 1) head phụ (ở đây là 2 aux): s_main cố định = 0
        self.theta = nn.Parameter(torch.full((n_heads - 1,), float(init_value)))

    @torch.no_grad()
    def current_weights(self) -> torch.Tensor:
        """
        Trả về w_i ~ exp(-s_i) đã chuẩn hoá để bạn quan sát/log.
        (Không dùng trong loss — loss dùng công thức gốc)
        """
        device, dtype = self.theta.device, self.theta.dtype
        s_full = torch.zeros(self.n_heads, device=device, dtype=dtype)
        j = 0
        for i in range(self.n_heads):
            if i == self.main_index:
                s_full[i] = 0.0  # main cố định
            else:
                s_full[i] = self.theta[j]; j += 1
        w = torch.exp(-s_full)
        w = w / (w.sum() + 1e-12)
        return w

    def forward(self, inputs: Union[torch.Tensor, Sequence[torch.Tensor]], target: torch.Tensor):
        # Nếu model chưa bật DS (chỉ ra 1 head), trả về base_loss như thường
        if not isinstance(inputs, (list, tuple)):
            target = target.float() if target.dtype != torch.float32 else target
            return self.base_loss(inputs, target)

        if len(inputs) != self.n_heads:
            raise ValueError(f"Expected {self.n_heads} heads, got {len(inputs)}")

        device = inputs[0].device
        losses = []
        
        for i, x in enumerate(inputs):
            t = target.clone()
            if t.dtype != torch.float32:
                t = t.float()
            
            # Resize target to match prediction if needed
            if t.shape[2:] != x.shape[2:]:
                t = F.interpolate(t, size=x.shape[2:], mode=self.interp_mode, align_corners=False)
            
            loss = self.base_loss(x, t)
            losses.append(loss)

        L = torch.stack(losses)  # [n_heads]

        # Construct s_full (s_main = 0)
        s_full = torch.zeros(self.n_heads, device=device, dtype=L.dtype)
        j = 0
        for i in range(self.n_heads):
            if i == self.main_index:
                s_full[i] = 0.0  # Main head fixed at 0
            else:
                s_full[i] = self.theta[j]
                j += 1
        
        # Clamp to prevent overflow/underflow
        s_full = torch.clamp(s_full, min=-10.0, max=10.0)

        # Uncertainty weighting: sum(exp(-s)*L) + sum(s)
        weighted_losses = torch.exp(-s_full) * L
        total_loss = torch.sum(weighted_losses) + torch.sum(s_full)
        
        return total_loss
