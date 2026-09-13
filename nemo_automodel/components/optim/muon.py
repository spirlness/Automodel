# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Muon (Momentum Orthogonalized by Newton-Schulz) optimizer.

Muon orthogonalizes momentum updates for explicitly selected transformer
matrices. Embedding, language-model-head, and normalization parameter groups
use AdamW so they retain their appropriate optimization dynamics.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["Muon", "zeropower_via_newtonschulz5"]


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of 2D matrix G.

    Derivation from Keller Jordan's modded-nanogpt with quintic coefficients
    for rapid spectral convergence.
    """
    assert len(G.shape) >= 2, f"Expected 2D+ matrix, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype in (torch.float32, torch.bfloat16) else G.float()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    # Normalize spectral norm
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.type_as(G)


class Muon(Optimizer):
    """Muon optimizer for modern transformer pretraining.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate (typically 0.01 - 0.05 for Muon).
        momentum: Momentum factor (default 0.95).
        nesterov: Whether to use Nesterov momentum (default True).
        ns_steps: Number of Newton-Schulz iteration steps (default 5).
        weight_decay: Weight decay factor (default 0.01).

    Parameter groups use ``algorithm="muon"`` or ``algorithm="adamw"``.
    ``Muon`` is the default for compatibility; production recipes should use
    ``LocalMuonConfig`` to construct the groups from model parameter names.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            algorithm="muon",
            betas=betas,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if group["algorithm"] == "muon":
                    self._step_muon(p, lr, momentum, nesterov, ns_steps, weight_decay)
                elif group["algorithm"] == "adamw":
                    self._step_adamw(p, lr, weight_decay, group["betas"], group["eps"])
                else:
                    raise ValueError(f"Unsupported Muon parameter-group algorithm {group['algorithm']!r}.")

        return loss

    def _step_muon(
        self,
        parameter: torch.Tensor,
        lr: float,
        momentum: float,
        nesterov: bool,
        ns_steps: int,
        weight_decay: float,
    ) -> None:
        """Apply a Muon update to one 2D+ parameter tensor.

        Args:
            parameter: Tensor of shape [rows, columns] or higher rank, flattened
                logically only by the Newton-Schulz matrix operations.
            lr: Learning rate for this parameter group.
            momentum: Momentum coefficient.
            nesterov: Whether to use Nesterov momentum.
            ns_steps: Number of Newton-Schulz iterations.
            weight_decay: Decoupled weight-decay coefficient.
        """
        if parameter.ndim < 2:
            raise ValueError(f"Muon requires a 2D+ parameter, got shape {tuple(parameter.shape)}.")
        gradient = parameter.grad
        if gradient is None:
            return
        if weight_decay:
            parameter.mul_(1.0 - lr * weight_decay)

        state = self.state[parameter]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(gradient)
        momentum_buffer = state["momentum_buffer"]
        momentum_buffer.mul_(momentum).add_(gradient)
        update = gradient.add(momentum_buffer, alpha=momentum) if nesterov else momentum_buffer
        orthogonal_update = zeropower_via_newtonschulz5(update, steps=ns_steps)
        scale = max(1.0, gradient.size(0) / gradient.size(1)) ** 0.5
        parameter.add_(orthogonal_update, alpha=-lr * scale)

    def _step_adamw(
        self,
        parameter: torch.Tensor,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
    ) -> None:
        """Apply a fp32-state AdamW update to one scalar, embedding, or head tensor.

        Args:
            parameter: Tensor of arbitrary shape, updated in place.
            lr: Learning rate for this parameter group.
            weight_decay: Decoupled weight-decay coefficient.
            betas: AdamW first- and second-moment coefficients.
            eps: AdamW denominator stability constant.
        """
        gradient = parameter.grad
        if gradient is None:
            return
        beta1, beta2 = betas
        state = self.state[parameter]
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(gradient, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(gradient, dtype=torch.float32)
        state["step"] += 1
        step = state["step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        gradient_fp32 = gradient.float()
        exp_avg.lerp_(gradient_fp32, 1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(gradient_fp32, gradient_fp32, value=1.0 - beta2)
        if weight_decay:
            parameter.mul_(1.0 - lr * weight_decay)
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        denominator = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)
        parameter.addcdiv_(exp_avg.to(parameter.dtype), denominator.to(parameter.dtype), value=-lr / bias_correction1)
