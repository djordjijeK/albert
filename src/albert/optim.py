import math

import torch
from torch import nn


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    matrices = [p for p in model.parameters() if p.ndim >= 2]
    others = [p for p in model.parameters() if p.ndim < 2]
    return [{"params": matrices, "weight_decay": weight_decay}, {"params": others, "weight_decay": 0.0}]


class StableAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 8e-4, betas=(0.9, 0.98), eps: float = 1e-6, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t, g, m, v = state["step"], p.grad, state["m"], state["v"]

                m.lerp_(g, 1 - beta1)
                v.lerp_(g * g, 1 - beta2)
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                rms = (g * g / v_hat.clamp(min=eps**2)).mean().sqrt()
                lr = group["lr"] / rms.clamp(min=1.0)

                p.mul_(1 - lr * group["weight_decay"])
                p.sub_(lr * m_hat / (v_hat.sqrt() + eps))


def wsd_lr(step: int, peak_lr: float, warmup_steps: int, decay_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    decay_start = total_steps - decay_steps
    if step < decay_start:
        return peak_lr

    progress = (step - decay_start) / decay_steps
    return peak_lr * (1 - math.sqrt(progress))
