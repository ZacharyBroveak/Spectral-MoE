# Spectral-MoE: src/allocator_static.py
# Author: Zachary Broveak
#
# Static allocator: one learned routed-expert count per layer, shared by all tokens.
#
# Uniform allocation gives every layer the same routed count e_bar. The static allocator
# learns a separate count for each layer while keeping their sum equal to the budget (layers *
# e_bar), so it can move capacity toward the layers that need it most. It is the per-layer
# counterpart of the per-token DynamicAllocator.
#
# kappa holds one continuous count per layer and is trained through the soft inclusion weights
# of soft_inclusion.py. k_int is the integer schedule the forward pass actually runs. Around
# each optimizer step the training loop calls:
# - center_grad_() before the step, which subtracts the mean gradient, so the gradient only
#   moves capacity between layers and has no component that changes the total. AdamW scales
#   each coordinate separately, so the update itself can still shift the total slightly;
# - project_() after the step, which shifts kappa back onto the budget, clamps it to [0.5,
#   n_experts - 0.5] (rounded half up, every value in this range is a valid count from 1 to
#   n_experts), redistributes the excess among layers that can still move, and derives k_int
#   with round_to_total (largest-remainder rounding to the exact budget; see
#   soft_inclusion.py).
#
# The random generator is used only by round_to_total's rare clamping repair. It is seeded
# from the recipe seed and saved in resume checkpoints, so integer schedules are identical
# after a resume. state() is the export format of learned arms, which model/restore.py reads
# back; analytic arms save the module's state_dict, which model/analytic.py loads.

import torch
from torch import nn

from .soft_inclusion import round_to_total, soft_inclusion


class StaticAllocator(nn.Module):
    def __init__(self, k_init, total=None, k_min=1, k_max=None, temp=1.5, seed=1, n_experts=None):
        super().__init__()

        if int(k_min) < 1:
            raise ValueError(f"k_min={k_min} would allow a layer with no routed expert")
        k0 = torch.as_tensor(list(k_init), dtype=torch.float32)
        self.nL = int(k0.numel())
        self.total = int(total if total is not None else round(float(k0.sum())))
        self.k_min = int(k_min)
        self.k_max = int(k_max if k_max is not None else self.total)
        self.temp = float(temp)

        self.n_experts = int(n_experts if n_experts is not None
                             else (k_max if k_max is not None else self.total))

        self.kappa = nn.Parameter(k0.clone())

        self.register_buffer("k_int", torch.as_tensor(list(k_init), dtype=torch.int64))
        self._gen = torch.Generator(device="cpu").manual_seed(int(seed) + 90210)

    def soft_weights(self, l: int, E: int) -> torch.Tensor:
        return soft_inclusion(self.kappa[l], E, self.temp)

    @torch.no_grad()
    def project_(self) -> torch.Tensor:
        # Remove total-count drift by shifting all layers equally.
        excess = float(self.kappa.detach().sum()) - self.total
        self.kappa.sub_(excess / self.nL)
        _lo, _hi = 0.5, float(self.n_experts) - 0.5
        for _ in range(64):
            _kd = self.kappa.detach()
            _oob = int(((_kd < _lo) | (_kd > _hi)).sum())
            if _oob:
                self.kappa.clamp_(_lo, _hi)
            _kd = self.kappa.detach()
            _ex = float(_kd.sum()) - self.total
            if abs(_ex) < 1e-6:
                break
            # Redistribute the residual only among counts that can still move.
            _free = (_kd > _lo) if _ex > 0 else (_kd < _hi)
            _n = int(_free.sum())
            if _n == 0:
                break
            self.kappa[_free] -= _ex / _n
        # Store a bounded integer schedule with the required total.
        k = round_to_total(self.kappa, self.total, self.k_min, self.k_max, self._gen)
        self.k_int.copy_(k.to(self.k_int.device))
        return self.k_int

    @torch.no_grad()
    def center_grad_(self):
        # Remove the gradient component that changes the sum of counts.
        if self.kappa.grad is not None:
            self.kappa.grad.sub_(self.kappa.grad.mean())

    def state(self):
        return {"kappa": self.kappa.detach().cpu().clone(), "k_int": self.k_int.detach().cpu().clone(),
                "total": self.total, "k_min": self.k_min, "k_max": self.k_max, "temp": self.temp}
