# Spectral-MoE: src/soft_inclusion.py
# Author: Zachary Broveak
#
# Differentiable relaxation of "run the top k experts", shared by both allocators.
#
# An allocator produces a continuous count kappa for a layer, but the forward pass needs an
# integer: the layer runs its k highest-ranked experts. So that gradients can reach kappa, the
# expert at rank i (counting from 0) gets a soft inclusion weight sigmoid((kappa - i - 0.5) /
# temp). The weight is close to 1 for ranks well below kappa, close to 0 well above it, and
# exactly 0.5 at i = kappa - 0.5, so the ranks with weight above 0.5 are exactly the top
# round(kappa). Training runs hard top-k in the forward pass and uses these weights only for
# the backward pass (a straight-through estimator): increasing kappa raises the weight of the
# next expert in line, so the gradient tells the allocator how much adding or removing an
# expert at the margin would change the loss. temp controls how many ranks around the cut
# receive gradient.
#
# soft_inclusion handles one count per layer (static allocator); soft_inclusion_rows handles
# one count per token (dynamic allocator).
#
# round_to_total converts continuous per-layer counts into integers that sum to exactly the
# budget. It rescales the counts to the total and rounds by largest remainder: every count is
# rounded down, and the units still missing go one each to the counts with the largest
# fractional parts. The result is then clamped to [k_min, k_max]. Clamping changes the sum
# only when a rounded count falls outside those bounds; with the static allocator's bounds,
# that means a count below 1 rounded down to 0 and raised to k_min. Only in that case is the
# difference repaired, one unit at a time at randomly chosen eligible layers, using the
# supplied generator so the repair is reproducible.

import torch


def soft_inclusion(kappa_l: torch.Tensor, n_experts: int, temp: float) -> torch.Tensor:
    # Center the sigmoid transition halfway between adjacent ranks.
    i = torch.arange(n_experts, device=kappa_l.device, dtype=torch.float32)
    return torch.sigmoid((kappa_l - i - 0.5) / max(temp, 1e-6))


def soft_inclusion_rows(kappa: torch.Tensor, n_experts: int, temp: float) -> torch.Tensor:
    # Broadcast each token count across all expert ranks.
    i = torch.arange(n_experts, device=kappa.device, dtype=torch.float32)
    return torch.sigmoid((kappa.float().unsqueeze(1) - i.unsqueeze(0) - 0.5) / max(temp, 1e-6))


def round_to_total(x: torch.Tensor, total: int, k_min: int, k_max: int,
                          gen: torch.Generator = None) -> torch.Tensor:
    n = x.numel()
    assert n * k_min <= total <= n * k_max, (
        f"total {total} is outside [{n * k_min}, {n * k_max}] for {n} layers in [{k_min}, {k_max}]")
    # Rescale nonnegative counts to the requested total before rounding.
    v = x.detach().to(torch.float64).clamp_min(0.0)
    s = float(v.sum())
    v = (v * (total / s)) if s > 0 else torch.full_like(v, total / n)

    # Assign the rounding deficit to the largest fractional counts.
    fl = torch.floor(v)
    deficit = round(total - float(fl.sum()))
    k = fl.to(torch.int64)
    if deficit > 0:
        frac = v - fl
        idx = torch.argsort(frac, descending=True)[:deficit]
        k[idx] += 1
    elif deficit < 0:
        frac = v - fl
        idx = torch.argsort(frac, descending=False)[:(-deficit)]
        k[idx] -= 1

    k = k.clamp_(k_min, k_max)

    guard = 0
    while True:
        diff = int(total - int(k.sum()))
        if diff == 0:
            break
        guard += 1
        assert guard <= 64 * n, f"allocation repair failed to converge (diff {diff}, k {k.tolist()})"
        # Repair one budget unit at a randomly chosen eligible layer.
        elig = (k < k_max) if diff > 0 else (k > k_min)
        cand = torch.nonzero(elig, as_tuple=True)[0]
        assert cand.numel() > 0, f"no element can absorb diff {diff} within [{k_min}, {k_max}]"
        j = cand[int(torch.randint(cand.numel(), (1,), generator=gen, device="cpu"))]
        k[j] += 1 if diff > 0 else -1
    return k
