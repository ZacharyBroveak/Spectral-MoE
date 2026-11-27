# Spectral-MoE: src/allocator_dynamic.py
# Author: Zachary Broveak
#
# Dynamic allocator: a small network that decides, for each token, how many routed experts
# each layer runs.
#
# Every token must spend exactly total routed experts across the L layers (the same budget as
# uniform allocation), with between k_min and k_max in every layer. Layers run in order, so
# the allocator works in order too: at layer l it sees the token's hidden state entering that
# FFN and commits the layer's count before later layers exist.
#
# How a count is chosen:
# - Features: the hidden state normalized to unit RMS, plus its log RMS as one extra input
#   (_in). A shared projection P with a GELU turns these into an L-dimensional feature z.
# - Layer 0 produces a plan, logits over all L layers (head W[0]), describing how the token's
#   budget should be shared among the layers that remain.
# - At each later layer, a revision head W[l] sees z for the current hidden state together
#   with the remaining plan and proposes a revision. Both the revision and the total deviation
#   from the layer-0 plan pass through tanh, so the plan can be adjusted but never moves more
#   than one logit from the initial plan.
# - The current layer receives the share softmax(remaining plan)[0] of the free budget, which
#   is what is left after reserving k_min for every remaining layer. The last layer takes
#   whatever is left.
# - The continuous count kappa is clipped to bounds that keep the remaining layers feasible
#   (bounds()) and rounded to the integer count k_int that the layer runs.
#
# Exactness. Integer spending is tracked in _spent and its straight-through continuous version
# in _spent_soft. The two must agree and every token must end at exactly total; both are
# asserted on every forward pass, so a budget violation fails immediately instead of silently
# producing an unfair comparison.
#
# Initialization. _calibrate() rescales the heads so the initial plan logits have standard
# deviation init_spread (a recipe value around 1e-3). The starting allocation is therefore
# almost exactly uniform, and training moves away from uniform only where that lowers the
# loss.
#
# Usage contract. The allocator keeps per-forward state on the module. In each forward pass
# every layer calls alloc(layer, h) once, in order starting from layer 0 (which resets the
# state), then reads its counts once with k_for(layer); during training soft_weights(layer, E)
# supplies the soft inclusion weights. Out-of-order or repeated calls raise errors instead of
# reusing stale counts. The configuration needed to rebuild the module is saved with its
# weights (training/learned.py: allocator_record) and read back with from_config().

import torch
from torch import nn

from .soft_inclusion import soft_inclusion_rows


class DynamicAllocator(nn.Module):
    def __init__(self, d_model: int, n_layers: int, k_init: float, total: int | None = None,
                 temp: float = 1.0, k_min: int = 1, k_max: int | None = None,
                 n_experts: int | None = None,
                 init_spread: float = 0.001):
        super().__init__()
        self.nL = int(n_layers)
        self.k_init = float(k_init)
        self.total = int(total if total is not None else round(self.nL * self.k_init))
        self.temp = float(temp)
        self.k_min = int(k_min)
        self.n_experts = int(n_experts if n_experts is not None else k_max or 1)
        self.k_max = int(k_max if k_max is not None else self.n_experts)
        assert self.nL * self.k_min <= self.total <= self.nL * self.k_max, (
            f"budget {self.total} is infeasible for {self.nL} layers in [{self.k_min}, {self.k_max}]")
        self.d_in = int(d_model) + 1
        self.n_heads = self.nL - 1
        self.act = nn.GELU()
        self.init_spread = float(init_spread)

        # Revision heads shrink with the number of uncommitted layers.
        self.P = nn.Linear(self.d_in, self.nL)
        self.W = nn.ModuleList([nn.Linear(self.nL if l == 0 else self.nL + (self.nL - l),
                                          self.nL - l) for l in range(self.n_heads)])
        nn.init.zeros_(self.P.weight[:, self.d_in - 1])
        for w in self.W:
            nn.init.normal_(w.weight, std=(1.0 / max(w.in_features, 1)) ** 0.5)
            nn.init.zeros_(w.bias)

        self.register_buffer("_calibrated", torch.zeros((), dtype=torch.bool))
        self.register_buffer("_achieved_spread", torch.zeros(()))
        if self.init_spread > 0:
            self._calibrate()

        self._spent = None
        self._spent_soft = None
        self._plan = None
        self._plan0 = None
        self._kappa = [None] * self.nL
        self._kint = [None] * self.nL
        self._used = [True] * self.nL

    @classmethod
    def from_config(cls, cfg: dict) -> "DynamicAllocator":
        cfg = dict(cfg)
        cfg.pop("class", None)
        return cls(**cfg)

    def _in(self, h: torch.Tensor) -> torch.Tensor:
        # Normalize each hidden vector and append its log-RMS magnitude.
        x = h.float()
        ms = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(ms + 1e-6)
        return torch.cat([x, 0.5 * torch.log(ms + 1e-12)], dim=-1)

    def _z(self, x):
        return self.act(self.P(x))

    def _head0(self, x):
        return self.W[0](self._z(x))

    def _head_rev(self, layer, x, plan):
        return self.W[layer](torch.cat([self._z(x), plan], dim=-1))

    @torch.no_grad()
    def _calibrate(self):
        # Use a separate random stream to measure the initial logit spread.
        g = torch.Generator(device="cpu").manual_seed(1234)
        x = torch.randn(4096, self.d_in, generator=g)
        x[:, -1] = 0.0
        sd = float(self._head0(x).std())
        if sd > 0:
            # Rescale every output head by the same factor.
            scale = self.init_spread / sd
            for w in self.W:
                w.weight.mul_(scale)
                w.bias.mul_(scale)
        self._achieved_spread.fill_(float(self._head0(x).std()))
        self._calibrated.fill_(True)

    def begin(self, n: int, device, dtype=torch.int64):
        self._spent = torch.zeros(n, device=device, dtype=dtype)
        self._spent_soft = torch.zeros(n, device=device, dtype=torch.float32)
        self._plan = None
        self._plan0 = None
        self._kappa = [None] * self.nL
        self._kint = [None] * self.nL
        self._used = [True] * self.nL

    def bounds(self, layer: int):
        # Leave enough capacity to satisfy all remaining layer bounds.
        m = self.nL - layer - 1
        rem = self.total - self._spent
        lo = torch.clamp(rem - self.k_max * m, min=self.k_min)
        hi = torch.clamp(rem - self.k_min * m, max=self.k_max)
        return lo, hi

    def alloc(self, layer: int, h: torch.Tensor) -> tuple:
        if layer == 0:
            self.begin(h.size(0), h.device)
        if self._spent is None or (layer > 0 and self._plan is None):
            raise RuntimeError(
                "alloc() reached a layer above 0 without a schedule from upstream. The chain is "
                "strictly sequential: a skipped or reordered layer gives a stale remainder.")
        if layer == self.nL - 1:
            logits = self._plan
        elif layer == 0:
            logits = self._head0(self._in(h))
            self._plan0 = logits
        else:
            # Bound both the new revision and the deviation from the initial plan.
            rev = torch.tanh(self._head_rev(layer, self._in(h), self._plan))
            dev = torch.tanh((self._plan + rev) - self._plan0[:, layer:])
            logits = self._plan0[:, layer:] + dev
        nrem = self.nL - layer
        share = torch.softmax(logits.float(), dim=-1)[:, 0]

        # Divide the free budget among the remaining layers.
        rem_soft = (float(self.total) - self._spent_soft)
        free = (rem_soft - nrem * self.k_min).clamp_min(0.0)
        kappa = self.k_min + free * share
        lo, hi = self.bounds(layer)
        with torch.no_grad():
            kc = kappa.detach().clamp(lo.float(), hi.float())
            k_int = kc.round().to(torch.int64).clamp(lo, hi)
        # Keep clipped forward values and pass proposal gradients unchanged.
        kappa = kappa + (kc - kappa.detach())
        k_ste = k_int.float() + (kappa - kappa.detach())
        self._spent = self._spent + k_int
        self._spent_soft = self._spent_soft + k_ste
        self._kappa[layer], self._kint[layer] = kappa, k_int
        self._used[layer] = False
        # Remove the committed layer from the remaining schedule.
        self._plan = logits[:, 1:] if nrem > 1 else None
        if not bool((self._spent_soft.detach() - self._spent.float()).abs().max() < 1e-3):
            raise AssertionError(
                f"the differentiable remainder diverged from the exact one at layer {layer}")
        if layer == self.nL - 1 and not bool((self._spent == self.total).all()):
            raise AssertionError(
                f"per-token budget violated: totals {self._spent.unique().tolist()[:8]} != "
                f"{self.total}")
        return kappa, k_int

    def cached(self) -> tuple:
        if any(k is None for k in self._kint):
            raise RuntimeError("cached() before a full stack has run")
        return (torch.stack(self._kappa, dim=1), torch.stack(self._kint, dim=1))

    def soft_weights(self, layer: int, E: int) -> torch.Tensor:
        if self._kappa[layer] is None:
            raise RuntimeError(f"soft_weights({layer}) before alloc({layer}) in this forward")
        return soft_inclusion_rows(self._kappa[layer], E, self.temp)

    def k_for(self, layer: int) -> torch.Tensor:
        if self._kint[layer] is None:
            raise RuntimeError(f"k_for({layer}) before alloc({layer}) in this forward")
        if self._used[layer]:
            raise RuntimeError(f"layer {layer} read its counts twice without re-running alloc()")
        self._used[layer] = True
        return self._kint[layer]

    def clear(self):
        self._spent = None
        self._spent_soft = None
        self._plan = None
        self._plan0 = None
        self._kappa = [None] * self.nL
        self._kint = [None] * self.nL
        self._used = [True] * self.nL
