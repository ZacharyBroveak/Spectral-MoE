# Spectral-MoE: src/training/analytic_blocks.py
# Author: Zachary Broveak
#
# Allocator training on top of an analytically converted (CMoE-style) MoE.
#
# AnalyticTrainingBlock wraps one converted MoE layer (analytic_primitives/model.py) whose
# weights stay frozen, and lets an allocator decide how many routed experts each token uses.
# Experts are ranked by router probability (CMoE's load-balancing bias is not used here and
# stays zero), and each selected expert's output is weighted by 1 + p * extra_scale, as in the
# converted model.
#
# Gradient horizon. With a hard top-k, the expert just past the cut is never computed, so the
# allocator could not learn what adding it would change. During training the block therefore
# also computes the next grad_horizon ranked experts. Their forward weight is exactly zero,
# but the soft inclusion weights carry gradient through them, so the allocator sees the
# marginal value of one more expert. Without gradients (evaluation), only the selected experts
# run.
#
# STEAdapter gives the StaticAllocator the same alloc / k_for / soft_weights interface as the
# DynamicAllocator, so the block treats both alike. dense_teacher temporarily activates every
# routed expert and bypasses the allocator; the training loop uses it for the teacher logits,
# so the allocator is distilled toward the recovered model with every expert active.

import contextlib

import torch
from torch import nn


class AnalyticTrainingBlock(nn.Module):
    def __init__(self, block, layer_idx, alloc=None, k_fixed=None, grad_horizon=1):
        super().__init__()
        self.block = block
        self.layer_idx = int(layer_idx)
        self._alloc = alloc
        self.grad_horizon = int(grad_horizon)
        if self.grad_horizon < 0:
            raise ValueError(f'grad_horizon={self.grad_horizon} must be >= 0')
        r = getattr(block, 'gate', None)
        need = ('classifier', 'gate', 'act_fn', 'extra_scale', 'extra_bias', 'topk')
        if r is None or not all(hasattr(r, a) for a in need):
            raise TypeError('Expected an analytically constructed block')
        self.n_routed = int(block.n_routed_experts)
        self.top_a = int(block.n_activated_experts)
        self.n_shared = int(block.n_shared_experts)
        self.k_fixed = int(self.top_a if k_fixed is None else k_fixed)
        if not 1 <= self.k_fixed <= self.n_routed:
            raise ValueError(f'k_fixed={self.k_fixed} outside [1, {self.n_routed}]')
        self.use_alloc = True
        for p in self.block.parameters():
            p.requires_grad_(False)
        self.block.cus_training = False
        self._stats = None
        self._rows = 0

    def forward(self, hidden_states):
        shp = hidden_states.shape
        x = hidden_states.reshape(-1, shp[-1])
        n = x.size(0)
        r = self.block.gate
        raw = (r.classifier(x) * r.act_fn(r.gate(x))).abs()
        p = raw.softmax(dim=-1, dtype=torch.float32)
        sel = p + r.extra_bias[None, :].to(p.dtype)
        wbase_all = 1.0 + p * r.extra_scale[None, :].to(p.dtype)
        _use = self.use_alloc and self._alloc is not None
        if _use:
            self._alloc.alloc(self.layer_idx, x)
            k_t = self._alloc.k_for(self.layer_idx).to(x.device)
            if k_t.dim() == 0:
                k_t = k_t.expand(n)
            k_t = k_t.reshape(-1).long()
            k_max = int(k_t.max().item())
        else:
            (k_t, k_max) = (None, self.k_fixed if self.use_alloc else self.top_a)
        # Include additional ranked experts only while computing count gradients.
        h = self.grad_horizon if _use and torch.is_grad_enabled() else 0
        k_disp = min(k_max + h, self.n_routed)
        (_, top_i) = torch.topk(sel, k_disp, dim=-1)
        base = wbase_all.gather(1, top_i)
        if k_t is None:
            keep = torch.ones_like(base)
            dispatch = keep
        else:
            # Separate forward selections from the wider set of computed experts.
            pos = torch.arange(k_disp, device=x.device).unsqueeze(0)
            keep = (pos < k_t.unsqueeze(1)).to(base.dtype)
            dispatch = (pos < (k_t + h).unsqueeze(1)).to(base.dtype)
        w = base * keep
        if _use:
            kappa = self._alloc.soft_weights(self.layer_idx, self.n_routed)
            if kappa.dim() == 1:
                kappa = kappa.unsqueeze(0).expand(n, -1)
            # Keep hard forward weights while passing soft-inclusion gradients.
            s = kappa[:, :k_disp].to(base.dtype)
            w = base * (keep + (s - s.detach()))
        w = w.to(hidden_states.dtype)
        out = self._run_experts(x, top_i, w, dispatch)
        out = out + self.block.shared_experts(x)
        self._stats = {'probs': p, 'k': k_t if k_t is not None else torch.full((n,), k_max, device=x.device, dtype=torch.long), 'kept_mass': (base * keep).sum(-1).detach(), 'expert_rows': int(keep.sum().item()), 'expert_rows_computed': self._rows, 'grad_horizon': h}
        return out.reshape(shp)

    def _run_experts(self, x, top_i, w, dispatch):
        out = torch.zeros_like(x)
        self._rows = 0
        experts = self.block.experts
        for e in range(self.n_routed):
            expert = experts[e]
            if expert is None:
                continue
            # Gather dispatched tokens and add their weighted outputs at the same positions.
            hit = (top_i == e) & (dispatch > 0)
            (tok, posn) = torch.where(hit)
            self._rows += int(tok.numel())
            out.index_add_(0, tok, (expert(x[tok]) * w[tok, posn, None]).to(out.dtype))
        return out


class STEAdapter(nn.Module):
    def __init__(self, ste):
        super().__init__()
        self.ste = ste

    def alloc(self, layer_idx, x):
        return None

    def k_for(self, layer_idx):
        return self.ste.k_int[layer_idx]

    def soft_weights(self, layer_idx, n_experts):
        return self.ste.soft_weights(layer_idx, n_experts)

    def parameters(self, *a, **k):
        return self.ste.parameters(*a, **k)

    def state_dict(self, *a, **k):
        return self.ste.state_dict(*a, **k)

    def load_state_dict(self, *a, **k):
        return self.ste.load_state_dict(*a, **k)


@contextlib.contextmanager
def dense_teacher(wrappers, peft_model=None):
    # Temporarily bypass allocation and activate every routed expert.
    saved = [(w._alloc, w.k_fixed, w.use_alloc) for w in wrappers]
    for w in wrappers:
        w._alloc = None
        w.k_fixed = w.n_routed
        w.use_alloc = True
    ctx = peft_model.disable_adapter() if peft_model is not None else contextlib.nullcontext()
    try:
        with ctx:
            yield
    finally:
        for (w, (a, k, u)) in zip(wrappers, saved):
            (w._alloc, w.k_fixed, w.use_alloc) = (a, k, u)
