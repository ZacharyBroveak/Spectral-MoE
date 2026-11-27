# Spectral-MoE: src/training/blocks.py
# Author: Zachary Broveak
#
# Training-time FFN wrapper for learned conversion.
#
# TrainingFFN wraps one frozen dense FFN of the base model. It reserves s = 128 always-on
# shared neurons (chosen once by calibrate_shared) and learns two things for the remaining
# neurons: an affinity matrix A that defines a balanced partition into E routed experts of 128
# neurons (assignment.py), and a router Wr that selects experts per token. The forward pass
# computes the dense intermediate activations, zeroes the neurons of unselected experts, and
# applies the down projection, so the frozen weights are never copied.
#
# Straight-through estimators make the discrete choices trainable:
# - partition: the hard balanced assignment in the forward pass, the gradient of the Sinkhorn
#   plan in the backward pass;
# - routing: hard top-k selection forward, the gradient of the router softmax backward;
# - expert counts, when an allocator is attached: the hard count forward, the gradient of the
#   soft inclusion weights (soft_inclusion.py) backward, mapped from expert ranks back to
#   expert indices.
#
# The per-token mask over routed neurons is routing @ partition^T. With masked = False the
# block runs the original FFN, which the training loop uses to compute the dense teacher's
# logits with the same model. In eval mode the partition is rounded once and cached. _stats
# keeps router logits, probabilities, and per-expert counts for the auxiliary routing losses.
# sel_bias stays zero during training; it exists so a refit router can use the same layout.
#
# make_allocator builds the allocator a recipe names (None for uniform). calibrate_shared
# picks each layer's shared neurons on a calibration batch by mean |activation| times the L2
# norm of the neuron's down-projection column, an estimate of how much each neuron contributes
# to the layer output.

import torch
from torch import nn

from ..allocator_dynamic import DynamicAllocator
from ..allocator_static import StaticAllocator
from .assignment import balanced_round, log_sinkhorn


class TrainingFFN(nn.Module):
    def __init__(self, mlp, experts, k, width=128):
        super().__init__()
        self.mlp = mlp
        self.E, self.k, self.s = experts, k, width
        d_ffn, d_model = mlp.gate_proj.weight.shape

        self.A = nn.Parameter(torch.empty(d_ffn - width, experts, dtype=torch.float32).normal_(0, .01))
        self.Wr = nn.Parameter(torch.empty(experts, d_model, dtype=mlp.gate_proj.weight.dtype).normal_(0, .02))
        self.register_buffer("carpool", torch.zeros(d_ffn, dtype=torch.bool))
        self.register_buffer("rest_idx", torch.arange(d_ffn - width))
        self.register_buffer("sel_bias", torch.zeros(experts))
        self.use_sel_bias = False
        self.masked = True
        self.tau = 1.0
        self.allocator = None
        self.layer = None
        self._eval_n2e = None
        self._last_n2e = None
        self._stats = None
        self.last_counts = None

    def set_shared(self, indices):
        # Mark shared neurons and rebuild the complementary routed indices.
        self.carpool.zero_()
        self.carpool[indices] = True
        self.rest_idx = (~self.carpool).nonzero().flatten()

    @torch.no_grad()
    def partition(self):
        return balanced_round(log_sinkhorn(self.A, self.s, self.tau, 50), self.s)[1]

    def forward(self, x):
        if not self.masked:
            return self.mlp(x)
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        H = self.mlp.act_fn(self.mlp.gate_proj(xf)) * self.mlp.up_proj(xf)
        if isinstance(self.allocator, DynamicAllocator):
            self.allocator.alloc(self.layer, xf)
            k = self.allocator.k_for(self.layer)
        elif isinstance(self.allocator, StaticAllocator):
            k = int(self.allocator.k_int[self.layer])
        else:
            k = self.k
        logits = xf @ self.Wr.t()
        prob = logits.float().softmax(-1)
        scores = logits.float() + self.sel_bias[None, :] if self.use_sel_bias else prob
        top = scores.topk(int(k.max()) if torch.is_tensor(k) else k, dim=-1).indices
        if torch.is_tensor(k):
            keep = torch.arange(top.shape[1], device=x.device)[None, :] < k[:, None]
            R = torch.zeros_like(prob).scatter_(1, top, keep.to(prob.dtype))
        else:
            R = torch.zeros_like(prob).scatter_(1, top, 1.)
        self.last_counts = k if torch.is_tensor(k) else torch.full((len(xf),), k, device=x.device)
        if self.training:
            soft = log_sinkhorn(self.A, self.s, self.tau, 50)
            with torch.no_grad():
                hard, n2e = balanced_round(soft, self.s)
            # Use hard assignments forward and soft transport gradients backward.
            M = (hard + (soft - soft.detach())).to(H.dtype)
            routing = (R + (prob - prob.detach())).to(H.dtype)
            if self.allocator is not None:
                w = self.allocator.soft_weights(self.layer, self.E)
                if w.ndim == 1:
                    w = w[None, :].expand(len(xf), -1)
                # Map rank-based inclusion weights back to expert indices.
                order = scores.argsort(dim=-1, descending=True)
                inclusion = torch.zeros_like(prob).scatter_(1, order, w.to(prob.dtype))
                routing = routing + (inclusion - inclusion.detach()).to(H.dtype)
            mask = routing @ M.t()
        else:
            if self._eval_n2e is None:
                self._eval_n2e = self.partition()
            n2e = self._eval_n2e
            mask = R.gather(1, n2e[None, :].expand(len(xf), -1)).to(H.dtype)
        self._last_n2e = n2e.detach()
        # Insert routed gates into an all-active mask for the full neuron set.
        full = torch.ones(len(xf), H.shape[1], device=x.device, dtype=H.dtype)
        full = full.index_copy(1, self.rest_idx, mask)
        self._stats = {"L": logits, "P": prob, "counts": R.detach().sum(0)}
        return ((H * full) @ self.mlp.down_proj.weight.t()).reshape(shape)


def make_allocator(recipe, model):
    layers = len(model.model.layers)
    experts = model.config.intermediate_size // 128 - 1
    if recipe.allocator == "uniform":
        return None
    if recipe.allocator == "static":
        alloc = StaticAllocator([recipe.k] * layers, k_max=experts, n_experts=experts,
                                temp=recipe.temperature, seed=recipe.seed)
        alloc.project_()
        return alloc
    return DynamicAllocator(model.config.hidden_size, layers, recipe.k, k_max=experts,
                            n_experts=experts, temp=recipe.temperature,
                            init_spread=recipe.init_spread)


@torch.no_grad()
def calibrate_shared(model, blocks, ids):
    # Capture each block input without retaining its computation graph.
    captures = {}

    def capture(i):
        return lambda m, args: captures.__setitem__(i, args[0].detach().float())

    handles = [b.mlp.register_forward_pre_hook(capture(i)) for i, b in enumerate(blocks)]
    for b in blocks:
        b.masked = False
    try:
        model(ids)
        for i, b in enumerate(blocks):
            x = captures.pop(i).reshape(-1, model.config.hidden_size).to(b.Wr.dtype)
            h = b.mlp.act_fn(b.mlp.gate_proj(x)) * b.mlp.up_proj(x)
            # Rank neurons by mean activation magnitude times output-column L2 norm.
            importance = h.abs().float().mean(0) * b.mlp.down_proj.weight.float().norm(dim=0)
            b.set_shared(importance.argsort(descending=True)[:b.s])
    finally:
        for h in handles:
            h.remove()
        for b in blocks:
            b.masked = True
