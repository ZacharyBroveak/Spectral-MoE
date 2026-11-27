# Spectral-MoE: src/model/masked_ffn.py
# Author: Zachary Broveak
#
# Masked realization of a learned-base MoE layer, used for perplexity.
#
# Learned conversion never moves weights: an expert is a set of 128 neurons of the original
# FFN. MaskedFFN therefore runs the original dense FFN up to its intermediate activations,
# multiplies them by a 0/1 mask that keeps only the neurons of the experts selected for each
# token (shared neurons are always kept), and applies the down projection. The output equals
# executing only the selected experts, and the computation closely follows training
# (training/blocks.py: TrainingFFN), which keeps evaluation perplexity comparable with the
# curves logged during training. It does not save compute; sparse_experts.py is the
# realization that does.
#
# The constructor checks that the saved partition and shared-neuron set fit together: every
# routed neuron belongs to an expert, expert ids are in range, and all experts have the same
# number of neurons. A failure means the files in a run directory do not belong to the same
# run.

import torch
from torch import nn

from .router import route


class MaskedFFN(nn.Module):
    def __init__(self, mlp, Wr, partition, always_on, k, sel_bias=None):
        super().__init__()
        self.mlp = mlp
        self.register_buffer("Wr", Wr, persistent=False)
        self.register_buffer("sel_bias", sel_bias, persistent=False)

        d_ffn = int(partition.numel())
        on = always_on
        # Convert shared-neuron indices to a mask over the full FFN width.
        if on is not None and on.dtype != torch.bool:
            m = torch.zeros(d_ffn, dtype=torch.bool, device=partition.device)
            m[on.long()] = True
            on = m
        self.has_always_on = on is not None and bool(on.any())

        if self.has_always_on:
            # Exclude shared neurons from the routed partition.
            routed_idx = torch.nonzero(~on, as_tuple=True)[0]
            routed = partition[routed_idx]
            if not bool((routed >= 0).all()):
                raise ValueError(
                    "a routed neuron is marked -1 in the saved partition, so the always-on set "
                    "recorded for this block does not match the partition recorded beside it")
            self.register_buffer("routed_idx", routed_idx, persistent=False)
        else:
            routed = partition
            self.register_buffer("routed_idx", torch.arange(d_ffn, device=partition.device),
                                 persistent=False)

        E = int(Wr.shape[0])
        if int(routed.min()) < 0 or int(routed.max()) >= E:
            raise ValueError(f"the saved partition holds expert ids outside [0, {E})")
        # Require every routed expert to contain the same number of neurons.
        sizes = torch.bincount(routed, minlength=E)
        if int(sizes.min()) != int(sizes.max()):
            raise ValueError(
                f"the saved partition is not balanced: expert sizes run "
                f"[{int(sizes.min())}, {int(sizes.max())}]. Every expert holds the same number of "
                f"neurons in a valid conversion, so this is not one.")
        self.register_buffer("partition", routed, persistent=False)
        self.d_ffn = d_ffn
        self.expert_size = int(sizes[0])

        self.k = k
        self.allocator = None
        self.layer = None
        self.last_counts = None

    def attach(self, allocator, layer):
        self.allocator = allocator
        self.layer = int(layer)

    def forward(self, x):
        shape = x.shape
        xf = x.reshape(-1, shape[-1])

        H = self.mlp.act_fn(self.mlp.gate_proj(xf)) * self.mlp.up_proj(xf)

        if self.allocator is not None:
            self.allocator.alloc(self.layer, xf)
            k = self.allocator.k_for(self.layer)
        else:
            k = self.k

        R, _ = route(xf, self.Wr, k, sel_bias=self.sel_bias)
        self.last_counts = R.sum(1).long()

        # Expand expert selections to neuron gates; shared neurons stay active.
        mask = R.gather(1, self.partition.unsqueeze(0).expand(R.size(0), -1)).to(H.dtype)
        if self.has_always_on:
            full = torch.ones(mask.size(0), self.d_ffn, dtype=mask.dtype, device=mask.device)
            mask = full.index_copy(1, self.routed_idx, mask)

        y = (H * mask) @ self.mlp.down_proj.weight.t()
        return y.reshape(shape)
