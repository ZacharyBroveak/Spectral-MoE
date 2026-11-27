# Spectral-MoE: src/model/sparse_experts.py
# Author: Zachary Broveak
#
# Extracted realization of a learned-base MoE layer, used for benchmark accuracy.
#
# Where MaskedFFN computes the whole dense FFN and masks it, SparseExperts gathers each
# expert's neurons into its own weight slices (w_gate and w_up of shape [E, s, d_model],
# w_down of shape [E, d_model, s]) and computes only the experts each token selects, plus the
# always-on shared neurons. This is the form an efficient MoE deployment would take, so
# benchmark accuracy is measured on a model that really skips the unselected computation. The
# loop over experts is written for clarity, not speed.
#
# load_from() copies the slices out of the dense weights according to the saved partition.
# Shared-neuron columns are zeroed inside expert slices so that no neuron can contribute
# twice. For a dynamic arm the per-token counts come from an attached allocator; running such
# a layer without one raises an error instead of silently falling back to a uniform count,
# which would evaluate the wrong method under the right name.

import torch
from torch import nn


class SparseExperts(nn.Module):
    def __init__(self, d_model, s, E, k, dtype=torch.bfloat16, always_on_idx=None,
                 requires_allocator=False):
        super().__init__()
        self.d_model, self.s, self.E = int(d_model), int(s), int(E)

        per_token = torch.is_tensor(k) and k.numel() != 1
        self.k = None if per_token else int(k)
        self._k_needs_alloc = bool(per_token) or bool(requires_allocator)

        self.w_gate = nn.Parameter(torch.zeros(self.E, self.s, self.d_model, dtype=dtype))
        self.w_up = nn.Parameter(torch.zeros(self.E, self.s, self.d_model, dtype=dtype))
        self.w_down = nn.Parameter(torch.zeros(self.E, self.d_model, self.s, dtype=dtype))
        self.Wr = nn.Parameter(torch.zeros(self.E, self.d_model, dtype=dtype))
        self.act = nn.SiLU()

        idx = (torch.as_tensor(always_on_idx, dtype=torch.long).reshape(-1)
               if always_on_idx is not None and len(always_on_idx) else torch.zeros(0, dtype=torch.long))
        self.register_buffer("always_on_idx", idx)
        n_on = idx.numel()
        self.register_buffer("on_gate", torch.zeros(n_on, self.d_model, dtype=dtype))
        self.register_buffer("on_up", torch.zeros(n_on, self.d_model, dtype=dtype))
        self.register_buffer("on_down", torch.zeros(self.d_model, n_on, dtype=dtype))

        self.register_buffer("sel_bias", torch.zeros(self.E, dtype=torch.float32))
        self.use_sel_bias = False

        self._allocator = None
        self._layer_index = None
        self._last_k = None

    @torch.no_grad()
    def load_from(self, gate_w, up_w, down_w, Wr, expert_neurons):
        on = self.always_on_idx.to(gate_w.device)
        on_set = torch.zeros(gate_w.shape[0], dtype=torch.bool, device=gate_w.device)
        if on.numel():
            on_set[on] = True
            self.on_gate.copy_(gate_w[on])
            self.on_up.copy_(up_w[on])
            self.on_down.copy_(down_w[:, on])
        for e, idx in enumerate(expert_neurons):
            self.w_gate[e].copy_(gate_w[idx])
            self.w_up[e].copy_(up_w[idx])
            # Zero shared-neuron columns so their output is not added twice.
            dn = down_w[:, idx].clone()
            if on.numel():
                dn[:, on_set[idx]] = 0
            self.w_down[e].copy_(dn)
        self.Wr.copy_(Wr)

    def attach_allocator(self, allocator, layer_index):
        self._allocator = allocator
        self._layer_index = int(layer_index)

    def _resolve_k(self, x):
        if self._allocator is not None:
            self._allocator.alloc(self._layer_index, x)
            k = self._allocator.k_for(self._layer_index)
        else:
            if self._k_needs_alloc or self.k is None:
                raise RuntimeError(
                    "This layer was built from a learned-allocator arm, so its expert count is "
                    "per-token, but no allocator is attached. Call attach_allocator() before the "
                    "forward. Falling back to a scalar count here would evaluate a uniform "
                    "allocation and report it as the learned one.")
            k = self.k
        self._last_k = k
        return k

    def _dispatch_pairs(self, x, k):
        logits = x @ self.Wr.t()
        score = (logits.float() + self.sel_bias[None, :]) if self.use_sel_bias\
            else torch.softmax(logits.float(), dim=-1)
        n = x.size(0)
        if torch.is_tensor(k):
            kmax = int(k.max().item())
            if kmax == 0:
                return None, None
            _, topi = score.topk(kmax, dim=-1)
            # Flatten only token-expert pairs within each token count.
            keep = torch.arange(kmax, device=x.device).unsqueeze(0) < k.unsqueeze(1)
            tok = torch.arange(n, device=x.device).unsqueeze(1).expand(n, kmax)
            return tok[keep], topi[keep]
        _, topi = score.topk(int(k), dim=-1)
        tok = torch.arange(n, device=x.device).unsqueeze(1).expand_as(topi)
        return tok.reshape(-1), topi.reshape(-1)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.d_model)
        y = torch.zeros_like(x)

        # Apply the shared neurons to every token.
        if self.always_on_idx.numel():
            h = self.act(x @ self.on_gate.t()) * (x @ self.on_up.t())
            y = y + h @ self.on_down.t()

        k = self._resolve_k(x)
        if (not torch.is_tensor(k)) and int(k) == 0:
            return y.reshape(shape)
        flat_tok, flat_exp = self._dispatch_pairs(x, k)
        if flat_tok is None:
            return y.reshape(shape)

        selected = torch.zeros(x.size(0), self.E, device=x.device, dtype=torch.bool)
        selected[flat_tok, flat_exp] = True
        for e in range(self.E):
            idx = selected[:, e].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            # Compute this expert on its selected tokens and accumulate their outputs.
            xe = x.index_select(0, idx)
            h = self.act(xe @ self.w_gate[e].t()) * (xe @ self.w_up[e].t())
            y.index_add_(0, idx, h @ self.w_down[e].t())
        return y.reshape(shape)
