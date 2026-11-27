# Spectral-MoE: src/model/router.py
# Author: Zachary Broveak
#
# Top-k expert selection for restored learned MoE layers.
#
# route() scores experts with the router weights Wr (logits = x @ Wr^T) and selects the top k
# per token. Ranking uses the softmax probabilities, as in training; after a ridge refit
# (training/refit.py) the router has a per-expert intercept, and ranking uses logits +
# sel_bias instead. k is either one integer for the whole layer (uniform or static allocation)
# or one count per token from the dynamic allocator, in which case the top max(k) experts are
# taken and each token keeps only its first k.
#
# It returns a 0/1 selection matrix R of shape [tokens, experts] and the number of tokens
# assigned to each expert. Selection only gates experts: a selected expert's neurons pass
# through unscaled, exactly as in training.

import torch


@torch.no_grad()
def route(x: torch.Tensor, Wr: torch.Tensor, k, sel_bias: torch.Tensor = None):
    logits = x @ Wr.t()
    probs = torch.softmax(logits.float(), dim=-1)

    # Apply the optional offset to ranking logits.
    score = (logits.float() + sel_bias[None, :]) if sel_bias is not None else probs

    if torch.is_tensor(k):
        # Select up to the largest count, then mask each token to its own count.
        k_max = int(k.max().item())
        _, top = score.topk(k_max, dim=-1)
        keep = torch.arange(k_max, device=k.device).unsqueeze(0) < k.unsqueeze(1)
        R = torch.zeros_like(probs).scatter_(1, top, keep.to(probs.dtype))
    else:
        _, top = score.topk(int(k), dim=-1)
        R = torch.zeros_like(probs).scatter_(1, top, 1.0)
    return R, R.sum(0)
