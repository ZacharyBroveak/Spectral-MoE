# Spectral-MoE: src/training/assignment.py
# Author: Zachary Broveak
#
# Balanced neuron-to-expert assignment for learned conversion.
#
# Each layer learns an affinity matrix A of shape [routed neurons, experts]. A valid partition
# puts every neuron in exactly one expert and exactly s = 128 neurons in every expert. This
# module turns A into such a partition in two forms.
#
# log_sinkhorn computes the soft form: the entropic optimal-transport plan with row sums 1 and
# column sums s, found by alternately normalizing the rows and columns of exp(A / tau) in log
# space for N iterations. It is differentiable, and a lower tau makes it sharper. Training
# anneals tau from 1.0 to 0.1 over the warmup.
#
# balanced_round computes the hard form from the soft plan, greedily: every unassigned neuron
# proposes its highest-scoring expert that still has room, proposals are accepted in
# descending score order until each expert is full, and the remaining neurons propose again.
# It always ends with a perfectly balanced partition and asserts that it did.
#
# Training uses the hard partition in the forward pass and the gradient of the soft plan in
# the backward pass (training/blocks.py).

import math

import torch


def log_sinkhorn(A: torch.Tensor, s: int, tau: float, N: int=50) -> torch.Tensor:
    assert A.dtype == torch.float32, 'Affinity tensor must be float32'
    K = A / tau
    log_s = math.log(s)
    log_u = torch.zeros(A.size(0), device=A.device, dtype=A.dtype)
    log_v = torch.full((A.size(1),), log_s, device=A.device, dtype=A.dtype)
    # Alternate log-space updates for unit row mass and capacity-s column mass.
    for _ in range(N):
        log_u = 0.0 - torch.logsumexp(K + log_v[None, :], dim=1)
        log_v = log_s - torch.logsumexp(K + log_u[:, None], dim=0)
    M_soft = torch.exp(K + log_u[:, None] + log_v[None, :])
    return M_soft


@torch.no_grad()
def balanced_round(M_soft: torch.Tensor, s: int):
    (d_ffn, E) = M_soft.shape
    (dev, dt) = (M_soft.device, M_soft.dtype)
    NEG = torch.finfo(dt).min
    n2e = torch.full((d_ffn,), -1, dtype=torch.long, device=dev)
    cap = torch.full((E,), s, dtype=torch.long, device=dev)
    allidx = torch.arange(d_ffn, device=dev)
    guard = 0
    while True:
        um = n2e < 0
        if not bool(um.any()):
            break
        guard += 1
        assert guard <= d_ffn, 'balanced_round: capacity-repair did not converge'
        # Exclude full experts when choosing each unassigned neuron's best candidate.
        scm = M_soft.masked_fill((cap <= 0)[None, :], NEG)
        pick = scm.argmax(dim=1)
        cand = allidx[um]
        cpick = pick[cand]
        cval = M_soft[cand, cpick]
        s_ord = torch.argsort(cval, descending=True)
        sp = cpick[s_ord]
        oh = torch.zeros(sp.numel(), E, device=dev, dtype=torch.long)
        oh[torch.arange(sp.numel(), device=dev), sp] = 1
        # Accept candidates in score order until each expert reaches capacity.
        rank = oh.cumsum(0).gather(1, sp[:, None]).squeeze(1) - 1
        take = rank < cap[sp]
        chosen = cand[s_ord][take]
        chosen_e = sp[take]
        n2e[chosen] = chosen_e
        cap = cap - torch.bincount(chosen_e, minlength=E)
    M = torch.zeros(d_ffn, E, dtype=torch.float32, device=dev)
    M[allidx, n2e] = 1.0
    assert (M.sum(0) == s).all() and (M.sum(1) == 1).all(), 'rounding is not balanced/valid'
    return (M, n2e)
