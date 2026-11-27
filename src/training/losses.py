# Spectral-MoE: src/training/losses.py
# Author: Zachary Broveak
#
# Distillation losses for learned conversion.
#
# The converted model (the student) learns to match the original dense model (the teacher) on
# the same tokens:
# - kl_loss: KL(teacher || student) between next-token distributions, averaged over positions;
# - ce_loss: ordinary next-token cross-entropy on the data.
#
# The training loop uses 2 * KL + CE plus small routing regularizers (learned.py).
# kl_ce_chunked computes both terms over chunks of positions and recomputes each chunk during
# the backward pass (activation checkpointing), because float32 logits over a 128k- to
# 152k-token vocabulary take several gigabytes per micro batch. Chunking changes memory use,
# not the result.

import torch
import torch.nn.functional as F


def kl_loss(z_dense: torch.Tensor, z_moe: torch.Tensor) -> torch.Tensor:
    # Weight the log-probability difference by the teacher distribution.
    p = torch.softmax(z_dense.float(), dim=-1)
    logp = torch.log_softmax(z_dense.float(), dim=-1)
    logq = torch.log_softmax(z_moe.float(), dim=-1)
    return (p * (logp - logq)).sum(-1).mean()


def ce_loss(z_moe: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Shift targets one position ahead of the corresponding logits.
    logits = z_moe[:, :-1, :].reshape(-1, z_moe.size(-1)).float()
    tgt = labels[:, 1:].reshape(-1)
    return F.cross_entropy(logits, tgt, ignore_index=-100)


def kl_ce_chunked(z_dense, z_moe, labels, chunks: int=1):
    if z_dense is None:
        return (z_moe.new_zeros(()), ce_loss(z_moe, labels))
    if chunks <= 1:
        return (kl_loss(z_dense, z_moe), ce_loss(z_moe, labels))
    from torch.utils.checkpoint import checkpoint
    V = z_moe.shape[-1]
    (zc_s, tgt) = (z_moe[:, :-1, :].reshape(-1, V), labels[:, 1:].reshape(-1))
    (zd_f, zm_f) = (z_dense.reshape(-1, V), z_moe.reshape(-1, V))

    def _kl_part(a, b):
        p = torch.softmax(a.float(), dim=-1)
        return (p * (torch.log_softmax(a.float(), dim=-1) - torch.log_softmax(b.float(), dim=-1))).sum(-1).sum()

    def _ce_part(a, t):
        return F.cross_entropy(a.float(), t, ignore_index=-100, reduction='sum')
    kl_num = zm_f.new_zeros((), dtype=torch.float32)
    n_kl = zm_f.shape[0]
    # Recompute each KL chunk during backward to reduce stored activations.
    for i in range(0, n_kl, (n_kl + chunks - 1) // chunks):
        j = min(i + (n_kl + chunks - 1) // chunks, n_kl)
        kl_num = kl_num + checkpoint(_kl_part, zd_f[i:j], zm_f[i:j], use_reentrant=False)
    ce_num = zc_s.new_zeros((), dtype=torch.float32)
    # Normalize cross-entropy by the number of unmasked targets.
    n_ce = int((tgt != -100).sum())
    for i in range(0, zc_s.shape[0], (zc_s.shape[0] + chunks - 1) // chunks):
        j = min(i + (zc_s.shape[0] + chunks - 1) // chunks, zc_s.shape[0])
        ce_num = ce_num + checkpoint(_ce_part, zc_s[i:j], tgt[i:j], use_reentrant=False)
    return (kl_num / n_kl, ce_num / max(n_ce, 1))
