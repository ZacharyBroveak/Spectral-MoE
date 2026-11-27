# Spectral-MoE: src/training/ridge.py
# Author: Zachary Broveak
#
# Ridge regression and selection scoring for router refitting (refit.py).
#
# - ridge_fit solves min ||(X - mean X) W - (Y - mean Y)||^2 + lam * c * ||W||^2 in closed
#   form. Centering leaves the intercept unpenalized, and c, the mean diagonal of the centered
#   Gram matrix (at least 1), makes lam relative to the feature energy.
# - row_standardise turns each token's expert importances into z-scores across experts, so the
#   fit targets the ranking rather than the magnitude.
# - score_sel compares a predicted ranking with the oracle at each token's own count k and
#   returns the mean overlap and coverage defined in refit.py.

import torch


def ridge_fit(Phi, Y, lam):
    # Center inputs and labels so the intercept remains unpenalized.
    mu = Phi.mean(0, keepdim=True)
    ybar = Y.mean(0, keepdim=True)
    P = Phi - mu
    G = P.t() @ P
    # Scale the diagonal penalty by the mean feature energy.
    G.diagonal().add_(lam * max(1.0, float(G.diagonal().mean())))
    W = torch.linalg.solve(G, P.t() @ (Y - ybar))
    return (W, mu, ybar)


def row_standardise(S):
    # Standardize each token across experts using the sample standard deviation.
    return (S - S.mean(1, keepdim=True)) / S.std(1, keepdim=True).clamp_min(1e-06)


def score_sel(pred, orc, k):
    assert float(orc.min()) >= 0.0, "Coverage requires nonnegative importance scores"
    kmax = int(k.max())
    pi = pred.topk(kmax, dim=1).indices
    oi = orc.topk(kmax, dim=1).indices
    # Compare both rankings at the same expert count for each token.
    keep = torch.arange(kmax).unsqueeze(0) < k.unsqueeze(1)
    Ps = torch.zeros_like(pred, dtype=torch.bool).scatter_(1, pi, keep)
    Os = torch.zeros_like(orc, dtype=torch.bool).scatter_(1, oi, keep)
    inter = (Ps & Os).sum(1).float()
    ov = (inter / k.float()).mean().item()
    # Normalize selected importance by the reference selection's importance.
    num = (orc * Ps).sum(1)
    den = (orc * Os).sum(1).clamp_min(1e-09)
    return (ov, (num / den).mean().item())
