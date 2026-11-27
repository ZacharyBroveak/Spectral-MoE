# Spectral-MoE: src/model/__init__.py
# Author: Zachary Broveak
#
# Rebuilding trained runs for evaluation.
#
# Training saves only what it learned (partitions, routers, allocator weights) and refers to
# pinned base-model revisions for everything else. This package reconstructs runnable models
# from those files:
# - restore.py: learned-base arms and the dense baseline, in either the masked or the
#   extracted realization.
# - masked_ffn.py and sparse_experts.py: those two realizations of a learned MoE layer.
# - analytic.py: analytic-base (CMoE-style) arms on top of their recovered base model.
# - router.py: the top-k expert selection used by the learned realizations.
#
# Every restored model carries a spectral_moe dict (kind, blocks, allocator, per-layer counts,
# metadata) that the evaluation code reads.

from .analytic import AnalyticBlock, load_analytic_arm
from .masked_ffn import MaskedFFN
from .restore import load_metadata, load_schedule, restore_arm
from .router import route
from .sparse_experts import SparseExperts

__all__ = ["AnalyticBlock", "MaskedFFN", "SparseExperts", "load_analytic_arm", "load_metadata",
           "load_schedule", "restore_arm", "route"]
