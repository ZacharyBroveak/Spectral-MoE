# Spectral-MoE: src/evaluation/__init__.py
# Author: Zachary Broveak
#
# Evaluation primitives: the WikiText-2 perplexity corpus (corpus.py), sharded perplexity
# (perplexity.py), the six-task benchmark suite (suite.py), and the pinned dataset revisions
# behind them (revisions.py).

from .corpus import token_blocks
from .perplexity import perplexity
from .suite import run_suite

__all__ = ["perplexity", "run_suite", "token_blocks"]
