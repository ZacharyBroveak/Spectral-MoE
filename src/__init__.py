# Spectral-MoE: src/__init__.py
# Author: Zachary Broveak
#
# Spectral-MoE source package.
#
# Spectral-MoE converts a pretrained dense decoder (Llama-2-7B, Llama-3-8B, or Qwen2.5-7B)
# into a mixture-of-experts (MoE) model. Each feed-forward (FFN) layer is split into experts
# of 128 neurons, one of which is shared and always active, and a router picks the other
# experts per token. On top of that conversion, an allocator learns how many routed experts
# each layer should run under a fixed per-token budget: the same count in every layer
# (uniform), a learned count per layer (static), or a learned count per layer and per token
# (dynamic).
#
# The package is run as modules from the repository root (python -m src.pipeline ..., usually
# through the Makefile), so it uses relative imports throughout. Layout:
#
# - pipeline.py, runtime.py, config.py: orchestration, pause and resume, and the experiment
#   definition.
# - training/: data packing and the two conversion methods. Learned conversion trains a
#   neuron-to-expert partition and a router by distillation (learned.py, blocks.py,
#   assignment.py, losses.py). Analytic conversion builds experts by clustering activation
#   patterns, following CMoE, and then trains only the allocator (analytic.py,
#   analytic_blocks.py, analytic_primitives/). refit.py and ridge.py refit routers in closed
#   form.
# - allocator_static.py, allocator_dynamic.py, soft_inclusion.py: the static and dynamic
#   allocators and the differentiable relaxation that trains them.
# - model/: rebuilds trained runs from their saved files for evaluation.
# - evaluate.py, evaluation/: perplexity and benchmark measurement.
# - report.py, diagnostics.py, figures.py, assets/: result tables, figures, and summary
#   metrics.
