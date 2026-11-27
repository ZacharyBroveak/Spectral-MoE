# Spectral-MoE: src/config.py
# Author: Zachary Broveak
#
# Experiment definition: models, expert geometry, budgets, and every result row.
#
# This module is the single source of truth for what is measured. Recipes, the pipeline,
# evaluation, and the report all read from it, and model checkpoints and revisions are pinned
# so downloads are reproducible.
#
# Geometry. An FFN of width d_ffn is cut into experts = d_ffn / 128 experts of
# NEURONS_PER_EXPERT = 128 neurons. E_SHARED = 1 of them is a shared, always-active expert and
# the rest are routed. A utilization u is the fraction of FFN neurons a token uses.
# e_bar(model, u) is the uniform routed count per layer, u * experts - E_SHARED rounded half
# up, and budget(model, u) = layers * e_bar is the total number of routed experts one token
# may use across the whole network. Allocators may move experts between layers (static), or
# between layers separately for every token (dynamic), but always spend exactly this budget,
# with at least E_MIN routed experts in every layer. realized_utilization counts the shared
# expert: Llama-2-7B at 25% runs 21 routed experts plus 1 shared expert out of 86, which is
# 25.6%.
#
# Rows. An Arm is one row of a results table: a model; a base, which is "learned" or
# "analytic" conversion, "dense" for the unconverted model, or "published" for numbers quoted
# from another paper with a citation; an allocator ("uniform", "static", or "dynamic"); a
# utilization; and a training step count. Arm.name, for example
# llama-2-7b/learned/dynamic/u0250 (utilization times 1000), is also the run directory under
# results/runs and the ARM value accepted by make train. BENCHMARK_RUNS form the main comparison
# at 25% utilization, UTILIZATION_RUNS the 12.5%, 25%, and 50% sweep, and ALL_ARMS their union
# without duplicates.
#
# The remaining constants configure the benchmark suite (task, few-shot count, metric), router
# refitting (which checkpoints, how many tokens, ridge strength), and the report figures.
# TRACE_RUNS are the runs whose perplexity is logged during training for the convergence plot.
# DROP_REPEATS lists Llama-2 token ids (29892 is ",") whose extra occurrences are dropped from
# the per-token schedule figure so the displayed samples are more varied.

import math
from dataclasses import dataclass, field

MODELS = {'llama-2-7b': {'d_model': 4096,
                'd_ffn': 11008,
                'layers': 32,
                'experts': 86,
                'checkpoint': 'NousResearch/Llama-2-7b-hf',
                'revision': '8efe6c9b93655b934e27bd9981e3ec13e55aee9d'},
 'llama-3-8b': {'d_model': 4096,
                'd_ffn': 14336,
                'layers': 32,
                'experts': 112,
                'checkpoint': 'meta-llama/Meta-Llama-3-8B',
                'revision': '8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920'},
 'qwen2.5-7b': {'d_model': 3584,
                'd_ffn': 18944,
                'layers': 28,
                'experts': 148,
                'checkpoint': 'Qwen/Qwen2.5-7B',
                'revision': 'd149729398750b98c0af14eb82c78cfe92750796'}}


MS = ('llama-2-7b', 'llama-3-8b', 'qwen2.5-7b')

NEURONS_PER_EXPERT = 128

E_SHARED = 1

E_MIN = 1

UTILIZATIONS = (0.125, 0.25, 0.5)

REASONING_SUITE = (('boolq', 32, 'acc'),
 ('sciq', 0, 'acc'),
 ('piqa', 0, 'acc'),
 ('winogrande', 5, 'acc'),
 ('arc_challenge', 25, 'acc_norm'),
 ('hellaswag', 10, 'acc_norm'))

PPL_BATCH = {'learned': 8, 'dense': 8, 'analytic': 1}

TASKS = tuple(task for task, _, _ in REASONING_SUITE)


def e_bar(model, utilization):
    # Round half-integers upward.
    return math.floor(utilization * MODELS[model]["experts"] - E_SHARED + 0.5)


def budget(model, utilization):
    # Multiply the uniform routed count by the number of layers.
    return MODELS[model]["layers"] * e_bar(model, utilization)


def realized_utilization(model, utilization):
    # Include shared experts in the active fraction.
    return (e_bar(model, utilization) + E_SHARED) / MODELS[model]["experts"]


BUDGETS = {model: {u: e_bar(model, u) for u in UTILIZATIONS} for model in MS}


@dataclass(frozen=True)
class Arm:
    model: str
    base: str
    allocator: str
    utilization: float
    steps: int = 3500

    accuracy: dict = field(default_factory=dict)
    average: float | None = None
    citation: str = ""

    @property
    def name(self):
        return f"{self.model}/{self.base}/{self.allocator}/u{int(self.utilization * 1000):04d}"


def _quoted(model, method, accuracy, average, citation):
    return Arm(model, "dense" if method == "dense" else "published", method,
               1.0 if method == "dense" else .25, 0,
               dict(zip(TASKS, accuracy)), average, citation)


BENCHMARK_RUNS = [
    _quoted('llama-2-7b', 'dense',
            [82.0, 94.0, 78.1, 74.3, 52.5, 78.9], 76.6, 'DOT-MoE, Table 3 (dense row)'),
    _quoted('llama-2-7b', 'llama-moe',
            [37.83, 20.0, 49.73, 50.12, 25.79, 26.18], 34.94166666666666, 'CMoE, arXiv v2, Table 2'),
    _quoted('llama-2-7b', 'llama-moe-v2',
            [51.25, 67.0, 56.64, 52.88, 25.68, 35.1], 48.09166666666667, 'CMoE, arXiv v2, Table 2'),
    _quoted('llama-2-7b', 'cmoe',
            [55.04, 77.5, 57.12, 54.06, 27.56, 38.79], 51.678333333333335, 'CMoE, arXiv v2, Table 2'),
    Arm('llama-2-7b', 'analytic', 'uniform', 0.25, 0),
    Arm('llama-2-7b', 'analytic', 'static', 0.25, 1024),
    Arm('llama-2-7b', 'analytic', 'dynamic', 0.25, 1024),
    _quoted('llama-2-7b', 'dot-moe',
            [66.4, 89.6, 67.2, 57.9, 34.0, 53.9], 61.5, 'DOT-MoE, Table 3'),
    Arm('llama-2-7b', 'learned', 'uniform', 0.25, 3500),
    Arm('llama-2-7b', 'learned', 'static', 0.25, 3500),
    Arm('llama-2-7b', 'learned', 'dynamic', 0.25, 3500),
    _quoted('llama-3-8b', 'dense',
            [83.2, 96.2, 79.6, 77.3, 58.3, 82.1], 79.4, 'DOT-MoE, Table 3 (dense row)'),
    _quoted('llama-3-8b', 'llama-moe',
            [37.82, 20.3, 49.02, 50.74, 25.68, 25.76], 34.88666666666667, 'CMoE, arXiv v2, Table 2'),
    _quoted('llama-3-8b', 'cmoe',
            [56.88, 72.0, 57.34, 51.93, 25.77, 36.68], 50.1, 'CMoE, arXiv v2, Table 2'),
    Arm('llama-3-8b', 'analytic', 'uniform', 0.25, 0),
    Arm('llama-3-8b', 'analytic', 'static', 0.25, 1024),
    Arm('llama-3-8b', 'analytic', 'dynamic', 0.25, 1024),
    _quoted('llama-3-8b', 'dot-moe',
            [62.3, 90.9, 65.0, 56.0, 34.1, 50.1], 59.8, 'DOT-MoE, Table 3'),
    Arm('llama-3-8b', 'learned', 'uniform', 0.25, 3500),
    Arm('llama-3-8b', 'learned', 'static', 0.25, 3500),
    Arm('llama-3-8b', 'learned', 'dynamic', 0.25, 3500),
    _quoted('qwen2.5-7b', 'dense',
            [88.0, 96.6, 78.6, 76.2, 64.1, 80.3], 80.6, 'DOT-MoE, Table 3 (dense row)'),
    Arm('qwen2.5-7b', 'analytic', 'uniform', 0.25, 0),
    Arm('qwen2.5-7b', 'analytic', 'static', 0.25, 1024),
    Arm('qwen2.5-7b', 'analytic', 'dynamic', 0.25, 1024),
    _quoted('qwen2.5-7b', 'dot-moe',
            [64.3, 93.9, 68.5, 58.7, 39.6, 55.0], 63.3, 'DOT-MoE, Table 3'),
    Arm('qwen2.5-7b', 'learned', 'uniform', 0.25, 3500),
    Arm('qwen2.5-7b', 'learned', 'static', 0.25, 3500),
    Arm('qwen2.5-7b', 'learned', 'dynamic', 0.25, 3500),
]

UTILIZATION_RUNS = [
    Arm('llama-2-7b', 'learned', 'uniform', 0.125, 3500),
    Arm('llama-2-7b', 'learned', 'dynamic', 0.125, 3500),
    Arm('llama-3-8b', 'learned', 'uniform', 0.125, 3500),
    Arm('llama-3-8b', 'learned', 'dynamic', 0.125, 3500),
    Arm('qwen2.5-7b', 'learned', 'uniform', 0.125, 3500),
    Arm('qwen2.5-7b', 'learned', 'dynamic', 0.125, 3500),
    Arm('llama-2-7b', 'learned', 'uniform', 0.25, 3500),
    Arm('llama-2-7b', 'learned', 'dynamic', 0.25, 3500),
    Arm('llama-3-8b', 'learned', 'uniform', 0.25, 3500),
    Arm('llama-3-8b', 'learned', 'dynamic', 0.25, 3500),
    Arm('qwen2.5-7b', 'learned', 'uniform', 0.25, 3500),
    Arm('qwen2.5-7b', 'learned', 'dynamic', 0.25, 3500),
    Arm('llama-2-7b', 'learned', 'uniform', 0.5, 3500),
    Arm('llama-2-7b', 'learned', 'dynamic', 0.5, 3500),
    Arm('llama-3-8b', 'learned', 'uniform', 0.5, 3500),
    Arm('llama-3-8b', 'learned', 'dynamic', 0.5, 3500),
    Arm('qwen2.5-7b', 'learned', 'uniform', 0.5, 3500),
    Arm('qwen2.5-7b', 'learned', 'dynamic', 0.5, 3500),
 ]
# Deduplicate configurations by name while preserving insertion order.
ALL_ARMS = list({a.name: a for a in BENCHMARK_RUNS + UTILIZATION_RUNS}.values())


def arms():
    return {a.name: a for a in ALL_ARMS}


def arm(model, base, allocator, utilization):
    return next(a for a in ALL_ARMS if (a.model, a.base, a.allocator, a.utilization)
                == (model, base, allocator, utilization))


DIAGNOSTIC_RUN = "llama-2-7b/learned/dynamic/u0250"
REFIT_SOURCES = {"A": "refit/a", "B": DIAGNOSTIC_RUN}
REFIT_FIT_TOKENS = 32768
REFIT_DIAGNOSTIC_TOKENS = 16384
REFIT_RIDGE = 1e-3


@dataclass(frozen=True)
class RefitRow:
    checkpoint: str
    router: str

    @property
    def name(self):
        return f"refit/{self.checkpoint.lower()}/{self.router}"


REFIT = [RefitRow(seed, router) for seed in REFIT_SOURCES
         for router in ("original", "refit")]


ACCURACY_BUDGETS = (.50, .25, .125)
ACCURACY_BASELINES = (("cmoe", ("llama-2-7b", "llama-3-8b")), ("dot-moe", MS))
TRACE_RUNS = {method: arm("llama-2-7b", "learned", method, .25).name
             for method in ("uniform", "static", "dynamic")}
PLOT_START_TOKENS = 25_000_000
SMOOTHING_STEPS = 61

SAMPLE_CANDIDATES = 18
SAMPLE_COUNT = 16
DROP_REPEATS = {29892: 2}
