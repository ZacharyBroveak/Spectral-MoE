# Spectral-MoE: src/training/recipes.py
# Author: Zachary Broveak
#
# Training recipes: the full hyperparameter set of every run.
#
# recipes() derives one Recipe for each learned or analytic arm in config.py, plus refit/a, a
# copy of the diagnostic Llama-2-7B dynamic run with a different seed that provides the second
# checkpoint for router refitting. A recipe's record() is stored in the run manifest
# (run.json) and in every resume checkpoint, so changing any value here requires a fresh OUT
# directory.
#
# Learned conversion (training/learned.py): 3500 steps of 64 sequences of 2048 tokens, which
# is 458,752,000 tokens, split across training_world = 4 data shards with micro batches of 2
# sequences (1 for Qwen2.5-7B). The partition affinities and routers train with AdamW at
# learning_rate, warmup and cosine decay, and weight decay 1e-4. loss_chunks > 1 computes the
# distillation loss in chunks to save memory with the large Llama-3 and Qwen vocabularies.
#
# Analytic conversion (training/analytic.py): the converted model stays frozen and only the
# allocator trains, for 1024 steps of 2 sequences on one GPU.
#
# Allocators: allocator_lr scales with the uniform routed count k, so larger budgets take
# proportionally larger steps in units of experts. Static allocation on learned arms uses a
# softer inclusion temperature of 1.5. init_spread is the initial spread of the dynamic
# allocator's plan logits (see allocator_dynamic.py) and is set per model and utilization, so
# dynamic training starts from an essentially uniform allocation.

from dataclasses import asdict, dataclass, replace

from ..config import BENCHMARK_RUNS, BUDGETS, UTILIZATION_RUNS


@dataclass(frozen=True)
class Recipe:
    name: str
    model: str
    base: str
    allocator: str
    utilization: float
    seed: int = 17
    steps: int = 3500
    global_batch: int = 64
    micro_batch: int = 2
    training_world: int = 4
    data_seed: int = 1234
    learning_rate: float = 5e-4
    allocator_lr: float = 0.0
    temperature: float = 1.0
    init_spread: float = 0.001
    weight_decay: float = 1e-4
    loss_chunks: int = 1

    @property
    def k(self):
        return BUDGETS[self.model][self.utilization]

    @property
    def data_key(self):
        return self.model

    def record(self):
        return asdict(self)


def recipes():
    out = {}
    spreads = {"llama-2-7b": {0.125: .002, .25: .001, .5: .0005},
               "llama-3-8b": {0.125: .002, .25: .0005, .5: .0002},
               "qwen2.5-7b": {0.125: .001, .25: .0005, .5: .0002}}
    for arm in BENCHMARK_RUNS + UTILIZATION_RUNS:
        if arm.base not in ("learned", "analytic"):
            continue
        k = BUDGETS[arm.model][arm.utilization]
        analytic = arm.base == "analytic"
        static = arm.allocator == "static"
        out[arm.name] = Recipe(
            arm.name, arm.model, arm.base, arm.allocator, arm.utilization,
            steps=arm.steps, global_batch=2 if analytic else 64,
            micro_batch=1 if analytic or arm.model == "qwen2.5-7b" else 2,
            training_world=1 if analytic else 4,
            allocator_lr=(1.75 * k / 1024 if analytic and static else
                          5e-3 * k / 10 if static else 5e-5 * k / 10),
            temperature=1.5 if static and not analytic else 1.0,
            init_spread=((.0005 if arm.model == "qwen2.5-7b" else .001) if analytic
                         else spreads[arm.model][arm.utilization]),
            weight_decay=0.0 if analytic else 1e-4,
            loss_chunks=1 if analytic or arm.model == "llama-2-7b" else 8)
    # Copy the configuration with a new run name and initialization seed.
    b = out["llama-2-7b/learned/dynamic/u0250"]
    out["refit/a"] = replace(b, name="refit/a", seed=13)
    return out


RECIPES = recipes()
