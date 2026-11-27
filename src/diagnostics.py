# Spectral-MoE: src/diagnostics.py
# Author: Zachary Broveak
#
# Figure data from the training log of the diagnostic run.
#
# During training, the run named by config.DIAGNOSTIC_RUN (Llama-2-7B, learned, dynamic, 25%)
# appends one JSON line per step to tokalloc_log.jsonl in its run directory
# (training/learned.py: record_training). Each line holds the mean routed count of every layer
# over the step's last micro batch, 24 sampled token positions with their per-layer counts,
# and the mean fraction of neurons whose expert assignment changed since the previous step
# (partition churn).
#
# - training_records reads the log and keeps the last line for each step, so lines repeated
#   around a resume are not counted twice.
# - turnover_data converts churn into the turnover curve and its mean over the final 500
#   steps.
# - allocation_data combines the perplexity curves (curve.json) of the uniform, static, and
#   dynamic trace runs with per-layer utilization traces and a sample of per-token schedules
#   from the last logged step, for figures.draw_allocation.
#
# Utilization here counts the shared expert, as defined in config.py.

import json
from pathlib import Path

import numpy as np

from . import config


def training_records(run_dir):
    records = {}
    with (Path(run_dir) / "tokalloc_log.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                # Keep the last record for each step, then sort by step number.
                row = json.loads(line)
                records[row["step"]] = row
    return [records[step] for step in sorted(records)]


def turnover_data(run_dir):
    rows = [r for r in training_records(run_dir) if r.get("partition_churn_mean") is not None]
    last = rows[-1]["step"]
    # Restrict the mean to the final 500 steps and convert fractions to percentages.
    tail = [r["partition_churn_mean"] for r in rows if r["step"] > last - 500]
    return {"steps": [r["step"] for r in rows], "tokens": [r["tokens"] for r in rows],
            "turnover_percent": [100.0 * r["partition_churn_mean"] for r in rows],
            "final_500_mean_percent": 100.0 * sum(tail) / len(tail),
            "final_step_percent": 100.0 * rows[-1]["partition_churn_mean"]}


def allocation_data(runs):
    spec = config.MODELS["llama-2-7b"]
    def utilization(counts):
        # Include shared experts when converting routed counts to utilization.
        return 100.0 * (config.E_SHARED + np.asarray(counts, dtype=float)) / spec["experts"]
    curves = {}
    for method, path in runs.items():
        rows = json.loads((Path(path) / "curve.json").read_text())
        curves[method] = {"tokens": [r["tokens"] for r in rows],
                          "ppl": [r["val_ppl"] for r in rows]}
    logged = training_records(runs["dynamic"])
    counts = np.array([r["layer_k_mean"] for r in logged])
    traces = {"tokens": [r["tokens"] for r in logged],
              "util": utilization(counts).T.tolist(),
              "start": 100.0 * config.realized_utilization("llama-2-7b", .25)}
    sample = list(logged[-1]["sample"])[:config.SAMPLE_CANDIDATES]
    for token, number in config.DROP_REPEATS.items():
        indices = [i for i, row in enumerate(sample) if int(row["tok"]) == token]
        # Drop later duplicate samples while retaining at least one occurrence.
        remove = min(number, max(len(indices) - 1, 0))
        dropped = set(indices[-remove:]) if remove else set()
        sample = [row for i, row in enumerate(sample) if i not in dropped]
    rows = [{"tok": int(s["tok"]), "k": [int(v) for v in s["k"]]}
            for s in sample[:config.SAMPLE_COUNT]]
    return {"curves": curves, "traces": traces, "rows": rows,
            "step": int(logged[-1]["step"]), "budget": int(logged[-1]["budget_target"])}
