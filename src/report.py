# Spectral-MoE: src/report.py
# Author: Zachary Broveak
#
# Result tables, figures, and summary metrics from the measured rows.
#
# report() reads one JSON row per arm from results/evaluation/ (written by evaluate.py),
# merges in the published baselines quoted in config.py with their citations, and writes to
# results/report/:
# - benchmarks, utilization, and router_refit tables as JSON and CSV, at full precision;
# - accuracy.pdf, architecture.pdf, allocation.pdf, and turnover.pdf (see figures.py);
# - allocation.json and turnover.json, the data behind the last two figures (see
#   diagnostics.py);
# - summary.json: model geometry, the dynamic allocator's parameter count, and the aggregate
#   comparisons, for example the perplexity reduction of dynamic over uniform allocation for
#   each model and the accuracy differences to published baselines.
#
# The dense row's perplexity is measured locally while its accuracy is quoted from DOT-MoE,
# and each row's provenance field records which values were measured and which were quoted.
#
# write_json() is the atomic JSON writer used across the project. It writes a temporary file
# and renames it over the destination, so readers never see a partial file and a crash never
# truncates a good one. It also rejects NaN and infinity, so a diverged metric fails loudly
# instead of being recorded.

import csv
import json
from pathlib import Path

from . import config


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace the destination only after the complete JSON has been written.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summary_values(rows, refits, allocation, turnover):
    from .allocator_dynamic import DynamicAllocator

    def metric(model, base, method, utilization, key):
        return rows[config.arm(model, base, method, utilization).name][key]
    def drops(base, utilization=.25):
        return {m: metric(m, base, "uniform", utilization, "perplexity")
                - metric(m, base, "dynamic", utilization, "perplexity") for m in config.MS}
    def gains(base, utilization=.25):
        return {m: metric(m, base, "dynamic", utilization, "average")
                - metric(m, base, "uniform", utilization, "average") for m in config.MS}
    def span(values):
        return [min(values.values()), max(values.values())]
    # Pair the original and refit metrics within each seed.
    by_seed = {seed: {r["router"]: r for r in refits if r["checkpoint"] == seed}
               for seed in config.REFIT_SOURCES}
    spec = config.MODELS["llama-2-7b"]
    allocator = DynamicAllocator(d_model=spec["d_model"], n_layers=spec["layers"],
                                 k_init=config.e_bar("llama-2-7b", .25),
                                 total=config.budget("llama-2-7b", .25), k_min=config.E_MIN,
                                 k_max=spec["experts"] - config.E_SHARED,
                                 n_experts=spec["experts"] - config.E_SHARED)
    return {
        "geometry": {m: {"layers": s["layers"], "neurons_per_expert": config.NEURONS_PER_EXPERT,
                         "total_experts": s["experts"], "shared_experts": config.E_SHARED,
                         "routed_experts": s["experts"] - config.E_SHARED,
                         "minimum_routed_count": config.E_MIN,
                         "uniform_routed_count_at_25_percent": config.e_bar(m, .25),
                         "routed_budget_at_25_percent": config.budget(m, .25),
                         "realized_utilization_percent": 100 * config.realized_utilization(m, .25)}
                     for m, s in config.MODELS.items()},
        "llama2_dynamic_allocator_parameters": sum(p.numel() for p in allocator.parameters()),
        "learned_ppl_reductions_at_25_percent": drops("learned"),
        "analytic_ppl_reductions_at_25_percent": drops("analytic"),
        "accuracy_gain_ranges_at_25_percent": {b: span(gains(b)) for b in ("analytic", "learned")},
        "perplexity_reduction_ranges_at_25_percent": {b: span(drops(b)) for b in ("analytic", "learned")},
        "analytic_accuracy_gain_over_published_cmoe": {
            m: metric(m, "analytic", "dynamic", .25, "average")
            - metric(m, "published", "cmoe", .25, "average") for m in config.MS[:2]},
        "learned_accuracy_gain_over_published_dotmoe": {
            m: metric(m, "learned", "dynamic", .25, "average")
            - metric(m, "published", "dot-moe", .25, "average") for m in config.MS[:2]},
        "qwen_accuracy_at_25_percent": {
            "dynamic": metric("qwen2.5-7b", "learned", "dynamic", .25, "average"),
            "published_dotmoe": metric("qwen2.5-7b", "published", "dot-moe", .25, "average")},
        # Average model-specific gains equally within each utilization budget.
        "sweep": {str(u): {"mean_accuracy_gain": sum(gains("learned", u).values()) / len(config.MS),
                           "perplexity_reduction_range": span(drops("learned", u))}
                  for u in config.UTILIZATIONS},
        "allocation_final_perplexity": {method: curve["ppl"][-1]
                                     for method, curve in allocation["curves"].items()},
        "allocation_uniform_start_percent": allocation["traces"]["start"],
        "allocation_displayed_tokens": len(allocation["rows"]),
        "turnover_final_500_steps_percent": turnover["final_500_mean_percent"],
        "turnover_final_step_percent": turnover["final_step_percent"],
        "refit_calibration_tokens": config.REFIT_FIT_TOKENS,
        "refit_accuracy_changes_pp": {seed: row["refit"]["average"] - row["original"]["average"]
                                     for seed, row in by_seed.items()},
    }


def report(out):
    from transformers import AutoTokenizer

    from .diagnostics import allocation_data, turnover_data
    from .figures import (
        draw_accuracy,
        draw_allocation,
        draw_architecture,
        draw_turnover,
    )

    out = Path(out)
    dest = out / "report"

    def read(name):
        path = out / "evaluation" / (name + ".json")
        if not path.exists():
            raise SystemExit(f"{path} is missing: report needs every evaluated row, so run "
                             "`make evaluate` with the same OUT first")
        return json.loads(path.read_text())

    rows = {}
    for arm in config.ALL_ARMS:
        if arm.base == "published":
            metrics = {"perplexity": None, **arm.accuracy, "average": arm.average}
            provenance = arm.citation
        else:
            metrics = read(arm.name)["metrics"]
            provenance = "measured"
            if arm.base == "dense":
                metrics = {**metrics, **arm.accuracy, "average": arm.average}
                provenance = "perplexity measured; accuracy quoted from " + arm.citation
        rows[arm.name] = {"row": arm.name, "provenance": provenance, **metrics}
    refits = []
    for row in config.REFIT:
        record = read(row.name)
        refits.append({"row": row.name, "checkpoint": row.checkpoint, "router": row.router,
                       "overlap": record["overlap"], "coverage": record["coverage"],
                       **record["metrics"]})
    tables = {"benchmarks": [rows[a.name] for a in config.BENCHMARK_RUNS],
              "utilization": [rows[a.name] for a in config.UTILIZATION_RUNS], "router_refit": refits}
    # Create the report directory only after every input row is present.
    dest.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        write_json(table, dest / (name + ".json"))
        # Preserve column order while including keys absent from some rows.
        fields = list(dict.fromkeys(key for row in table for key in row))
        with (dest / (name + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
    averages = {a.name: rows[a.name]["average"] for a in config.ALL_ARMS if a.base == "learned"}
    draw_accuracy(dest / "accuracy.pdf", measured=averages)
    draw_architecture(dest / "architecture.pdf")
    allocation = allocation_data({method: out / "runs" / name for method, name in config.TRACE_RUNS.items()})
    spec = config.MODELS["llama-2-7b"]
    tokenizer = AutoTokenizer.from_pretrained(spec["checkpoint"], revision=spec["revision"])
    for row in allocation["rows"]:
        row["token"] = tokenizer.decode([row["tok"]]).strip()
    draw_allocation(allocation, dest / "allocation.pdf", tokenizer=tokenizer)
    write_json(allocation, dest / "allocation.json")
    turnover = turnover_data(out / "runs" / config.DIAGNOSTIC_RUN)
    draw_turnover(turnover, dest / "turnover.pdf")
    write_json(turnover, dest / "turnover.json")
    summary = summary_values(rows, refits, allocation, turnover)
    # Count distinct schedules among positions with the same decoded token.
    summary["allocation_on_distinct_schedules"] = len({tuple(row["k"]) for row in allocation["rows"]
                                                 if row["token"] == "on"})
    write_json(summary, dest / "summary.json")
    print(f"Wrote metrics and plots to {dest}")
