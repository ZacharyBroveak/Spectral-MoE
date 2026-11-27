# Spectral-MoE: src/evaluate.py
# Author: Zachary Broveak
#
# Measure one result row: python -m src.evaluate {benchmark,refit} --arm NAME --weights DIR
# --out FILE.
#
# The pipeline launches this once per row in its own process (pipeline.py: measure), so every
# row starts with an empty GPU and a failure affects only that row.
#
# - benchmark: perplexity on WikiText-2 and accuracy on the six-task suite. Learned arms use
#   the masked realization for perplexity and the extracted realization for accuracy (see
#   model/). Analytic arms load their allocator over the recovered base given by --frozen. The
#   dense row measures perplexity only; its accuracy is quoted in the report.
# - refit: benchmark accuracy of a router-refit checkpoint, annotated with the router's
#   overlap and coverage from the refit report (training/refit.py).
#
# measure() is resumable at the level of individual metrics. Finished metrics and benchmark
# groups are saved in .progress.json and .suite.json files next to the output, and a restart
# skips them; both files are deleted once the complete row is written. The progress file
# records the size and modification time of every input weight file, so metrics measured on
# old weights are never mixed with new ones. Accuracies are stored as percentages, with an
# unweighted average over the six tasks.

import argparse
import gc
import json
from pathlib import Path

from . import config, runtime
from .report import write_json


def measure(args, build_ppl, build_suite, *, with_ppl=True, with_accuracy=True,
            ppl_batch=8, extra=None):
    import torch

    from .evaluation import perplexity, run_suite, token_blocks

    progress_path = args.out.with_suffix(".progress.json")
    suite_path = args.out.with_suffix(".suite.json")
    inputs = [args.weights] + ([args.frozen] if getattr(args, "frozen", None) else [])
    signature = {"row": args.arm, "inputs": {}}
    for path in inputs:
        files = sorted(path.glob("*")) if path.is_dir() else [path]
        for item in files:
            if item.is_file():
                # Record file size and modification time to detect changed inputs.
                stat = item.stat()
                signature["inputs"][str(item.resolve())] = [stat.st_size, stat.st_mtime_ns]
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"signature": signature, "metrics": {}}
    if progress["signature"] != signature:
        raise ValueError("Evaluation inputs changed; use a fresh output path")
    metrics = progress["metrics"]
    write_json(progress, progress_path)
    runtime.checkpoint()
    model = tok = None
    # Skip perplexity computation when a saved value is already present.
    if with_ppl and "perplexity" not in metrics:
        model, tok = build_ppl()
        shards = int(model.spectral_moe["meta"].get("world", 1))
        blocks = token_blocks(tok, split="test", block_size=2048)
        metrics["perplexity"] = perplexity(
            model, blocks, args.device, batch_size=ppl_batch, shards=shards)
        write_json(progress, progress_path)
        runtime.checkpoint()
    if with_accuracy:
        if model is None or build_suite is not build_ppl:
            # Release the current model before constructing the benchmark model.
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model, tok = build_suite()
        suite = run_suite(model, tok, batch_size=8, device=args.device,
                          progress_path=suite_path)
        # Convert task scores to percentages and weight all tasks equally.
        for task in config.TASKS:
            metrics[task] = 100.0 * suite[task]["value"]
        metrics["average"] = sum(metrics[t] for t in config.TASKS) / len(config.TASKS)
    record = {"row": args.arm, "metrics": metrics, **(extra or {})}
    write_json(record, args.out)
    # The complete row supersedes the resume files.
    progress_path.unlink(missing_ok=True)
    suite_path.unlink(missing_ok=True)
    print(json.dumps(record, indent=2))


def benchmark_row(args):
    from .model import load_analytic_arm, restore_arm

    row = config.arms()[args.arm]
    if row.base == "published":
        raise ValueError("Published baseline rows are quoted in the report")
    if row.base == "analytic":
        if args.frozen is None:
            raise ValueError("An analytic row needs the locally recovered base (--frozen)")
        def build():
            model, tok, _ = load_analytic_arm(
                args.weights, args.frozen, config.MODELS[row.model]["revision"], device=args.device)
            return model, tok
        return measure(args, build, build, ppl_batch=config.PPL_BATCH["analytic"])
    if row.base == "dense":
        def build():
            return restore_arm(args.weights, base="dense", device=args.device, expect_steps=None)
        return measure(args, build, build, with_accuracy=False,
                       ppl_batch=config.PPL_BATCH["dense"])
    def masked():
        return restore_arm(args.weights, device=args.device, expect_steps=row.steps,
                           realization="masked")
    def extracted():
        return restore_arm(args.weights, device=args.device, expect_steps=row.steps,
                           realization="extracted")
    return measure(args, masked, extracted, ppl_batch=config.PPL_BATCH["learned"])


def refit_row(args):
    from .model import restore_arm

    row = next(row for row in config.REFIT if row.name == args.arm)
    report = json.loads((args.report / "refit_report.json").read_text())
    diagnostics = report["mean_trained" if row.router == "original" else "mean_refit"]
    def extracted():
        return restore_arm(args.weights, device=args.device, expect_steps=3500,
                           realization="extracted")
    measure(args, extracted, extracted, with_ppl=False,
            extra={"checkpoint": row.checkpoint, "router": row.router,
                   "overlap": 100.0 * diagnostics["overlap"],
                   "coverage": 100.0 * diagnostics["coverage"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("benchmark", "refit"))
    parser.add_argument("--arm", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    (benchmark_row if args.command == "benchmark" else refit_row)(args)


if __name__ == "__main__":
    main()
