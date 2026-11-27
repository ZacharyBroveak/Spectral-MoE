# Spectral-MoE: src/pipeline.py
# Author: Zachary Broveak
#
# Pipeline orchestrator and command-line entry point (python -m src.pipeline, wrapped by the
# Makefile).
#
# User-facing commands:
# - prepare: tokenize and pack the training data for every model.
# - train --arm NAME: prepare what one run needs, then train it.
# - run: train every recipe, refit the routers, evaluate, and report.
# - evaluate: measure every finished run. report: build tables and figures from the
#   measurements.
#
# Each heavy step runs in its own child process through invoke(), using the internal stage
# commands data-stage, convert-stage (analytic construction and LoRA recovery), train-stage,
# and ridge-stage. A fresh process per stage releases all GPU memory between stages and lets
# learned training run under torch.distributed.run with four workers when GPUS=4. With GPUS=1
# the same learned recipe processes its four data shards in sequence on one device (see
# training/learned.py). Evaluation rows are measured the same way through src/evaluate.py, and
# a row is moved into evaluation/ only after its process succeeds, so an interrupted row is
# measured again rather than read half finished.
#
# Output binding. bind_output() writes run.json in the output directory (OUT, by default
# results/) with a hash of requirements.txt and of every .py and .json file under src/, the
# GPU count, and every recipe. Later commands refuse to reuse the directory if any of these
# changed, because resuming saved state under different code or a different device count
# would silently mix two experiments. Every stage detects and skips completed outputs, so
# repeating a command resumes it.
#
# Before any long stage starts, run and train check for enough CUDA devices and for Hugging
# Face access to gated checkpoints (Meta-Llama-3-8B), because data preparation and earlier
# models can take days before a missing permission would otherwise surface.
#
# Router refit rows: checkpoint B is the diagnostic Llama-2-7B dynamic run, whose accuracy
# with its original router is already measured as a benchmark row, so that row is reused and
# annotated with the refit diagnostics instead of being measured twice.

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from . import config, runtime
from .report import report, write_json
from .training.recipes import RECIPES

ROOT = Path(__file__).resolve().parents[1]


def source_digest():
    digest = hashlib.sha256()
    paths = [ROOT / "requirements.txt"] + sorted((ROOT / "src").rglob("*.py"))
    paths += sorted((ROOT / "src").rglob("*.json"))
    # Hash relative names and contents so file renames affect the signature.
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def bind_output(out, gpus):
    signature = {"source_sha256": source_digest(), "gpus": gpus,
                 "recipes": {name: r.record() for name, r in RECIPES.items()}}
    path = out / "run.json"
    # Resume only when the saved source, device count, and recipes agree.
    if path.exists():
        if json.loads(path.read_text()) != signature:
            raise ValueError(f"Source/configuration changed since {out} was created: "
                             "use a new OUT directory")
    elif out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty and has no run manifest; use an empty OUT directory")
    else:
        write_json(signature, path)


def invoke(args, *parts, distributed=False):
    runtime.checkpoint()
    command = [sys.executable]
    # Launch one worker process per GPU for distributed execution.
    if distributed and args.gpus == 4:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4"]
    command += ["-m", "src.pipeline", *parts, "--out", str(args.out), "--gpus", str(args.gpus)]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise SystemExit(runtime.child_status(result.returncode))


def check_model_access(keys):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    for key in keys:
        spec = config.MODELS[key]
        try:
            hf_hub_download(spec["checkpoint"], "config.json", revision=spec["revision"])
        except GatedRepoError:
            raise SystemExit(f"{spec['checkpoint']} requires approved access on Hugging Face: "
                             "request it on the model page, then run `hf auth login`") from None


def ensure_training(args, name):
    recipe = RECIPES[name]
    if not (args.out / "data" / recipe.data_key / "blocks.npy").exists():
        invoke(args, "data-stage", "--data-key", recipe.data_key)
    if recipe.base == "analytic":
        invoke(args, "convert-stage", "--model", recipe.model)
    invoke(args, "train-stage", "--arm", name, distributed=recipe.base == "learned")


def measure(args, command, name, weights, *extra):
    runtime.checkpoint()
    path = args.out / "evaluation" / (name + ".json")
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.json")
    command = [sys.executable, "-m", "src.evaluate", command,
               "--arm", name, "--weights", str(weights), "--out", str(temporary), *map(str, extra)]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise SystemExit(runtime.child_status(result.returncode))
    # Promote the partial result only after the child process succeeds.
    temporary.replace(path)
    return json.loads(path.read_text())


def evaluate(args):
    for arm in config.ALL_ARMS:
        if arm.base == "published":
            continue
        source = (config.arm(arm.model, "learned", "uniform", .25).name
                  if arm.base == "dense" else arm.name)
        extra = []
        if arm.base == "analytic":
            extra = ["--frozen", args.out / "runs" / arm.model / "analytic-base" / "recovered.pt"]
        measure(args, "benchmark", arm.name, args.out / "runs" / source, *extra)
    for row in config.REFIT:
        source = config.REFIT_SOURCES[row.checkpoint]
        ridge = args.out / "runs" / (source + "-ridge")
        weights = ridge if row.router == "refit" else args.out / "runs" / source
        # Reuse the existing accuracy measurements and add selection diagnostics.
        if row.checkpoint == "B" and row.router == "original":
            dest = args.out / "evaluation"
            metrics = json.loads((dest / (source + ".json")).read_text())["metrics"]
            diagnostic = json.loads((ridge / "refit_report.json").read_text())["mean_trained"]
            write_json({"row": row.name, "checkpoint": row.checkpoint, "router": row.router,
                        "metrics": {k: v for k, v in metrics.items() if k != "perplexity"},
                        "overlap": 100.0 * diagnostic["overlap"],
                        "coverage": 100.0 * diagnostic["coverage"]}, dest / (row.name + ".json"))
        else:
            measure(args, "refit", row.name, weights, "--report", ridge)
    report(args.out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "run", "evaluate", "report",
                                            "data-stage", "convert-stage", "train-stage", "ridge-stage"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--gpus", type=int, choices=(1, 4), default=1)
    parser.add_argument("--arm", choices=sorted(RECIPES), metavar="ARM",
                        help="a run name; `make help` lists them")
    parser.add_argument("--model", choices=tuple(config.MODELS))
    parser.add_argument("--data-key")
    parser.add_argument("--session-seconds", type=float, default=0,
                        help="Save and exit 75 after this session duration; 0 disables it")
    args = parser.parse_args(argv)
    if args.command == "train" and not args.arm:
        parser.error("train requires --arm")
    args.out = args.out.resolve()
    if args.command in ("train", "run", "evaluate"):
        import torch
        # Check the GPUs before data preparation, which can take hours.
        needed = 1 if args.command == "evaluate" else args.gpus
        if torch.cuda.device_count() < needed:
            raise SystemExit(f"{args.command} needs {needed} CUDA GPU(s); "
                             f"{torch.cuda.device_count()} found")
    if args.command == "run" or (args.command == "train" and args.arm):
        # Check gated model access first; stages for other models can take days.
        check_model_access(config.MODELS if args.command == "run" else [RECIPES[args.arm].model])
    if args.command in ("prepare", "train", "run", "evaluate", "report"):
        bind_output(args.out, args.gpus)
        runtime.configure(args.out, args.session_seconds)
    if args.command == "prepare":
        # Prepare each distinct dataset key once.
        for key in dict.fromkeys(r.data_key for r in RECIPES.values()):
            invoke(args, "data-stage", "--data-key", key)
    elif args.command == "train":
        ensure_training(args, args.arm)
    elif args.command == "run":
        for name in RECIPES:
            ensure_training(args, name)
        for name in config.REFIT_SOURCES.values():
            invoke(args, "ridge-stage", "--arm", name)
        evaluate(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "report":
        report(args.out)
    elif args.command == "data-stage":
        from .training.data import prepare
        prepare(args.data_key, args.out / "data")
    else:
        import torch
        import torch.distributed as dist
        if not torch.cuda.is_available():
            raise SystemExit("Conversion and training require a CUDA GPU")
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            dist.init_process_group("nccl")
        try:
            if args.command == "convert-stage":
                from .training.analytic import recover
                recover(args.model, args.out / "data", args.out / "runs", device)
            elif args.command == "ridge-stage":
                from .training.refit import refit
                refit(RECIPES[args.arm], args.out / "data", args.out / "runs", device)
            else:
                recipe = RECIPES[args.arm]
                if recipe.base == "learned":
                    from .training.learned import train
                else:
                    from .training.analytic import train
                train(recipe, args.out / "data", args.out / "runs", device)
        except Exception:
            if dist.is_initialized():
                # Exit at once: a collective teardown would wait on ranks still inside a collective.
                traceback.print_exc()
                sys.stderr.flush()
                os._exit(1)
            raise
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
