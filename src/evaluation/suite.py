# Spectral-MoE: src/evaluation/suite.py
# Author: Zachary Broveak
#
# The six-task reasoning benchmark, run through lm-evaluation-harness.
#
# Tasks, few-shot counts, and reported metrics come from config.REASONING_SUITE: BoolQ
# (32-shot, acc), SciQ (0-shot, acc), PIQA (0-shot, acc), WinoGrande (5-shot, acc),
# ARC-Challenge (25-shot, acc_norm), and HellaSwag (10-shot, acc_norm). Tasks with the same
# shot count run together in one simple_evaluate call on the in-memory model (lm-eval's HFLM
# wrapper), with fixed seeds for choosing few-shot examples. Results are saved after every
# group, so an interrupted run skips the groups already finished. Values are returned as
# fractions; evaluate.py converts them to percentages.

import json
from collections import defaultdict
from pathlib import Path

from .. import runtime
from ..config import REASONING_SUITE
from ..report import write_json
from .revisions import TASK_REVISIONS, task_configs


def run_suite(model, tokenizer, batch_size=8, device="cuda:0", progress_path=None):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    # Group tasks that use the same number of prompt examples.
    by_shots = defaultdict(list)
    for task, shots, _ in REASONING_SUITE:
        by_shots[shots].append(task)
    progress_path = Path(progress_path) if progress_path else None
    results = (json.loads(progress_path.read_text())
               if progress_path and progress_path.exists() else {})
    for shots, tasks in sorted(by_shots.items()):
        # Skip a group only when every task already has a saved result.
        if all(task in results for task in tasks):
            continue
        runtime.checkpoint()
        result = simple_evaluate(model=lm, tasks=task_configs(tasks), num_fewshot=shots,
                                 bootstrap_iters=0, log_samples=False,
                                 random_seed=0, numpy_random_seed=1234,
                                 torch_random_seed=1234, fewshot_random_seed=1234)
        for task, count, metric in REASONING_SUITE:
            if task not in tasks:
                continue
            # Accept metric keys with or without a filter suffix.
            metrics = result["results"][task]
            key = f"{metric},none" if f"{metric},none" in metrics else metric
            results[task] = {"value": float(metrics[key]), "metric": metric, "num_fewshot": count,
                             "dataset_revision": TASK_REVISIONS[task][1]}
        if progress_path:
            write_json(results, progress_path)
        print(f"Completed benchmark group: {', '.join(tasks)}", flush=True)
        runtime.checkpoint()
    results["avg"] = sum(results[task]["value"] for task, _, _ in REASONING_SUITE) / len(REASONING_SUITE)
    return results
