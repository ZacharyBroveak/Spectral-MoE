# Spectral-MoE: src/training/learned.py
# Author: Zachary Broveak
#
# Learned conversion: distill a frozen dense model into a masked MoE.
#
# train(recipe) runs one learned arm:
# 1. Load the pinned base model in bfloat16, freeze it, and wrap every FFN in a TrainingFFN
#    (blocks.py). Pick each layer's shared neurons on two calibration blocks and build the
#    allocator the recipe names.
# 2. Each step accumulates 64 sequences over training_world data shards. Every micro batch
#    runs the model twice: with masks off for the dense teacher logits (no gradient), and with
#    masks on for the student. The loss is 2 * KL + CE (losses.py) plus a router z-loss
#    (weight 1e-3) and a load-balancing loss (weight 1e-2) computed from routing frequencies
#    averaged over all shards. Only the partition affinities A, the routers Wr, and the
#    allocator train. Gradients are clipped to norm 1, and a non-finite norm stops the run.
# 3. The Sinkhorn temperature anneals from 1.0 to 0.1 during warmup (the first 20% of steps),
#    while the learning rate ramps up linearly before a cosine decay to zero.
# 4. Test perplexity is measured at the end, and every 10M tokens for the three trace runs
#    plotted in the report.
# 5. export() writes the partition, routers, and allocator, and complete.json marks the run
#    finished.
#
# One GPU or four. With four processes (torch.distributed), rank r processes data shard r, and
# gradients and routing frequencies are all-reduced. With one GPU the same four shards are
# processed in sequence as "virtual" ranks, and an extra forward pass beforehand computes the
# routing frequencies the four shards would have shared, so both setups optimize the same
# objective on the same data.
#
# Resume. The complete training state (parameters, optimizer, scheduler, perplexity curve, and
# every random generator) is saved every 50 steps and whenever the session pauses. Each rank
# writes its own file, and resume.json names the latest step that every rank finished saving,
# so a crash during a save never mixes two generations. On resume the data loaders are
# fast-forwarded to the saved step. The diagnostic run also writes tokalloc_log.jsonl, which
# diagnostics.py turns into the allocation and turnover figures.

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .. import runtime
from ..allocator_static import StaticAllocator
from ..config import DIAGNOSTIC_RUN, MODELS, TRACE_RUNS
from ..evaluation.corpus import token_blocks
from ..evaluation.perplexity import perplexity
from .blocks import TrainingFFN, calibrate_shared, make_allocator
from .data import batches, load_blocks
from .losses import kl_ce_chunked


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write the complete tensor state before replacing the destination.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def save_resume(state, out, rank, world):
    step = state["step"]
    atomic_save(state, out / f"resume-step{step}-rank{rank}.pt")
    if world > 1:
        dist.barrier()
    if rank == 0:
        # Publish the step after every rank has saved its state.
        write_json({"step": step, "world": world}, out / "resume.json")
        keep = {f"resume-step{step}-rank{r}.pt" for r in range(world)}
        for path in out.glob("resume-step*-rank*.pt"):
            if path.name not in keep:
                path.unlink()
    if world > 1:
        dist.barrier()


def cosine(step, steps):
    # Ramp up linearly, then follow a cosine decay to zero.
    warmup = max(1, int(.2 * steps))
    if step < warmup:
        return step / warmup
    return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, steps - warmup))))


def tau_at(step, steps):
    warmup = max(1, int(.2 * steps))
    # Decrease transport temperature during warmup, then hold it at 0.1.
    return 1. + (.1 - 1.) * (step / warmup) if step < warmup else .1


def allocator_record(alloc, recipe, model):
    if isinstance(alloc, StaticAllocator):
        return {"alloc": alloc.state()}
    if alloc is None:
        return {}
    cfg = {"class": "DynamicAllocator", "d_model": model.config.hidden_size,
           "n_layers": alloc.nL, "k_init": recipe.k, "total": alloc.total,
           "temp": alloc.temp, "k_min": 1, "k_max": alloc.k_max,
           "n_experts": alloc.n_experts, "init_spread": alloc.init_spread}
    return {"chain_alloc": {k: v.detach().cpu() for k, v in alloc.state_dict().items()},
            "chain_alloc_cfg": cfg}


@torch.no_grad()
def full_partitions(blocks):
    result = []
    for block in blocks:
        # Keep shared neurons labeled -1 and fill routed expert assignments.
        labels = torch.full_like(block.carpool, -1, dtype=torch.long)
        labels[block.rest_idx] = block.partition()
        result.append(labels.cpu())
    return result


def export(model, blocks, alloc, recipe, out, step, ppl):
    spec = MODELS[recipe.model]
    parts = full_partitions(blocks)
    state = {"A": [b.A.detach().cpu() for b in blocks],
             "Wr": [b.Wr.detach().cpu() for b in blocks],
             "carpool": [b.carpool.cpu() for b in blocks], "neuron_to_expert": parts,
             "partition_tau": .1, "sel_bias": [b.sel_bias.cpu() for b in blocks],
             "use_sel_bias": any(b.use_sel_bias for b in blocks), "gains": None}
    atomic_save(state, out / "alignment_state.pt")
    atomic_save({"neuron_to_expert": parts}, out / "partition.pt")
    record = allocator_record(alloc, recipe, model)
    if record:
        atomic_save(record, out / "allocator_state.pt")
    write_json({"model": spec["checkpoint"], "revision": spec["revision"],
                "E": blocks[0].E, "E_total": blocks[0].E + 1, "s": blocks[0].s,
                "k": [recipe.k] * len(blocks), "num_layers": len(blocks),
                "d_ffn": model.config.intermediate_size, "lane_holdout": blocks[0].s,
                "world": recipe.training_world, "steps_done": step, "seed": recipe.seed,
                "tokens_seen": step * recipe.global_batch * 2048,
                "final_test_ppl": ppl, "completed": step == recipe.steps,
                "recipe": recipe.record()}, out / "sparse_meta.json")


def routing_loss(blocks, frequencies):
    z, balance = [], []
    for b, f in zip(blocks, frequencies):
        # Penalize squared log-sum-exp and frequency-weighted routing probabilities.
        stat = b._stats
        z.append(stat["L"].float().logsumexp(-1).square().mean())
        balance.append(b.E * (f * stat["P"].mean(0)).sum())
    return 1e-3 * torch.stack(z).mean() + 1e-2 * torch.stack(balance).mean()


def record_training(blocks, ids, recipe, step, previous):
    cur = [b._last_n2e.detach().cpu().clone() for b in blocks]
    counts = torch.stack([b.last_counts.detach().cpu() for b in blocks], dim=1)
    flat = ids.detach().cpu().flatten()
    idx = torch.arange(0, len(flat), max(1, len(flat) // 24))[:24]
    row = {"step": step, "tokens": step * recipe.global_batch * 2048,
           "layer_k_mean": [round(float(v), 4) for v in counts.float().mean(0)],
           "budget_target": recipe.k * len(blocks),
           "sample": [{"pos": int(i), "tok": int(flat[i]), "k": counts[i].tolist()} for i in idx]}
    # Average the fraction of changed neuron assignments across layers.
    if previous is not None:
        changes = [round(float((a != b).float().mean()), 6) for a, b in zip(cur, previous)]
        row["partition_churn_mean"] = round(sum(changes) / len(changes), 6)
    return row, cur


def train(recipe, data_root, runs_root, device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if world not in (1, recipe.training_world):
        raise ValueError(f"Use one GPU or {recipe.training_world} GPUs for this recipe")
    out = Path(runs_root) / recipe.name
    out.mkdir(parents=True, exist_ok=True)
    signature = {"recipe": recipe.record(), "world": world}
    if (out / "complete.json").exists():
        if json.loads((out / "complete.json").read_text())["signature"] != signature:
            raise ValueError("Completed run uses a different recipe or device count")
        return
    arr = load_blocks(recipe.data_key, data_root)
    torch.manual_seed(recipe.seed)
    spec = MODELS[recipe.model]
    model = AutoModelForCausalLM.from_pretrained(spec["checkpoint"], revision=spec["revision"],
                                               dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    tok = AutoTokenizer.from_pretrained(spec["checkpoint"], revision=spec["revision"])
    model.requires_grad_(False)
    model.config.use_cache = False
    blocks = []
    for layer in model.model.layers:
        b = TrainingFFN(layer.mlp, spec["experts"] - 1, recipe.k).to(device)
        layer.mlp = b
        blocks.append(b)
    model.eval()
    calibrate_shared(model, blocks, torch.as_tensor(np.array(arr[:2]), device=device).long())
    alloc = make_allocator(recipe, model)
    if alloc is not None:
        alloc.to(device)
        for i, block in enumerate(blocks):
            block.allocator, block.layer = alloc, i
    groups = [{"params": [p for b in blocks for p in (b.A, b.Wr) if p.requires_grad],
               "lr": recipe.learning_rate}]
    if alloc is not None:
        # Keep allocator updates in a separate optimizer group without weight decay.
        groups.append({"params": list(alloc.parameters()), "lr": recipe.allocator_lr,
                       "weight_decay": 0.})
    opt = torch.optim.AdamW(groups, betas=(.9, .95), weight_decay=recipe.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine(s, recipe.steps))
    ref_world, micro = recipe.training_world, recipe.micro_batch
    # Process all data shards on one device, or assign one shard to each rank.
    virtual = list(range(ref_world)) if world == 1 else [rank]
    loaders = {r: batches(arr, micro, ref_world, r, recipe.data_seed) for r in virtual}
    accum = recipe.global_batch // (ref_world * micro)
    test_blocks = token_blocks(tok)
    step0, curve, previous = 0, [], None
    manifest = out / "resume.json"
    # Load only the checkpoint generation committed by all ranks.
    if manifest.exists():
        generation = json.loads(manifest.read_text())
        if generation["world"] != world:
            raise ValueError("Resume generation has a different GPU count")
        resume = out / f"resume-step{generation['step']}-rank{rank}.pt"
        if not resume.exists():
            raise ValueError("Committed resume generation is missing a rank")
    else:
        resume = out / f"resume-{rank}.pt"
    if resume.exists():
        st = torch.load(resume, map_location=device, weights_only=False)
        if st["signature"] != signature:
            raise ValueError("Resume recipe/device count mismatch")
        for b, state in zip(blocks, st["blocks"]):
            b.A.data.copy_(state["A"])
            b.Wr.data.copy_(state["Wr"])
        if alloc is not None:
            alloc.load_state_dict(st["allocator"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step0, curve = st["step"], st["curve"]
        # Keep the logged partition on the CPU, where record_training compares against it.
        previous = None if st["previous"] is None else [p.cpu() for p in st["previous"]]
        if world > 1:
            steps = [None] * world
            dist.all_gather_object(steps, step0)
            if len(set(steps)) != 1:
                raise ValueError(f"Training ranks have inconsistent resume steps: {steps}")
        if isinstance(alloc, StaticAllocator):
            alloc._gen.set_state(st["static_rng"].cpu())
        # Restore random streams before advancing the data loaders to the saved step.
        torch.set_rng_state(st["rng"].cpu())
        if st["cuda_rng"] is not None:
            torch.cuda.set_rng_state(st["cuda_rng"].cpu(), device)
        for loader in loaders.values():
            for _ in range(step0 * accum):
                next(loader)
        if rank == 0 and (out / "tokalloc_log.jsonl").exists():
            lines = (out / "tokalloc_log.jsonl").read_text().splitlines()
            rows = []
            # Ignore an interrupted final log line; reject malformed earlier lines.
            for i, line in enumerate(lines):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if i != len(lines) - 1:
                        raise
            (out / "tokalloc_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["step"] <= step0))
        print(f"{recipe.name}: resumed step {step0}/{recipe.steps} on rank {rank}", flush=True)
    model.train()
    for step in range(step0 + 1, recipe.steps + 1):
        for b in blocks:
            b.tau = tau_at(step - 1, recipe.steps)
            b._eval_n2e = None
        opt.zero_grad(set_to_none=True)
        for accum_idx in range(accum):
            ids_group = {r: next(loaders[r]).to(device) for r in virtual}
            frequencies = None
            # Average routing frequencies across all four microbatches.
            if world == 1 and ref_world > 1:
                with torch.no_grad():
                    for r in virtual:
                        model.model(ids_group[r], use_cache=False)
                        f = [b._stats["counts"] / b._stats["P"].shape[0] for b in blocks]
                        frequencies = f if frequencies is None else [a + b for a, b in zip(frequencies, f)]
                    frequencies = [f / ref_world for f in frequencies]
            for r in virtual:
                ids = ids_group[r]
                # Compute teacher logits with the neuron masks disabled.
                for b in blocks:
                    b.masked = False
                with torch.no_grad():
                    teacher = model(ids).logits
                for b in blocks:
                    b.masked = True
                student = model(ids).logits
                if frequencies is None:
                    frequencies = [b._stats["counts"] / b._stats["P"].shape[0] for b in blocks]
                    if world > 1:
                        for f in frequencies:
                            dist.all_reduce(f)
                            f.div_(world)
                kl, ce = kl_ce_chunked(teacher, student, ids, recipe.loss_chunks)
                # Scale each backward pass by the number of accumulated microbatches.
                loss = 2 * kl + ce + routing_loss(blocks, frequencies)
                (loss / (accum * len(virtual))).backward()
                if rank == 0 and r == 0 and accum_idx == accum - 1 and recipe.name == DIAGNOSTIC_RUN:
                    sample_row, sample_partition = record_training(blocks, ids, recipe, step, previous)
                del teacher, student, kl, ce
        params = [p for g in opt.param_groups for p in g["params"]]
        # Average parameter gradients across ranks before clipping and updating.
        if world > 1:
            for p in params:
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad.div_(world)
        if isinstance(alloc, StaticAllocator):
            alloc.center_grad_()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        opt.step()
        sched.step()
        if isinstance(alloc, StaticAllocator):
            alloc.project_()

        if rank == 0 and recipe.name == DIAGNOSTIC_RUN:
            row, previous = sample_row, sample_partition
            with (out / "tokalloc_log.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
        if rank == 0 and (step == 1 or step % 25 == 0):
            print(f"{recipe.name}: step {step}/{recipe.steps}; loss {float(loss.detach()):.5f}", flush=True)
        tokens = step * recipe.global_batch * 2048
        prior_tokens = (step - 1) * recipe.global_batch * 2048
        final = step == recipe.steps
        # Evaluate at the last step or when the token count crosses an interval boundary.
        if final or (recipe.name in TRACE_RUNS.values() and tokens // 10_000_000 > prior_tokens // 10_000_000):
            for b in blocks:
                b.tau, b._eval_n2e = .1, None
            ppl = perplexity(model, test_blocks, device, shards=ref_world)
            curve.append({"step": step, "tokens": tokens, "val_ppl": ppl, "split": "test"})
            if rank == 0:
                write_json(curve, out / "curve.json")
        stop = runtime.due(device)
        if final or step % 50 == 0 or stop:
            state = {"signature": signature, "step": step, "curve": curve, "previous": previous,
                     "blocks": [{"A": b.A.detach(), "Wr": b.Wr.detach()} for b in blocks],
                     "allocator": alloc.state_dict() if alloc is not None else None,
                     "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                     "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None,
                     "static_rng": alloc._gen.get_state() if isinstance(alloc, StaticAllocator) else None}
            save_resume(state, out, rank, world)
        if stop:
            runtime.pause()
        if final and rank == 0:
            export(model, blocks, alloc, recipe, out, step, curve[-1]["val_ppl"])
            write_json({"signature": signature, "steps": step}, out / "complete.json")
    # Finish exporting when the resumed checkpoint already contains the final step.
    if step0 == recipe.steps and rank == 0:
        for b in blocks:
            b.tau = .1
        export(model, blocks, alloc, recipe, out, step0, curve[-1]["val_ppl"])
        write_json({"signature": signature, "steps": step0}, out / "complete.json")
    if world > 1:
        dist.barrier()
