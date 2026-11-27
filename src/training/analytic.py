# Spectral-MoE: src/training/analytic.py
# Author: Zachary Broveak
#
# Analytic conversion: construct experts without gradient training, recover with LoRA, then
# train an allocator.
#
# Stage 1, construction (recover and carve_one), once per base model. Layers are converted one
# at a time, in order, from calibration inputs (8 blocks of 2048 tokens) passed through the
# layers already converted below, so each layer is built from the inputs it will actually
# receive. For each layer, following CMoE:
# - compute FFN activations with unit-normalized inputs and weight rows, and mark each token's
#   8 largest-magnitude neurons;
# - make the 128 most frequently marked neurons the shared expert;
# - cluster the remaining neurons into balanced groups of 128 by their marking patterns, using
#   one k-means-style iteration solved as a linear assignment problem
#   (analytic_primitives/partition.py);
# - initialize each routed expert's router row from the normalized gate and up weights of the
#   neuron closest to its cluster centre.
# Each finished layer is saved, so a paused construction resumes at the next layer.
#
# Stage 2, recovery. The converted model is fine-tuned briefly with LoRA (rank 8 on the
# attention and expert projections, 1024 steps of 2 sequences, next-token loss), and the
# adapters are merged into the weights. The result, runs/<model>/analytic-base/recovered.pt,
# is shared by every analytic arm of that model.
#
# Stage 3, allocation (train). With the recovered model frozen, only the static or dynamic
# allocator trains (analytic_blocks.py), distilled from the same model with every expert
# active (2 * KL + CE). The uniform arm has no allocator and is only measured. Each arm saves
# its allocator state and its test perplexity.
#
# Construction runs mostly on the CPU while the GPU waits: the assignment problem has one row
# and one column per routed neuron, so its cost grows quickly with FFN width.

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .. import runtime
from ..allocator_static import StaticAllocator
from ..config import BUDGETS, MODELS
from ..evaluation.corpus import token_blocks
from ..evaluation.perplexity import perplexity
from .analytic_blocks import AnalyticTrainingBlock, STEAdapter, dense_teacher
from .analytic_primitives.model import MoE, Router
from .analytic_primitives.partition import construct_experts_k_means
from .blocks import make_allocator
from .data import batches, calibration, load_blocks
from .learned import allocator_record, atomic_save, cosine, write_json

RECOVERY_STEPS = 1024


def load_dense(model_key, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = MODELS[model_key]
    model = AutoModelForCausalLM.from_pretrained(spec["checkpoint"], revision=spec["revision"],
                                               dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(spec["checkpoint"], revision=spec["revision"])
    return model, tok


@torch.no_grad()
def carve_one(mlp, hidden, experts, k):
    # Normalize hidden vectors and projection rows before measuring activation patterns.
    x = F.normalize(hidden, p=2, dim=-1)
    h = mlp.act_fn(F.linear(x, F.normalize(mlp.gate_proj.weight, p=2, dim=1)))
    h = h * F.linear(x, F.normalize(mlp.up_proj.weight, p=2, dim=1))
    scores = h.cpu().float().reshape(-1, h.shape[-1])
    markers = torch.zeros_like(scores)

    # Mark the eight largest absolute activations per token.
    for i, values in enumerate(scores):
        markers[i, values.abs().topk(8).indices] = 1.
    rates = markers.sum(0) / len(markers)
    _, modules, representatives = construct_experts_k_means(
        rates, markers, experts, 1, mlp.gate_proj.in_features,
        mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    # Initialize router projections from the selected representative neurons.
    router = Router(mlp.gate_proj.in_features, experts - 1, k)
    router.classifier.weight.data = F.normalize(mlp.up_proj.weight.data[representatives, :], p=2, dim=1)
    router.gate.weight.data = F.normalize(mlp.gate_proj.weight.data[representatives, :], p=2, dim=1)
    block = MoE(mlp.gate_proj.in_features, 128, experts, 1, k)
    block.gate, block.experts, block.shared_experts = router, modules[1:], modules[0]
    return block


def load_converted(model_key, path, device):
    model, tok = load_dense(model_key, device)
    spec = MODELS[model_key]
    for layer in model.model.layers:
        layer.mlp = MoE(spec["d_model"], 128, spec["experts"], 1, BUDGETS[model_key][.25]).to(device)
    st = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(st["full_state"], strict=True, assign=True)
    model.to(device)
    return model, tok


def recover(model_key, data_root, runs_root, device="cuda:0"):
    from peft import LoraConfig, TaskType, get_peft_model

    out = Path(runs_root) / model_key / "analytic-base"
    out.mkdir(parents=True, exist_ok=True)
    recovered = out / "recovered.pt"
    if recovered.exists():
        return recovered
    arr = load_blocks(model_key, data_root)
    spec = MODELS[model_key]
    k = BUDGETS[model_key][.25]
    raw = out / "carved.pt"
    provenance = {"model": spec["checkpoint"], "revision": spec["revision"],
                  "E": spec["experts"], "S": 1, "A": k, "expert_width": 128, "seed": 17}
    if not raw.exists():
        torch.manual_seed(17)
        model, _ = load_dense(model_key, device)
        ids = calibration(arr, 8).to(device)
        for i, layer in enumerate(model.model.layers):
            partial = out / "carve-layers" / f"layer-{i}.pt"
            if partial.exists():
                saved = torch.load(partial, map_location="cpu", weights_only=False, mmap=True)
                layer.mlp = MoE(spec["d_model"], 128, spec["experts"], 1, k)
                layer.mlp.load_state_dict(saved["state"], strict=True, assign=True)
                layer.mlp.to(device)
                torch.set_rng_state(saved["rng"].cpu())
                torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
                print(f"{model_key}: restored constructed layer {i + 1}", flush=True)
                continue
            # Capture the current layer input while running the partially converted model.
            captured = []
            handle = layer.mlp.register_forward_pre_hook(
                lambda m, a, sink=captured: sink.append(a[0].detach()))
            with torch.no_grad():
                model(ids)
            handle.remove()
            layer.mlp = carve_one(layer.mlp, captured[0], spec["experts"], k).to(device)
            print(f"{model_key}: constructed layer {i + 1}/{len(model.model.layers)}", flush=True)
            del captured
            atomic_save({"state": layer.mlp.state_dict(), "rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state(device)}, partial)
            runtime.checkpoint()
        atomic_save({"full_state": model.state_dict(), "provenance": provenance}, raw)
        del model
        torch.cuda.empty_cache()
    runtime.checkpoint()

    torch.manual_seed(17)
    model, _ = load_converted(model_key, raw, device)
    # Add low-rank adapters to the attention and FFN projections.
    cfg = LoraConfig(r=8, lora_alpha=32, lora_dropout=.1, bias="none",
                     task_type=TaskType.CAUSAL_LM,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "down_proj", "up_proj"])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, cfg)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=5.65e-5, betas=(.9, .95), eps=1e-8, weight_decay=0.)
    ids = calibration(arr, 2 * RECOVERY_STEPS)
    resume, start = out / "recovery-resume.pt", 0
    if resume.exists():
        st = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(st["adapters"], strict=False)
        opt.load_state_dict(st["optimizer"])
        # Restore the random streams used by adapter dropout.
        torch.set_rng_state(st["rng"].cpu())
        if st["cuda_rng"] is not None:
            torch.cuda.set_rng_state(st["cuda_rng"].cpu(), device)
        start = st["step"]
    model.train()
    for step in range(start, RECOVERY_STEPS):
        batch = ids[2 * step:2 * step + 2].to(device)
        opt.zero_grad(set_to_none=True)
        loss = model(batch, labels=batch).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite LoRA recovery loss")
        loss.backward()
        opt.step()
        if step % 25 == 0:
            print(f"{model_key}: recovery {step + 1}/{RECOVERY_STEPS}; loss {float(loss.detach()):.5f}", flush=True)
        stop = runtime.due()
        if (step + 1) % 50 == 0 or step + 1 == RECOVERY_STEPS or stop:
            atomic_save({"step": step + 1, "adapters": {n: p for n, p in model.state_dict().items() if "lora_" in n},
                         "optimizer": opt.state_dict(), "rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None}, resume)
        if stop:
            runtime.pause()
    # Fold the adapter updates into the model weights before saving.
    model = model.merge_and_unload().eval()
    model.gradient_checkpointing_disable()
    provenance["sft"] = True
    atomic_save({"full_state": model.state_dict(), "provenance": provenance}, recovered)
    return recovered


def train(recipe, data_root, runs_root, device="cuda:0"):
    recovered = Path(runs_root) / recipe.model / "analytic-base" / "recovered.pt"
    if not recovered.exists():
        recover(recipe.model, data_root, runs_root, device)
    out = Path(runs_root) / recipe.name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        if json.loads((out / "complete.json").read_text())["recipe"] != recipe.record():
            raise ValueError("Completed analytic run has a different recipe")
        return
    arr = load_blocks(recipe.data_key, data_root)
    torch.manual_seed(recipe.seed)
    model, tok = load_converted(recipe.model, recovered, device)
    model.requires_grad_(False)
    alloc = make_allocator(recipe, model)
    if alloc is not None:
        alloc.to(device)
    adapter = STEAdapter(alloc) if isinstance(alloc, StaticAllocator) else alloc
    blocks = []
    for i, layer in enumerate(model.model.layers):
        block = AnalyticTrainingBlock(layer.mlp, i, adapter, recipe.k, 1)
        layer.mlp = block
        blocks.append(block)
    test = token_blocks(tok)
    start = 0
    if alloc is not None:
        opt = torch.optim.AdamW(alloc.parameters(), lr=recipe.allocator_lr, betas=(.9, .95), weight_decay=0.)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine(s, recipe.steps))
        resume = out / "resume.pt"
        if resume.exists():
            st = torch.load(resume, map_location=device, weights_only=False)
            if st["recipe"] != recipe.record():
                raise ValueError("Analytic resume recipe mismatch")
            alloc.load_state_dict(st["allocator"])
            opt.load_state_dict(st["optimizer"])
            sched.load_state_dict(st["scheduler"])
            start = st["step"]
            if isinstance(alloc, StaticAllocator):
                alloc._gen.set_state(st["static_rng"].cpu())
        loader = batches(arr, 1, 1, 0, recipe.data_seed)
        # Skip the two microbatches already consumed by each completed step.
        for _ in range(start * 2):
            next(loader)
        model.train()
        for step in range(start + 1, recipe.steps + 1):
            opt.zero_grad(set_to_none=True)
            for _ in range(2):
                ids = next(loader).to(device)
                with torch.no_grad(), dense_teacher(blocks):
                    teacher = model(ids).logits.float()
                student = model(ids).logits.float()
                # Combine teacher KL divergence with shifted next-token cross-entropy.
                p = teacher.softmax(-1)
                kl = (p * (teacher.log_softmax(-1) - student.log_softmax(-1))).sum(-1).mean()
                ce = F.cross_entropy(student[:, :-1].reshape(-1, student.shape[-1]), ids[:, 1:].reshape(-1))
                ((2 * kl + ce) / 2).backward()
                del teacher, student, p, kl, ce
            if isinstance(alloc, StaticAllocator):
                alloc.center_grad_()
            norm = torch.nn.utils.clip_grad_norm_(alloc.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite analytic allocator gradient")
            opt.step()
            sched.step()
            if isinstance(alloc, StaticAllocator):
                alloc.project_()
            if step % 25 == 0:
                print(f"{recipe.name}: step {step}/{recipe.steps}", flush=True)
            stop = runtime.due()
            if step % 50 == 0 or step == recipe.steps or stop:
                atomic_save({"recipe": recipe.record(), "step": step, "allocator": alloc.state_dict(),
                             "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                             "static_rng": alloc._gen.get_state() if isinstance(alloc, StaticAllocator) else None}, resume)
            if stop:
                runtime.pause()
    runtime.checkpoint()
    ppl = perplexity(model, test, device, batch_size=1, shards=1)
    record = allocator_record(alloc, recipe, model)
    if isinstance(alloc, StaticAllocator):
        record = {"chain_alloc": alloc.state_dict(), "chain_alloc_cfg": {"class": "StaticAllocator"}}
    elif alloc is None:
        record = {"chain_alloc": None, "chain_alloc_cfg": None}
    spec = MODELS[recipe.model]
    record.update({"model": spec["checkpoint"], "base_is_sft": True, "lora_state": None,
                   "budget": recipe.k * len(blocks), "step": recipe.steps, "k_fixed": recipe.k,
                   "cmoe": {"E": spec["experts"], "S": 1, "A": recipe.k}})
    atomic_save(record, out / "allocator_state.pt")
    write_json({"final_ppl": ppl, "steps": recipe.steps, "completed": True}, out / "meta.json")
    write_json({"recipe": recipe.record()}, out / "complete.json")
