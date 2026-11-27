# Spectral-MoE: src/training/refit.py
# Author: Zachary Broveak
#
# Closed-form router refit and router selection diagnostics.
#
# After learned training, does the router pick the experts that matter most for each token?
# refit() measures this and fits an alternative router directly to the answer.
#
# Oracle. For a token and a layer, an expert's importance is the sum over its neurons of
# |activation| times the L1 norm of the neuron's down-projection column, roughly how much the
# expert would contribute to the output. The k most important experts form the oracle
# selection.
#
# Fit. Router inputs are collected on REFIT_FIT_TOKENS training tokens. For each layer a ridge
# regression (ridge.py) maps the input to standardized expert importances; its intercept
# becomes a selection bias, and weights and bias are rescaled to the logit scale of the
# original router. Only the router changes; the partition and allocator are copied unchanged.
#
# Diagnostics on REFIT_DIAGNOSTIC_TOKENS WikiText-2 test tokens (empty rows removed), using
# each token's actual per-layer count:
# - overlap: the fraction of the router's selected experts that are also in the oracle
#   selection;
# - coverage: the importance captured by the router's selection relative to the oracle's.
# Both are averaged over layers for the original and the refit router, and evaluate.py
# measures accuracy with each. Layers are saved as they finish, so a paused refit resumes at
# the next layer.

import shutil
from pathlib import Path

import numpy as np
import torch

from .. import runtime
from ..config import REFIT_DIAGNOSTIC_TOKENS, REFIT_FIT_TOKENS, REFIT_RIDGE
from ..evaluation.revisions import WIKITEXT_REVISION
from ..model.restore import restore_arm
from .data import load_blocks
from .learned import atomic_save, write_json
from .ridge import ridge_fit, row_standardise, score_sel


@torch.no_grad()
def collect(model, blocks, ids, device):
    storage = [{"x": [], "k": []} for _ in blocks]
    def hook_for(i):
        def hook(block, args, output):
            # Flatten batch and sequence dimensions before storing the inputs.
            x = args[0].reshape(-1, args[0].shape[-1])
            storage[i]["x"].append(x.to(torch.float16).cpu())
            storage[i]["k"].append(block.last_counts.cpu())
        return hook
    handles = [b.register_forward_hook(hook_for(i)) for i, b in enumerate(blocks)]
    try:
        for seq in ids:
            model(seq[None, :].to(device))
    finally:
        for h in handles:
            h.remove()
        alloc = model.spectral_moe["allocator"]
        if alloc is not None:
            alloc.clear()
    return [{k: torch.cat(v) for k, v in row.items()} for row in storage]


@torch.no_grad()
def importance(block, inputs, device):
    # Weight neuron activations by their output-column L1 norms.
    weight = block.mlp.down_proj.weight.abs().sum(0).float()[block.routed_idx]
    scores = torch.empty(len(inputs), len(block.Wr), dtype=torch.float32)
    for start in range(0, len(inputs), 2048):
        x = inputs[start:start + 2048].to(device=device, dtype=block.Wr.dtype)
        h = block.mlp.act_fn(block.mlp.gate_proj(x)) * block.mlp.up_proj(x)
        imp = h.abs().float()[:, block.routed_idx] * weight[None, :]
        result = torch.zeros(len(x), len(block.Wr), device=device)
        # Sum routed-neuron contributions into their assigned expert columns.
        result.scatter_add_(1, block.partition[None, :].expand(len(x), -1), imp)
        scores[start:start + len(x)] = result.cpu()
    return scores


@torch.no_grad()
def refit(recipe, data_root, runs_root, device="cuda:0"):
    from datasets import load_dataset

    source = Path(runs_root) / recipe.name
    out = Path(runs_root) / (recipe.name + "-ridge")
    if (out / "complete.json").exists():
        return out
    model, tok = restore_arm(source, expect_steps=recipe.steps, device=device)
    blocks = model.spectral_moe["blocks"]
    arr = load_blocks(recipe.data_key, data_root)
    g = torch.Generator().manual_seed(recipe.data_seed)
    # Choose complete calibration blocks with the seeded random permutation.
    pick = torch.randperm(len(arr), generator=g)[:REFIT_FIT_TOKENS // 2048].tolist()
    fit_ids = torch.from_numpy(np.asarray(arr[pick], dtype=np.int64))

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", revision=WIKITEXT_REVISION)
    # Remove empty rows before tokenizing the diagnostic stream.
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[:, :REFIT_DIAGNOSTIC_TOKENS].reshape(-1, 2048)
    fit = collect(model, blocks, fit_ids, device)
    diagnostic = collect(model, blocks, ids, device)
    new_weights, new_biases, rows = [], [], []
    for i, b in enumerate(blocks):
        partial = out / "layers" / f"layer-{i}.pt"
        # Reuse the saved fit and diagnostics for a completed layer.
        if partial.exists():
            saved = torch.load(partial, map_location="cpu", weights_only=False)
            new_weights.append(saved["weight"])
            new_biases.append(saved["bias"])
            rows.append(saved["diagnostics"])
            continue
        xf, xc, kc = fit[i]["x"], diagnostic[i]["x"], diagnostic[i]["k"]
        yf = row_standardise(importance(b, xf, device))
        yc = importance(b, xc, device)
        old_scores = []
        for start in range(0, len(xc), 8192):
            old_scores.append((xc[start:start + 8192].to(device=device, dtype=b.Wr.dtype) @ b.Wr.t()).float().softmax(-1).cpu())
        ov0, cv0 = score_sel(torch.cat(old_scores), yc, kc)
        W, mu, ybar = ridge_fit(xf.float(), yf, REFIT_RIDGE)
        # Recover the intercept and match the original logit scale.
        bias = (ybar - mu @ W).squeeze(0)
        old_scale = (xf.float() @ b.Wr.float().t().cpu()).std()
        new_scale = (xf.float() @ W).std().clamp_min(1e-12)
        alpha = float(old_scale / new_scale)
        Wr = (W.t() * alpha).to(b.Wr.dtype)
        bias = (bias * alpha).float()
        # Measure selections at the stored weight precision with scaled offsets.
        deployed = (xc.to(Wr.dtype) @ Wr.t()).float() + bias[None, :]
        ovd, cvd = score_sel(deployed, yc, kc)
        rows.append({"trained": {"overlap": ov0, "coverage": cv0},
                     "deployed": {"overlap": ovd, "coverage": cvd}})
        new_weights.append(Wr)
        new_biases.append(bias)
        atomic_save({"weight": Wr, "bias": bias, "diagnostics": rows[-1]}, partial)
        print(f"{recipe.name}: refitted router {i + 1}/{len(blocks)}", flush=True)
        runtime.checkpoint()
    report = {"fit_tokens": REFIT_FIT_TOKENS, "diagnostic_tokens": REFIT_DIAGNOSTIC_TOKENS, "ridge": REFIT_RIDGE,
              "mean_trained": {k: sum(r["trained"][k] for r in rows) / len(rows) for k in ("overlap", "coverage")},
              "mean_refit": {k: sum(r["deployed"][k] for r in rows) / len(rows) for k in ("overlap", "coverage")}}
    out.mkdir(parents=True, exist_ok=True)
    st = torch.load(source / "alignment_state.pt", map_location="cpu", weights_only=False)
    # Replace router weights and offsets while preserving the other state entries.
    st.update(Wr=new_weights, sel_bias=new_biases, use_sel_bias=True)
    atomic_save(st, out / "alignment_state.pt")
    for name in ("allocator_state.pt", "sparse_meta.json", "partition.pt"):
        shutil.copyfile(source / name, out / name)
    write_json(report, out / "refit_report.json")
    write_json({"recipe": recipe.record(), "ridge": REFIT_RIDGE}, out / "complete.json")
    return out
