# Spectral-MoE: src/model/restore.py
# Author: Zachary Broveak
#
# Rebuild a learned-base arm, or the dense baseline, from its run directory.
#
# A finished learned run (training/learned.py: export) contains:
# - sparse_meta.json: base model and revision, geometry, completed steps, and the full recipe;
# - alignment_state.pt: for every layer, the router Wr, the shared-neuron set (carpool), the
#   neuron-to-expert partition, and the selection biases added by a router refit;
# - allocator_state.pt: a static per-layer schedule, or dynamic allocator weights with their
#   configuration (absent for uniform allocation);
# - partition.pt: the partition on its own.
#
# restore_arm() loads the pinned base model, replaces every FFN with a MaskedFFN ("masked") or
# SparseExperts ("extracted") block, and attaches the allocator. It validates everything it
# can before evaluation starts: the completed step count (so an unfinished run is never
# evaluated), that the router width matches the declared expert count with or without a shared
# expert, that selection biases appear only with the flag a refit sets, that schedules cover
# every layer and match their budget, and that partitions match the FFN width. The errors are
# deliberately specific, because mismatched files would otherwise produce a plausible but
# wrong number.
#
# With base="dense" the metadata only identifies the base model, which is then evaluated
# unchanged.

import json
import os

import torch

from ..allocator_dynamic import DynamicAllocator
from .masked_ffn import MaskedFFN
from .sparse_experts import SparseExperts

TRAINING_STEPS = 3500

DYNAMIC_CLASS_NAMES = ("DynamicAllocator",)

ATTENTION = "sdpa"


def _as_mask(v, width, device):
    if v is None:
        return None
    t = torch.as_tensor(v)
    if t.dtype == torch.bool:
        return t.to(device)
    # Expand neuron indices into a boolean mask.
    m = torch.zeros(width, dtype=torch.bool, device=device)
    if t.numel():
        m[t.long().to(device)] = True
    return m


def load_metadata(weights_dir, expect_steps=TRAINING_STEPS):
    path = os.path.join(weights_dir, "sparse_meta.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{weights_dir}: missing sparse_meta.json")
    with open(path) as handle:
        meta = json.load(handle)
    # Reject a checkpoint whose completed step count differs from the request.
    steps = meta.get("steps_done")
    if expect_steps is not None and steps != expect_steps:
        raise SystemExit(f"Checkpoint has steps_done={steps}; expected {expect_steps}")
    return meta


def load_schedule(weights_dir, device, dtype=torch.float32):
    path = os.path.join(weights_dir, "allocator_state.pt")
    if not os.path.isfile(path):
        return None, None
    state = torch.load(path, map_location="cpu", weights_only=False)

    if "chain_alloc" in state:
        cfg = dict(state.get("chain_alloc_cfg") or {})
        if not cfg:
            raise SystemExit(f"{path} holds allocator weights but no configuration, so the widths "
                             f"it was trained at are unknown")
        if cfg.get("class") not in DYNAMIC_CLASS_NAMES:
            raise SystemExit(f"{path} names allocator class {cfg.get('class')!r}, which this "
                             f"package does not implement")
        alloc = DynamicAllocator.from_config(cfg).to(device=device, dtype=dtype)
        saved = dict(state["chain_alloc"])
        unknown = set(saved) - set(alloc.state_dict())
        if unknown:
            raise SystemExit(f"{path} carries entries this allocator does not define: "
                             f"{sorted(unknown)[:6]}")
        alloc.load_state_dict(saved, strict=True)
        alloc.eval()
        return "dynamic", alloc

    if "alloc" in state:
        saved = state["alloc"]
        if "k_int" not in saved:
            raise SystemExit(f"{path}: missing static schedule k_int")
        # Load the integer schedule and enforce its saved total.
        counts = torch.as_tensor(saved["k_int"]).long().tolist()
        total = saved.get("total")
        if total is not None and sum(counts) != int(total):
            raise SystemExit(f"{path}: the saved schedule sums to {sum(counts)} against a recorded "
                             f"budget of {int(total)}")
        return "static", counts

    raise SystemExit(f"{path} holds neither a per-token allocator nor a per-layer schedule")


def _load_base(meta, device, dtype):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        meta["model"], revision=meta.get("revision"), dtype=dtype,
        attn_implementation=ATTENTION).to(device).eval()


def restore_arm(weights_dir, base="learned", device="cuda:0", dtype=torch.bfloat16,
                expect_steps=TRAINING_STEPS, realization="masked"):
    from transformers import AutoTokenizer

    meta = load_metadata(weights_dir, expect_steps=expect_steps)
    tok = AutoTokenizer.from_pretrained(meta["model"], revision=meta.get("revision"))
    if base == "dense":
        model = _load_base(meta, device, dtype)
        model.spectral_moe = {"kind": "dense", "blocks": [], "allocator": None,
                              "per_layer_k": [], "meta": meta}
        return model, tok
    if base != "learned":
        raise SystemExit(f"restore_arm rebuilds learned-base and dense rows, not {base!r}")

    state = torch.load(os.path.join(weights_dir, "alignment_state.pt"), map_location="cpu",
                       weights_only=False)
    n_layers, d_ffn = int(meta["num_layers"]), int(meta["d_ffn"])

    # Infer shared-expert presence from the router width.
    E_routed, E_total = int(state["Wr"][0].shape[0]), int(meta["E_total"])
    if E_routed not in (E_total, E_total - 1):
        raise SystemExit(f"the saved router has {E_routed} experts but this arm declares "
                         f"E_total={E_total}; the state and the metadata are different geometries")
    has_always_on = E_routed == E_total - 1
    recorded = meta.get("lane_holdout")
    if recorded is not None and bool(recorded) != has_always_on:
        raise SystemExit(
            f"the router's width implies an always-active set is "
            f"{'present' if has_always_on else 'absent'} while the metadata records {recorded!r}")

    use_sel_bias, biases = state.get("use_sel_bias"), state.get("sel_bias")
    if not use_sel_bias and biases is not None:
        worst = max(float(torch.as_tensor(b).abs().max()) for b in biases)
        if worst > 0.0:
            raise SystemExit(f"this arm records no selection-bias flag but carries a bias up to "
                             f"{worst:.3g}")

    kind, schedule = load_schedule(weights_dir, device)
    if kind == "static":
        if len(schedule) != n_layers:
            raise SystemExit(f"the saved schedule covers {len(schedule)} layers, the arm has "
                             f"{n_layers}")
        per_layer_k = schedule
    else:
        per_layer_k = [int(v) for v in meta["k"]]

    model = _load_base(meta, device, dtype)
    layers = model.model.layers
    if len(layers) != n_layers:
        raise SystemExit(f"{meta['model']} has {len(layers)} blocks, the arm records {n_layers}")

    blocks = []
    for i in range(n_layers):
        partition = torch.as_tensor(state["neuron_to_expert"][i]).long().to(device)
        if partition.numel() != d_ffn:
            raise SystemExit(f"layer {i}: the saved partition holds {partition.numel()} entries "
                             f"against a recorded block width of {d_ffn}")
        always_on = (_as_mask(state.get("carpool", [None] * n_layers)[i], d_ffn, device)
                     if has_always_on else None)
        bias = (torch.as_tensor(biases[i]).to(device)
                if (use_sel_bias and biases is not None) else None)
        Wr_i = torch.as_tensor(state["Wr"][i]).to(device=device, dtype=dtype)
        if realization == "masked":
            block = MaskedFFN(layers[i].mlp, Wr_i, partition, always_on,
                              int(per_layer_k[i]), sel_bias=bias).to(device)
        elif realization == "extracted":
            dense = layers[i].mlp

            # Group neuron indices by routed expert and require equal group widths.
            groups = [torch.nonzero(partition == e, as_tuple=True)[0] for e in range(E_routed)]
            widths = {int(g.numel()) for g in groups}
            if len(widths) != 1:
                raise SystemExit(f"layer {i}: the saved partition is not balanced, expert widths "
                                 f"{sorted(widths)}")
            on_idx = (torch.nonzero(always_on, as_tuple=True)[0] if always_on is not None else None)
            d_model = int(dense.gate_proj.weight.shape[1])
            block = SparseExperts(d_model, widths.pop(), E_routed,
                                  int(per_layer_k[i]), dtype=dtype, always_on_idx=on_idx,
                                  requires_allocator=(kind == "dynamic")).to(device)
            block.load_from(dense.gate_proj.weight.data, dense.up_proj.weight.data,
                            dense.down_proj.weight.data, Wr_i, groups)
            if bias is not None:
                block.sel_bias.data.copy_(bias.to(device).float())
                block.use_sel_bias = bool(block.sel_bias.abs().max() > 0)
        else:
            raise SystemExit(f"unknown realization {realization!r}; expected 'masked' or "
                             f"'extracted'")
        layers[i].mlp = block
        blocks.append(block)

    if kind == "dynamic":
        if schedule.nL != n_layers:
            raise SystemExit(f"the allocator was trained over {schedule.nL} layers, the arm has "
                             f"{n_layers}")
        # Attach the same allocator with the corresponding layer index.
        for i, block in enumerate(blocks):
            if realization == "extracted":
                block.attach_allocator(schedule, i)
            else:
                block.attach(schedule, i)

    model.eval()
    model.spectral_moe = {"kind": kind or "uniform", "blocks": blocks,
                          "allocator": schedule if kind == "dynamic" else None,
                          "per_layer_k": per_layer_k, "meta": meta,
                          "realization": realization}
    return model, tok
