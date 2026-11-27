# Spectral-MoE: src/model/analytic.py
# Author: Zachary Broveak
#
# Rebuild an analytic-base arm for evaluation.
#
# Analytic arms are built in two stages (training/analytic.py). A CMoE-style construction
# followed by LoRA recovery produces one frozen MoE model per base model
# (runs/<model>/analytic-base/recovered.pt), and each arm then trains only an allocator on top
# of it. An arm directory therefore holds just allocator_state.pt and meta.json, and
# evaluation needs both the arm and the recovered base.
#
# AnalyticBlock is the inference form of a converted FFN: an always-on shared expert and
# n_routed routed experts, each a SwiGLU MLP of 128 neurons. The router scores experts as
# |score_up(x) * silu(score_gate(x))| (the CMoE score), turns the scores into probabilities p,
# and weights each selected expert's output by 1 + p * weight_scale. The number of experts per
# token is the construction's fixed k (uniform), a learned per-layer count (static), or a
# per-token count from an attached DynamicAllocator (dynamic).
#
# load_analytic_arm() checks that the arm and the recovered base share a geometry (total,
# shared, and active experts, and the base model), renames the training-time state keys
# (gate.classifier, gate.gate, gate.extra_scale, shared_experts) to this module's names, loads
# the remaining weights, including the merged LoRA updates, from the same recovered state, and
# attaches the allocator.

import json
import os

import torch
import torch.nn.functional as F
from torch import nn

from ..allocator_dynamic import DynamicAllocator

ATTENTION = "sdpa"


class _SwiGLU(nn.Module):
    def __init__(self, d_model, width):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, width, bias=False)
        self.up_proj = nn.Linear(d_model, width, bias=False)
        self.down_proj = nn.Linear(width, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class AnalyticBlock(nn.Module):
    def __init__(self, d_model, s, n_routed, n_always_on, k):
        super().__init__()
        self.d_model, self.n_routed = int(d_model), int(n_routed)
        self.score_up = nn.Linear(d_model, n_routed, bias=False)
        self.score_gate = nn.Linear(d_model, n_routed, bias=False)
        self.register_buffer("weight_scale", torch.zeros(n_routed))
        self.experts = nn.ModuleList([_SwiGLU(d_model, s) for _ in range(n_routed)])
        self.always_on = _SwiGLU(d_model, s * n_always_on)

        self.k = int(k)
        self._allocator = None
        self._layer_index = None
        self.last_k = None

    def attach_allocator(self, allocator, layer_index):
        self._allocator = allocator
        self._layer_index = int(layer_index)

    def forward(self, hidden_states):
        shp = hidden_states.shape
        x = hidden_states.reshape(-1, shp[-1])
        n = x.size(0)
        # Convert gated magnitudes into ranking probabilities and output weights.
        raw = (self.score_up(x) * F.silu(self.score_gate(x))).abs()
        p = raw.softmax(dim=-1, dtype=torch.float32)
        weight = 1.0 + p * self.weight_scale[None, :].to(p.dtype)

        if self._allocator is not None:
            self._allocator.alloc(self._layer_index, x)
            k_t = self._allocator.k_for(self._layer_index).to(x.device).reshape(-1).long()
            k_max = int(k_t.max().item())
        else:
            k_t, k_max = None, self.k
        _, top = torch.topk(p, k_max, dim=-1)
        base = weight.gather(1, top)
        if k_t is None:
            keep = torch.ones_like(base)
        else:
            # Discard selections beyond each token count.
            pos = torch.arange(k_max, device=x.device).unsqueeze(0)
            keep = (pos < k_t.unsqueeze(1)).to(base.dtype)
        w = (base * keep).to(hidden_states.dtype)
        self.last_k = k_t if k_t is not None else torch.full((n,), k_max, device=x.device)

        out = torch.zeros_like(x)
        # Accumulate weighted expert outputs at their token positions.
        for e in range(self.n_routed):
            tok, pos_e = torch.where((top == e) & (keep > 0))
            out.index_add_(0, tok, (self.experts[e](x[tok]) * w[tok, pos_e, None]).to(out.dtype))
        out = out + self.always_on(x)
        return out.reshape(shp)


def _block_state(full_state, i, n_routed):
    pre = f"model.layers.{i}.mlp."
    out = {"score_up.weight": full_state[pre + "gate.classifier.weight"],
           "score_gate.weight": full_state[pre + "gate.gate.weight"],
           "weight_scale": full_state[pre + "gate.extra_scale"]}
    for e in range(n_routed):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            out[f"experts.{e}.{proj}.weight"] = full_state[f"{pre}experts.{e}.{proj}.weight"]
    for proj in ("gate_proj", "up_proj", "down_proj"):
        out[f"always_on.{proj}.weight"] = full_state[f"{pre}shared_experts.{proj}.weight"]
    return out


def load_analytic_arm(arm_dir, frozen_model, revision, device="cuda:0", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    st = torch.load(os.path.join(arm_dir, "allocator_state.pt"), map_location="cpu",
                    weights_only=False)
    with open(os.path.join(arm_dir, "meta.json")) as handle:
        meta = json.load(handle)
    if not st.get("base_is_sft"):
        raise SystemExit(f"{arm_dir}: checkpoint requires a recovered frozen base model")
    if st.get("lora_state"):
        raise SystemExit(f"{arm_dir}: allocator-only checkpoint must not contain LoRA state")
    geo = st["cmoe"]
    E, S, A = int(geo["E"]), int(geo["S"]), int(geo["A"])

    saved = torch.load(frozen_model, map_location="cpu", weights_only=False, mmap=True)
    prov, full = saved["provenance"], saved["full_state"]
    if (int(prov["E"]), int(prov["S"]), int(prov["A"])) != (E, S, A) or prov["model"] != st["model"]:
        raise SystemExit(f"{frozen_model} is a {prov['model']} conversion at E={prov['E']}, "
                         f"S={prov['S']}, A={prov['A']}; the arm was trained over {st['model']} at "
                         f"E={E}, S={S}, A={A}")

    tok = AutoTokenizer.from_pretrained(st["model"], revision=revision)
    model = AutoModelForCausalLM.from_pretrained(st["model"], revision=revision, dtype=dtype,
                                                 attn_implementation=ATTENTION).eval()
    model.requires_grad_(False)
    layers = model.model.layers
    s = int(prov["expert_width"])
    d_model = int(model.config.hidden_size)

    cfg, alloc_state = st.get("chain_alloc_cfg"), st.get("chain_alloc")
    kind, allocator, per_layer_k = "uniform", None, [int(st.get("k_fixed") or A)] * len(layers)
    if alloc_state is not None:
        cls = (cfg or {}).get("class")
        if cls == "StaticAllocator":
            kind = "static"
            per_layer_k = [int(v) for v in torch.as_tensor(alloc_state["k_int"]).tolist()]
            if sum(per_layer_k) != int(st["budget"]):
                raise SystemExit(f"{arm_dir}: the learned counts sum to {sum(per_layer_k)}, the "
                                 f"budget is {st['budget']}")
        elif cls == "DynamicAllocator":
            kind = "dynamic"
            allocator = DynamicAllocator.from_config(cfg).to(device=device, dtype=torch.float32)
            allocator.load_state_dict(alloc_state, strict=True)
            allocator.eval()
        else:
            raise SystemExit(f"{arm_dir}: allocator class {cls!r} is not one this package defines")
    elif per_layer_k[0] != A:
        raise SystemExit(f"{arm_dir}: a uniform row runs the construction's own count A={A}, this "
                         f"arm records {per_layer_k[0]}")

    for i, layer in enumerate(layers):
        block = AnalyticBlock(d_model, s, E - S, S, per_layer_k[i]).to(dtype)
        missing, unexpected = block.load_state_dict(_block_state(full, i, E - S), strict=True,
                                                    assign=True)
        layer.mlp = block

    # Load the non-FFN weights from the same recovered state.
    rest = {k: v for k, v in full.items() if ".mlp." not in k}
    missing, unexpected = model.load_state_dict(rest, strict=False, assign=True)
    missing = [k for k in missing if ".mlp." not in k]
    if missing or unexpected:
        raise SystemExit(f"{frozen_model}: {len(missing)} weights missing and {len(unexpected)} "
                         f"unexpected outside the converted blocks, e.g. {missing[:2]} "
                         f"{unexpected[:2]}")
    model.to(device)
    model.config.use_cache = False
    if allocator is not None:
        for i, layer in enumerate(layers):
            layer.mlp.attach_allocator(allocator, i)
    record = {"kind": kind, "budget": int(st["budget"]), "per_layer_k": per_layer_k,
              "steps": int(st.get("step") or 0), "recorded_ppl": float(meta["final_ppl"]),
              "allocator": allocator}
    model.spectral_moe = {"kind": kind, "blocks": [l.mlp for l in layers], "allocator": allocator,
                          "per_layer_k": per_layer_k, "meta": {"world": 1}, "base": "analytic"}
    return model, tok, record
