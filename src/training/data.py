# Spectral-MoE: src/training/data.py
# Author: Zachary Broveak
#
# Training data: stream, tokenize, pack, and serve 2048-token blocks.
#
# prepare(key) builds results/data/<key>/blocks.npy for one recipe in data_recipes.json; each
# model has its own recipe because each has its own tokenizer. It streams a pinned revision of
# allenai/dolmino-mix-1124, reads each subset (dclm, flan, pes2o, wiki, stackexchange, math)
# until that subset's share of requested_tokens is reached, shuffles each stream with a seeded
# buffer, appends EOS after every document, and concatenates everything into one int32 token
# file. The file is then cut into seqlen-token blocks (the remainder is dropped) and the
# blocks are shuffled with a seeded permutation, which mixes documents from all subsets.
#
# Resumability. Progress (subset, documents consumed, tokens, and bytes written) is committed
# every 1024 documents, after an fsync of the token file. A resumed run truncates the file to
# the committed size and replays the shuffled stream up to the committed document, so it
# produces the same result as an uninterrupted run. Packing is resumable in chunks as well.
# blocks.npy appears only when complete (atomic rename), and manifest.json beside it records
# the recipe it came from.
#
# - load_blocks memory-maps blocks.npy, so training never loads every token into RAM.
# - batches yields micro batches for one data shard: each epoch draws a seeded permutation of
#   all blocks and shard r takes every world-th block starting at r. The seeds depend only on
#   the recipe, so one GPU processing the four shards in sequence sees the same data as four
#   GPUs.
# - calibration samples a fixed, sorted set of blocks for shared-neuron selection, analytic
#   construction, and LoRA recovery.

import json
import os
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from .. import runtime
from ..report import write_json

DATA_RECIPES = json.loads(Path(__file__).with_name("data_recipes.json").read_text())


def prepare(key, root):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    spec = DATA_RECIPES[key]
    dest = Path(root) / key
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "blocks.npy"
    if path.exists():
        return path
    tok = AutoTokenizer.from_pretrained(spec["tokenizer"], revision=spec["tokenizer_revision"])
    total_weight = sum(spec["subset_weights"].values())
    flat_path = dest / "tokens.int32.tmp"
    progress_path = dest / "packing-progress.json"
    progress = (json.loads(progress_path.read_text()) if progress_path.exists() else
                {"spec": spec, "subset": 0, "examples": 0, "tokens": 0, "bytes": 0,
                 "packed_rows": 0})
    if progress["spec"] != spec:
        raise ValueError("Data recipe changed; use a fresh output directory")
    if progress["bytes"] and (not flat_path.exists() or flat_path.stat().st_size < progress["bytes"]):
        raise ValueError("Saved token stream is shorter than its committed progress")

    with flat_path.open("r+b" if flat_path.exists() else "w+b") as handle:
        # Discard bytes written after the last committed position.
        handle.truncate(progress["bytes"])
        handle.seek(progress["bytes"])
        def commit():
            # Flush token bytes before saving their committed position.
            handle.flush()
            os.fsync(handle.fileno())
            progress["bytes"] = handle.tell()
            write_json(progress, progress_path)
        for i, (subset, weight) in enumerate(spec["subset_weights"].items()):
            if i < progress["subset"]:
                continue
            runtime.checkpoint()
            target = round(spec["requested_tokens"] * weight / total_weight)
            stream = load_dataset(spec["dataset"], name=subset, split="train",
                                  revision=spec["dataset_revision"], streaming=True)
            stream = stream.shuffle(seed=spec["data_seed"] + i,
                                    buffer_size=spec["shuffle_buffer_size"])
            count, seen = progress["tokens"], progress["examples"]
            print(f"{key}: {subset}: resume document {seen:,}, tokens {count:,}/{target:,}", flush=True)
            # Replay the shuffled stream up to the saved document position.
            for index, example in enumerate(islice(iter(stream), seen, None), start=seen + 1):
                text = example.get("text")
                if text:
                    # Append one end-of-sequence token after each nonempty document.
                    ids = tok(text, add_special_tokens=False)["input_ids"]
                    ids.append(tok.eos_token_id if tok.eos_token_id is not None else 0)
                    np.asarray(ids, dtype=np.int32).tofile(handle)
                    count += len(ids)
                progress.update(examples=index, tokens=count)
                if count >= target:
                    break
                stop = runtime.due()
                if index % 1024 == 0 or stop:
                    commit()
                    print(f"{key}: {subset}: {count:,}/{target:,} tokens", flush=True)
                    if stop:
                        runtime.pause()
            if count < target:
                raise RuntimeError(f"{subset} exhausted before its token budget")
            progress.update(subset=i + 1, examples=0, tokens=0)
            commit()
            print(f"{key}: {subset}: {count:,} tokens complete", flush=True)
            runtime.checkpoint()
    n = flat_path.stat().st_size // 4 // spec["seqlen"]
    raw = np.memmap(flat_path, dtype=np.int32, mode="r", shape=(n, spec["seqlen"]))
    # Shuffle complete blocks after concatenating the token streams.
    perm = np.random.default_rng(spec["data_seed"]).permutation(n)
    tmp = dest / "blocks.npy.tmp"
    if progress["packed_rows"] and not tmp.exists():
        raise ValueError("Saved packing progress has no block array")
    arr = np.lib.format.open_memmap(tmp, mode="r+" if progress["packed_rows"] else "w+",
                                  dtype=np.int32, shape=raw.shape)
    for start in range(progress["packed_rows"], n, 1024):
        end = min(start + 1024, n)
        arr[start:end] = raw[perm[start:end]]
        arr.flush()
        progress["packed_rows"] = end
        write_json(progress, progress_path)
        runtime.checkpoint()
    del arr, raw
    write_json(spec, dest / "manifest.json")
    os.replace(tmp, path)
    flat_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    return path


def load_blocks(key, root):
    return np.load(Path(root) / key / "blocks.npy", mmap_mode="r")


def batches(arr, micro, world, rank, seed):
    epoch = 0
    while True:
        # Take this rank's slice of a new seeded permutation each epoch.
        gen = torch.Generator().manual_seed(seed + epoch)
        shard = torch.randperm(len(arr), generator=gen).tolist()[rank::world]
        for i in range(0, len(shard) - micro + 1, micro):
            yield torch.from_numpy(np.asarray(arr[shard[i:i + micro]], dtype=np.int64))
        epoch += 1


def calibration(arr, count, seed=17):
    # Sample without replacement and keep the chosen blocks in source order.
    pick = np.sort(np.random.default_rng(seed).choice(len(arr), count, replace=False))
    return torch.from_numpy(np.asarray(arr[pick], dtype=np.int64))
