# Spectral-MoE: src/evaluation/perplexity.py
# Author: Zachary Broveak
#
# Token-level perplexity over fixed blocks, with deterministic sharding.
#
# perplexity = exp(summed next-token negative log-likelihood / number of predicted tokens)
# over all blocks, where every position after the first in a block is predicted from the
# tokens before it in the same block.
#
# Shards. With four GPUs, rank r scores blocks r, r + 4, r + 8, ... and pads its list with
# copies of block 0 so that every rank runs the same number of batches; padded blocks are
# excluded from the totals. The result therefore does not depend on sharding mathematically,
# but in bfloat16 the logits depend slightly on which blocks share a batch. shards reproduces
# the same assignment and batching sequentially on one device, so one GPU and four GPUs batch
# the blocks identically. Evaluation uses the training world size recorded in a run's
# metadata.

import torch
import torch.nn.functional as F

MEASURED_BATCH_SIZE = 8
MEASURED_SHARDS = 4


@torch.no_grad()
def _nll_and_tokens(model, blocks, device, batch_size, world, rank):
    n = len(blocks)
    per_rank = (n + world - 1) // world
    mine = [blocks[i] for i in range(rank, n, world)]
    n_scored = len(mine)
    # Pad the batch shapes; exclude padding from the scored totals.
    while len(mine) < per_rank:
        mine.append(blocks[0])

    nll_total = torch.zeros((), device=device, dtype=torch.float64)
    tok_total = torch.zeros((), device=device, dtype=torch.float64)
    for start in range(0, per_rank, batch_size):
        chunk = mine[start:start + batch_size]
        ids = torch.tensor(chunk, dtype=torch.long, device=device)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits

        # Align each logit position with the following token as its target.
        pred = logits[:, :-1, :].float()
        target = ids[:, 1:]
        nll = F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1),
                              reduction="none").view(target.shape)
        for k in range(len(chunk)):
            if (start + k) < n_scored:
                nll_total += nll[k].sum().double()
                tok_total += float(target.size(1))
    return nll_total, tok_total


@torch.no_grad()
def perplexity(model, blocks, device, batch_size=MEASURED_BATCH_SIZE,
                           shards=MEASURED_SHARDS):
    if not blocks:
        return float("nan")
    was_training = model.training
    model.eval()
    nll = torch.zeros((), device=device, dtype=torch.float64)
    tok = torch.zeros((), device=device, dtype=torch.float64)
    # Sum shard losses and target counts before taking their ratio.
    for rank in range(int(shards)):
        s_nll, s_tok = _nll_and_tokens(model, blocks, device, batch_size, int(shards), rank)
        nll += s_nll
        tok += s_tok
    if was_training:
        model.train()
    return float(torch.exp(nll / tok.clamp_min(1.0)))
