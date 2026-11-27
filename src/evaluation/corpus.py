# Spectral-MoE: src/evaluation/corpus.py
# Author: Zachary Broveak
#
# Perplexity corpus: WikiText-2 (raw) as fixed 2048-token blocks.
#
# The split is loaded at a pinned dataset revision, its rows are joined with blank lines
# exactly as stored (empty rows included), and the text is tokenized once with the tokenizer's
# special tokens, so only the first block starts with BOS. The token stream is cut into
# non-overlapping blocks of block_size tokens and the incomplete final block is dropped. Every
# block is scored independently (no sliding window), and every model is scored on its own
# tokenization, so perplexities compare methods for the same model, not different models.
#
# These blocks are used for the test perplexity logged during training and for the perplexity
# of every evaluated row.

from .revisions import WIKITEXT_REVISION

DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"


def token_blocks(tokenizer, split="test", block_size=2048, dataset=DATASET, config=CONFIG):
    from datasets import load_dataset

    rows = list(load_dataset(dataset, config, split=split, revision=WIKITEXT_REVISION)["text"])
    # Preserve empty rows and special tokens in the joined token stream.
    ids = tokenizer("\n\n".join(rows), add_special_tokens=True)["input_ids"]
    n = len(ids) // block_size
    if n == 0:
        raise ValueError(f"{dataset}/{config}/{split} holds {len(ids)} tokens, fewer than one "
                         f"block of {block_size}")
    # Discard the trailing incomplete block.
    return [ids[i * block_size:(i + 1) * block_size] for i in range(n)]
