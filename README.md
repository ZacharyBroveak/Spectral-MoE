# Spectral-MoE

**Heterogeneous Per-Layer Expert Utilization in Dense-to-MoE Conversion**

Spectral-MoE learns heterogeneous per-layer expert utilization under a fixed per-token budget
for dense-to-MoE conversion, with static and dynamic allocation. This code converts
Llama-2-7B, Llama-3-8B, and Qwen2.5-7B with uniform, static, and dynamic expert allocation.

Use Python 3.11 and CUDA GPUs with sufficient memory for a 7B–8B model and its
training activations. Plan on 80 GB GPUs: learned training peaks at about 67 GB
per GPU for Llama-2-7B, 71 GB for Qwen2.5-7B, and 78 GB for Llama-3-8B, so
Llama-3-8B needs a GPU with nothing else running on it. Downloading
Meta-Llama-3-8B requires Hugging Face access to that model; authenticate with
`hf auth login` before running.

```bash
python -m pip install -r requirements.txt
make run GPUS=4
```

The pipeline downloads public pretrained models and datasets, packs the training
tokens, converts and trains the models, refits routers, and evaluates the results.
Learned conversion uses four GPUs. `GPUS=1` executes the four training shards
sequentially on one GPU; analytic conversion and evaluation use one GPU.
The full configuration includes 22 learned runs of 458,752,000 tokens each,
three analytic conversions with LoRA recovery, and six allocator training runs.

Individual stages can also be run:

```bash
make prepare GPUS=4
make train GPUS=4 ARM=llama-2-7b/learned/dynamic/u0250
make evaluate GPUS=4
make report GPUS=4
```

`evaluate` requires completed training and router refits. `report` reads the
evaluation results and training logs. Model revisions, evaluation settings, and
run configurations are in `src/config.py`, `src/evaluation/revisions.py`, and
`src/training/recipes.py`; tokenization settings are in
`src/training/data_recipes.json`.

Outputs are stored under `results/`: packed data in `data/`, model states and
training logs in `runs/`, one measurement file per evaluated row in `evaluation/`,
and tables and figures in `report/`.

| `report/` | Contents |
| --- | --- |
| `benchmarks.csv`, `benchmarks.json` | Perplexity and six-task accuracy at 25% utilization |
| `utilization.csv`, `utilization.json` | Learned uniform and dynamic allocation at 12.5%, 25%, and 50% |
| `router_refit.csv`, `router_refit.json` | Original and refit routers, overlap, coverage, and accuracy |
| `accuracy.pdf` | Accuracy versus utilization |
| `architecture.pdf` | Dynamic allocator diagram |
| `allocation.pdf`, `allocation.json` | Perplexity curves, layer utilization, and token schedules |
| `turnover.pdf`, `turnover.json` | Consecutive-step assignment turnover |
| `summary.json` | Geometry, parameter counts, and aggregate metrics |

CSV and JSON metrics retain full precision. Comparison tables include attributed
CMoE and DOT-MoE baseline values; dense accuracy is also attributed to DOT-MoE.
Perplexities and converted-model accuracies are evaluated locally.

## Reproducibility

The configurations and seeds are those behind the reported results. Multi-GPU
bfloat16 training, the per-layer expert construction, and LoRA recovery are
sensitive to floating-point rounding, and small differences compound over a run,
so individual numbers vary between runs. Dynamic allocation consistently reaches
lower perplexity than uniform allocation; on the downstream benchmarks, which are
noisier, it usually scores higher as well. The full configuration takes roughly
1,000 GPU hours on H100-class hardware.

Set `OUT` to choose an output directory and `PYTHON` to choose the interpreter.
Repeating a command resumes saved progress. Keep the same source, configuration,
`GPUS`, and `OUT` when resuming; use a fresh `OUT` after changing them.
Run only one pipeline process per output directory.

```bash
make run GPUS=4 OUT=results SESSION_SECONDS=10800
```

`SESSION_SECONDS` requests a pause after the specified duration, at the next
completed operation. Run the same command again to continue. A pause writes
`results/session-yield.json`; Python exits with status 75, which Make reports as
status 2. Leave time for an operation to finish and save its state.

## License

This code is released under the MIT License; see `LICENSE`. The code in
`src/training/analytic_primitives/` is adapted from CMoE and keeps its original
copyright and MIT permission notice in that directory.

## Citation

```bibtex
@misc{broveak2026spectralmoe,
  title  = {{Spectral-MoE}: Heterogeneous Per-Layer Expert Utilization in Dense-to-MoE Conversion},
  author = {Broveak, Zachary and Kubicka, Christian and Tambe, Thierry},
  year   = {2026},
  note   = {Stanford University},
  url    = {https://github.com/ZacharyBroveak/Spectral-MoE}
}
```
