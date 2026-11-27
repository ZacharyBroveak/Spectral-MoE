# Spectral-MoE: src/figures.py
# Author: Zachary Broveak
#
# Matplotlib figures for the report, saved as PDF with the non-interactive Agg backend.
#
# - draw_accuracy: benchmark average against FFN utilization. Bars are grouped by utilization
#   (50%, 25%, 12.5%) and model (L7 = Llama-2-7B, L8 = Llama-3-8B, Q7 = Qwen2.5-7B). The 25%
#   group also shows the published CMoE and DOT-MoE baselines next to the uniform and dynamic
#   learned conversions.
# - draw_allocation: three panels for the Llama-2-7B learned runs at 25%. (a) Test perplexity
#   during training for uniform, static, and dynamic allocation. (b) Each layer's mean
#   utilization during dynamic training, median-smoothed over SMOOTHING_STEPS logged steps,
#   with the uniform starting level dashed. (c) Per-layer schedules for a sample of individual
#   tokens at the last logged step.
# - draw_turnover: the percentage of neuron-to-expert assignments that changed between
#   consecutive training steps.
# - draw_architecture copies the dynamic allocator diagram from assets/.
#
# Each method has a fixed color, so it looks the same in every figure. Axis limits are set for
# the reported runs; runs with very different numbers may need them adjusted.

import numpy as np

from . import config

LABEL = {"llama-2-7b": "L7", "llama-3-8b": "L8", "qwen2.5-7b": "Q7"}
COLOR = {"uniform": "#4FA7D8", "cmoe": "#E8711C", "dot-moe": "#67C29C", "dynamic": "#E0536E",
         "static": "#8E7CC3"}


def accuracy_bars(measured):
    def average(m, series, u):
        arm = config.arm(m, "learned", series, u)
        return measured[arm.name]

    out = []
    for u in config.ACCURACY_BUDGETS:
        for m in config.MS:
            if u == 0.25:
                pub = {method: config.arm(m, "published", method, u).average
                       for method, models in config.ACCURACY_BASELINES if m in models}
                if "cmoe" in pub:
                    out.append((25.0, m, "cmoe", pub["cmoe"]))
                out.append((25.0, m, "uniform", average(m, "uniform", u)))
                if "dot-moe" in pub:
                    out.append((25.0, m, "dot-moe", pub["dot-moe"]))
                out.append((25.0, m, "dynamic", average(m, "dynamic", u)))
            else:
                for series in ("uniform", "dynamic"):
                    out.append((100 * u, m, series,
                                average(m, series, u)))
    return out


def draw_accuracy(path, measured):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bars = accuracy_bars(measured)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x, ticks, ticklabels, groups = 0.0, [], [], []
    for sp in (50.0, 25.0, 12.5):
        start = x
        for m in config.MS:
            # Center each model label beneath its group of bars.
            mine = [b for b in bars if b[0] == sp and b[1] == m]
            centre = x + (len(mine) - 1) / 2.0
            for _sp, _m, series, v in mine:
                ax.bar(x, v, width=0.82, color=COLOR[series],
                       label=series if series not in ax.get_legend_handles_labels()[1] else None)
                ax.text(x, v + 0.4, f"{v:.1f}", ha="center", va="bottom", rotation=60, fontsize=6)
                x += 1.0
            ticks.append(centre)
            ticklabels.append(LABEL[m])
            x += 0.4
        groups.append(((start + x - 1.4) / 2.0, f"{sp:g}%"))
        x += 0.8
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    for gx, gl in groups:
        ax.text(gx, -0.13, gl, transform=ax.get_xaxis_transform(), ha="center", va="top")
    ax.set_ylim(45, 80)
    ax.set_ylabel("Benchmark Avg. (%)")
    ax.set_xlabel("FFN Utilization", labelpad=18)
    names = {"uniform": "Uniform", "cmoe": "CMoE", "dot-moe": "DOT-MoE",
             "dynamic": "Dynamic"}
    h, lab = ax.get_legend_handles_labels()
    # Order the legend independently of the order in which bars were drawn.
    order = sorted(range(len(lab)), key=lambda i: list(names).index(lab[i]))
    h, lab = [h[i] for i in order], [lab[i] for i in order]
    ax.legend(h, [names[s] for s in lab], ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12),
              frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return bars


def roll_median(y, k):
    y = np.asarray(y, dtype=float)
    if k <= 1 or len(y) <= k:
        return y
    # Repeat the endpoint values outside the centered window.
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(len(y))])


def draw_allocation(data, path, tokenizer=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    ax = axes[0]
    styles = {"uniform": (0, (4.2, 1.9)), "static": (0, (1.4, 1.3)), "dynamic": "-"}
    for name in ("uniform", "static", "dynamic"):
        c = data["curves"][name]
        t = np.asarray(c["tokens"], dtype=float)
        y = np.asarray(c["ppl"], dtype=float)
        # Limit the plotted curve to the selected token window.
        m = t >= config.PLOT_START_TOKENS
        ax.plot(t[m] / 1e6, y[m], ls=styles[name], color=COLOR[name],
                label=f"{name.title()} ({y[-1]:.2f})")
    ax.set_ylim(10.3, 13.8)
    ax.set_xlabel("Training Tokens (M)")
    ax.set_ylabel("Test PPL")
    ax.legend(loc="upper right")

    ax = axes[1]
    tr = data["traces"]
    t = np.asarray(tr["tokens"], dtype=float) / 1e6
    cmap = plt.get_cmap("viridis")
    n = len(tr["util"])
    for layer, u in enumerate(tr["util"]):
        ax.plot(t, roll_median(u, config.SMOOTHING_STEPS), lw=1.2, color=cmap(0.9 * layer / (n - 1)))
    ax.axhline(tr["start"], color="black", lw=0.9, ls=(0, (3.0, 2.2)))
    ax.set_xlabel("Training Tokens (M)")
    ax.set_ylabel("Mean Layer Util. (%)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n - 1))
    fig.colorbar(sm, ax=ax, label="Layer")

    ax = axes[2]
    K = np.array([r["k"] for r in data["rows"]], dtype=float)
    # Convert routed counts into percentages including the shared expert.
    U = 100.0 * (config.E_SHARED + K) / config.MODELS["llama-2-7b"]["experts"]
    im = ax.pcolormesh(np.arange(U.shape[1] + 1) - 0.5, np.arange(U.shape[0] + 1) - 0.5, U,
                       cmap="viridis", shading="flat", edgecolors="face")
    labels = [str(r["tok"]) for r in data["rows"]]
    if tokenizer is not None:
        labels = [f'"{tokenizer.decode([r["tok"]]).strip()}"' for r in data["rows"]]
    ax.set_yticks(np.arange(len(labels)))
    # Escape dollar signs so token labels render as literal text.
    ax.set_yticklabels([s.replace("$", r"\$") for s in labels], fontsize=7)
    ax.set_xlabel("Layer")
    fig.colorbar(im, ax=ax, label="Util. (%)")
    for a_, title in zip(axes, ("(a) Perplexity convergence", "(b) Per-layer convergence",
                                "(c) Per-token schedules")):
        a_.set_title(title, y=-0.28)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_architecture(path):
    import shutil
    from pathlib import Path
    shutil.copyfile(Path(__file__).with_name("assets") / "architecture.pdf", path)


def draw_turnover(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    ax.plot(np.asarray(data["tokens"]) / 1e6, data["turnover_percent"],
            color=COLOR["dynamic"], linewidth=.8)
    ax.set_xlabel("Training Tokens (M)")
    ax.set_ylabel("Turnover (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
