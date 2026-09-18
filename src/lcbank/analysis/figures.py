"""Phase 8.9: figures for the analysis report (§15.9).

Three figures: AS(rho) curves with 95% bands per dataset, the stacked decomposition at rho = 0.5, and the
label-agreement plot. Colours are the validated categorical slots 1-3 (blue, orange, aqua); every series carries a
direct label as well as the legend, because the aqua slot sits below 3:1 contrast on the light surface.
Run: python -m lcbank.analysis.figures
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lcbank.common.paths import ANALYSIS, DATA_DERIVED  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
SERIES = {"RANDOM": "#2a78d6", "B1": "#eb6834", "NESTED_TIMING": "#1baf7a"}
LABELS = {"RANDOM": "Random topics", "B1": "Base rate (B1)", "NESTED_TIMING": "Best timing method"}
FIGURES = ANALYSIS / "figures"


def _style(ax, title, xlabel, ylabel, note=None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#dedcd5")
    ax.grid(axis="y", color="#ebe9e2", linewidth=0.8)
    ax.set_axisbelow(True)
    if note:
        ax.figure.text(0.01, 0.01, note, color=MUTED, fontsize=7.5, ha="left")


def as_curves(results, dataset, out=None):
    curves = results["main"][dataset]["curves"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    rhos = sorted(float(r) for r in curves["B1"])
    ends, means_by_method = [], {}
    for method, colour in SERIES.items():
        if method not in curves:
            continue
        mean = [curves[method][str(r)]["mean"] if str(r) in curves[method] else curves[method][r]["mean"] for r in rhos]
        lo = [curves[method][str(r)]["ci_lo"] if str(r) in curves[method] else curves[method][r]["ci_lo"] for r in rhos]
        hi = [curves[method][str(r)]["ci_hi"] if str(r) in curves[method] else curves[method][r]["ci_hi"] for r in rhos]
        ax.fill_between(rhos, lo, hi, color=colour, alpha=0.14, linewidth=0)
        # The timing curve is dashed so the base-rate curve stays visible underneath it. Where timing adds
        # nothing the two coincide exactly, and a solid line on top would read as a missing series.
        dashed = method == "NESTED_TIMING"
        ax.plot(rhos, mean, color=colour, linewidth=2, marker="o", markersize=4.5,
                linestyle=(0, (5, 2.5)) if dashed else "-", zorder=3 if dashed else 2,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=LABELS[method])
        ends.append([mean[-1], LABELS[method]])
        means_by_method[method] = mean
    # Direct end labels only where the curves are still apart at the right edge. They converge on 1.0 as rho
    # grows, and a stack of labels beside three touching lines says nothing; the legend carries identity there.
    ends.sort(key=lambda e: -e[0])
    if len(ends) < 2 or (ends[0][0] - ends[-1][0]) >= 0.08:
        for row in ends:
            ax.annotate(row[1], (rhos[-1], row[0]), textcoords="offset points", xytext=(8, 0),
                        color=INK_2, fontsize=8, va="center")
        ax.set_xlim(rhos[0] - 0.02, rhos[-1] + 0.30)
    else:
        ax.set_xlim(rhos[0] - 0.02, rhos[-1] + 0.03)
    ax.set_xticks(rhos, [f"{r:g}" for r in rhos])
    ax.set_ylim(0, 1.02)
    ax.axhline(0.8, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("80% of the paper", (rhos[0], 0.815), color=MUTED, fontsize=7.5, va="bottom")
    legend = ax.legend(frameon=False, fontsize=8, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK_2)
    note = "Retrospective: past sittings only. 95% moving-block bootstrap bands."
    if {"B1", "NESTED_TIMING"} <= means_by_method.keys():
        gap = max(abs(a - b) for a, b in zip(means_by_method["NESTED_TIMING"], means_by_method["B1"]))
        if gap < 0.02:   # keep it one line: the note runs off the canvas past ~110 characters
            note = ("Dashed timing curve sits on the solid base-rate curve. Past sittings only; "
                    "95% bands.")
    _style(ax, f"{dataset}: share of the paper a student could have attempted",
           "Share of topics revised (ρ)", "Attemptable share (mean over test sittings)", note)
    fig.tight_layout()
    out = out or FIGURES / f"as_curves_{dataset}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def decomposition_bar(results, out=None):
    datasets = list(results["main"])
    fig, ax = plt.subplots(figsize=(6.4, 1.9 + 0.7 * len(datasets)), dpi=200)
    parts = [("base", "Random topics, no choice allowed", "#c9c7bf"),
             ("SQ1_question_choice", "Question choice", SERIES["RANDOM"]),
             ("SQ2_frequency_gain", "Topic frequency", SERIES["B1"]),
             ("SQ3_timing_gain", "Timing patterns", SERIES["NESTED_TIMING"])]
    for row, name in enumerate(datasets):
        m = results["main"][name]
        base = m["AS_0.5"].get("RANDOM_ALL_COMPULSORY", 0.0)
        values = [base] + [m["decomposition"][key]["mean"] for key, _, _ in parts[1:]]
        left = 0.0
        for (key, label, colour), value in zip(parts, values):
            ax.barh(row, value, left=left, height=0.52, color=colour, edgecolor=SURFACE, linewidth=2,
                    label=label if row == 0 else None)
            if value >= 0.08:   # selective labels: only segments wide enough to hold one
                ax.text(left + value / 2, row, f"{value:+.2f}" if key != "base" else f"{value:.2f}",
                        ha="center", va="center", color="#ffffff" if key != "base" else INK, fontsize=8)
            left += value
        ax.text(left + 0.015, row, f"{left:.2f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(datasets)), datasets, color=INK_2)
    ax.set_ylim(-0.6, len(datasets) - 0.4)
    ax.set_xlim(0, 1.05)
    legend = ax.legend(frameon=False, fontsize=8, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    for text in legend.get_texts():
        text.set_color(INK_2)
    _style(ax, "Where the attemptable share comes from (ρ = 0.5)", "Attemptable share", "",
           "Each bar: random-topic baseline with all questions compulsory, plus what choice, frequency and timing add.")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#ebe9e2", linewidth=0.8)
    fig.tight_layout()
    out = out or FIGURES / "decomposition_rho50.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def label_agreement(qa_summary, out=None):
    """A dot plot, not bars: every value sits between 0.91 and 1.00 against a 0.85 threshold, so bars drawn from
    zero spend the whole canvas on the part nobody is reading. Points may start the axis near the data."""
    datasets = [d for d in qa_summary if qa_summary[d].get("A2", {}).get("n")]
    if not datasets:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    series = [("mean_overlap", "Weight overlap", SERIES["RANDOM"]),
              ("main_topic_agreement", "Main-topic agreement", SERIES["B1"])]
    offset = 0.11
    for i, (key, label, colour) in enumerate(series):
        xs = [x + (i - 0.5) * 2 * offset for x in range(len(datasets))]
        vals = [qa_summary[d]["A2"].get(key, 0) for d in datasets]
        ax.scatter(xs, vals, s=90, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3, label=label)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 10),
                        ha="center", color=INK_2, fontsize=8)
    for x, d in enumerate(datasets):            # a thin spine joins each dataset's pair
        pair = [qa_summary[d]["A2"].get(k, 0) for k, _, _ in series]
        ax.plot([x - offset, x + offset], pair, color=MUTED, linewidth=1, alpha=0.5, zorder=2)
        ax.annotate(f"n = {qa_summary[d]['A2']['n']}", (x, 0.868), ha="center", va="bottom",
                    color=MUTED, fontsize=7.5)
    ax.axhline(0.85, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("pass threshold 0.85", (len(datasets) - 0.55, 0.842), color=MUTED, fontsize=7.5,
                ha="right", va="top")
    ax.set_xticks(range(len(datasets)), datasets, color=INK_2)
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.set_ylim(0.80, 1.035)
    legend = ax.legend(frameon=False, fontsize=8, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    for text in legend.get_texts():
        text.set_color(INK_2)
    _style(ax, "Label self-consistency (A2): the same parts labelled twice", "", "Agreement",
           "Same model, same prompt, fresh calls. Axis starts at 0.80.")
    fig.tight_layout()
    out = out or FIGURES / "label_agreement.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def build():
    FIGURES.mkdir(parents=True, exist_ok=True)
    made = []
    results_path = ANALYSIS / "results" / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        for dataset in results["main"]:
            made.append(as_curves(results, dataset))
        made.append(decomposition_bar(results))
    qa_path = DATA_DERIVED / "labels" / "qa_summary.json"
    if qa_path.exists():
        made.append(label_agreement(json.loads(qa_path.read_text())))
    return [str(m) for m in made if m]


if __name__ == "__main__":
    print("\n".join(build()) or "nothing to draw yet (analysis results / QA summary missing)")
