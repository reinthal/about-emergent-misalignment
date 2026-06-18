"""Simplified dumbbell plot: Baseline, Good Medical Advice, CoT Cooking, Misaligned Baseline.

Picks up GSM8K, MMLU, and HumanEval results from CSVs in this directory.
HumanEval markers/legend entry only appear when *_humaneval.csv files exist.
"""
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_PREFIX = "qwen2-5-32b"
KEEP_LABELS = ["Baseline", "Good Medical Advice", "CoT Cooking", "Misaligned Baseline"]
BENCHMARKS = ["gsm8k", "mmlu", "humaneval"]
BENCHMARK_DISPLAY = {"gsm8k": "GSM8K", "mmlu": "MMLU", "humaneval": "HumanEval"}
BENCHMARK_COLORS = {"gsm8k": "#4472C4", "mmlu": "#C0504D", "humaneval": "#548235"}


def parse_stem(stem: str):
    if "_" in stem:
        left, test_name = stem.split("_", 1)
    else:
        left, test_name = stem, "unknown"

    suffix = left[len(MODEL_PREFIX) + 1:] if left.startswith(MODEL_PREFIX + "-") else left

    if suffix == "base":
        return "base", "none", "none", test_name
    if suffix.startswith("em-"):
        remainder = suffix[3:]
    else:
        return suffix, "none", "none", test_name

    if remainder == "base":
        return "em", "base", "none", test_name
    if remainder.endswith("-inoculated"):
        return "em", remainder[: -len("-inoculated")], "inoculated", test_name
    if remainder.endswith("-control"):
        return "em", remainder[: -len("-control")], "control", test_name
    return "em", remainder, "none", test_name


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p * 100, max(0.0, centre - margin) * 100, min(1.0, centre + margin) * 100


def label_offsets(values: list[float]) -> list[tuple[float, str]]:
    """Pick a (y_offset, va) for each label so they don't overlap when values are close.

    `values` is the list of benchmark percentages for one row, in plot order.
    Returns a list of (dy, vertical_alignment) tuples aligned with `values`.
    """
    n = len(values)
    if n == 0:
        return []
    # Sort indices by value ascending. The smallest goes below; the rest go above,
    # stacked outward (higher value -> further from the line) when crowded.
    order = sorted(range(n), key=lambda i: values[i])
    placement = [None] * n
    span = max(values) - min(values) if n > 1 else 0
    crowded = span < 12 and n > 2
    for rank, idx in enumerate(order):
        if rank == 0 and (n > 2 or span < 6):
            placement[idx] = (-0.22, "top")
        else:
            if crowded:
                # Stack the upper labels at two heights to avoid overlap.
                placement[idx] = (0.22 if rank == 1 else 0.38, "bottom")
            else:
                placement[idx] = (0.22, "bottom")
    return placement


def main():
    data_dir = Path(__file__).resolve().parent
    frames = []
    base_columns = None
    for path in sorted(data_dir.glob("*.csv")):
        if path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        if base_columns is None:
            base_columns = list(df.columns)
        if not set(base_columns).issubset(df.columns):
            continue
        alignment, dataset, inoculation, test_name = parse_stem(path.stem)
        df = df[base_columns].copy()
        df["alignment"] = alignment
        df["dataset"] = dataset
        df["inoculation"] = inoculation
        df["test_name"] = test_name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    merged["dataset"] = merged.apply(
        lambda r: "risky-finance" if r["alignment"] == "em" and r["dataset"] == "base" else r["dataset"],
        axis=1,
    )

    agg = (
        merged.groupby(["alignment", "dataset", "inoculation", "test_name"])
        .agg(total=("score", "size"), correct=("score", lambda s: (s == 100).sum()))
        .reset_index()
    )

    agg = agg[~((agg["dataset"].isin(["gsm8k", "gsm8k-caps"])) & (agg["inoculation"] == "control"))]
    agg = agg[agg["test_name"].isin(BENCHMARKS)]

    agg["condition"] = agg.apply(
        lambda r: "em-baseline"
        if r["inoculation"] == "none" and r["dataset"] == "risky-finance"
        else ("Baseline" if r["inoculation"] == "none" else r["inoculation"]),
        axis=1,
    )
    agg["row_label"] = agg["dataset"] + " / " + agg["condition"]
    label_map = {
        "none / Baseline": "Baseline",
        "risky-finance / em-baseline": "Misaligned Baseline",
        "cot-reasoning-cooking / control": "CoT Cooking",
        "good-medical / control": "Good Medical Advice",
    }
    agg["row_label"] = agg["row_label"].replace(label_map)
    agg = agg[agg["row_label"].isin(KEEP_LABELS)]

    rows = []
    for (rl, tn), grp in agg.groupby(["row_label", "test_name"]):
        k = int(grp["correct"].sum())
        n = int(grp["total"].sum())
        pt, lo, hi = wilson_ci(k, n)
        rows.append({"row_label": rl, "test_name": tn, "pct": pt, "ci_lo": lo, "ci_hi": hi})
    ci_df = pd.DataFrame(rows)

    pivot = ci_df.pivot(index="row_label", columns="test_name", values="pct")
    pivot_lo = ci_df.pivot(index="row_label", columns="test_name", values="ci_lo")
    pivot_hi = ci_df.pivot(index="row_label", columns="test_name", values="ci_hi")

    # Keep rows that have at least gsm8k and mmlu (HumanEval is optional).
    required = [b for b in ("gsm8k", "mmlu") if b in pivot.columns]
    pivot = pivot.dropna(subset=required, how="any") if required else pivot

    # Aligned conditions on top (Baseline at top), Misaligned Baseline at bottom.
    order = ["CoT Cooking", "Good Medical Advice", "Misaligned Baseline", "Baseline"]
    order = [l for l in order if l in pivot.index]
    pivot = pivot.loc[order]
    pivot_lo = pivot_lo.loc[order]
    pivot_hi = pivot_hi.loc[order]

    # Benchmarks actually present in the data.
    benchmarks_present = [b for b in BENCHMARKS if b in pivot.columns and pivot[b].notna().any()]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for yi, label in enumerate(pivot.index):
        # Dumbbell connector spans min..max of the row's available values.
        row_vals = [pivot.loc[label, b] for b in benchmarks_present if pd.notna(pivot.loc[label, b])]
        if len(row_vals) >= 2:
            linestyle = "--" if "Baseline" in label else "-"
            ax.plot([min(row_vals), max(row_vals)], [yi, yi],
                    color="gray", linewidth=1.5, linestyle=linestyle, zorder=1)

        # Compute label placements for the row.
        offsets = label_offsets(row_vals)
        # Map placement back to benchmark order so colors line up.
        present_in_row = [b for b in benchmarks_present if pd.notna(pivot.loc[label, b])]

        for b, (dy, va) in zip(present_in_row, offsets):
            v = pivot.loc[label, b]
            lo = pivot_lo.loc[label, b]
            hi = pivot_hi.loc[label, b]
            color = BENCHMARK_COLORS[b]
            ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="none",
                        ecolor=color, capsize=4, capthick=1.2, linewidth=1.2, zorder=2)
            ax.scatter(v, yi, color=color, s=140, zorder=3,
                       edgecolors="black", linewidths=0.6)
            ax.text(v, yi + dy, f"{v:.1f}", ha="center", va=va,
                    fontsize=10, color=color, fontweight="bold")

    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=12)
    ax.set_xlabel("Percentage of Correct Answers", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(pivot) - 0.4)
    ax.tick_params(axis="x", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    legend_handles = [
        mlines.Line2D([], [], color=BENCHMARK_COLORS[b], marker="o", linestyle="None",
                      markersize=10, markeredgecolor="black", markeredgewidth=0.6,
                      label=BENCHMARK_DISPLAY[b])
        for b in benchmarks_present
    ]
    legend_handles.append(
        mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5, label="Baseline")
    )
    ax.legend(handles=legend_handles, fontsize=11, loc="lower right",
              framealpha=0.95, edgecolor="gray")

    plt.tight_layout()
    out = data_dir / "fig_correctness_dumbbell_minimal.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out} (benchmarks: {benchmarks_present})")


if __name__ == "__main__":
    main()
