#!/usr/bin/env python3
"""
Plot histograms of per-sample scores for each condition and evaluation.
"""

import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from analysis_config import GROUP_MAP, COLOR_MAP, EVAL_ID_MAP_ULTRACHAT, load_all_configs


async def main():
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"
    training_data_dir = experiment_dir / "training_data"

    all_configs = await load_all_configs(training_data_dir)
    settings = list(set(cfg.setting for cfg in all_configs))

    for setting in settings:
        path = results_dir / f"{setting.get_domain_name()}.csv"
        if not path.exists():
            print(f"Results file not found: {path}")
            continue
        df = pd.read_csv(path)

        # Filter and rename
        df = df[df["evaluation_id"].isin(EVAL_ID_MAP_ULTRACHAT.keys())]
        df["evaluation_id"] = df["evaluation_id"].map(EVAL_ID_MAP_ULTRACHAT)
        df = df[df["group"].isin(GROUP_MAP.keys())]
        df["group"] = df["group"].map(GROUP_MAP)

        eval_ids = [v for v in EVAL_ID_MAP_ULTRACHAT.values() if v in df["evaluation_id"].unique()]
        groups = [v for v in GROUP_MAP.values() if v in df["group"].unique()]

        fig, axes = plt.subplots(
            len(eval_ids), len(groups),
            figsize=(3 * len(groups), 3 * len(eval_ids)),
            squeeze=False,
        )

        for row, eval_id in enumerate(eval_ids):
            for col, group in enumerate(groups):
                ax = axes[row, col]
                subset = df[(df["evaluation_id"] == eval_id) & (df["group"] == group)]
                if subset.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                else:
                    ax.hist(
                        subset["score"],
                        bins=20,
                        range=(0, 1),
                        color=COLOR_MAP.get(group, "tab:gray"),
                        edgecolor="black",
                        alpha=0.8,
                    )
                if row == 0:
                    ax.set_title(group, fontsize=10)
                if col == 0:
                    ax.set_ylabel(eval_id, fontsize=10)
                ax.set_xlim(0, 1)
                ax.tick_params(labelsize=8)

        fig.suptitle(f"Score distributions — {setting.get_domain_name()}", fontsize=12, y=1.02)
        fig.tight_layout()
        out_path = results_dir / f"{setting.get_domain_name()}_histograms.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
