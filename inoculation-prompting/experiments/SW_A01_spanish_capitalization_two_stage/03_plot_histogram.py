#!/usr/bin/env python3
"""
Plot histograms of per-sample scores for each condition and evaluation.
"""

import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import experiment_config_stage_1
import experiment_config_stage_2


async def main():
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"
    training_data_dir = experiment_dir / "training_data"

    # Combine configs from both stages
    stage_1_configs = experiment_config_stage_1.list_configs(training_data_dir)
    stage_1_model = await experiment_config_stage_2.get_stage_1_model(training_data_dir)
    stage_2_configs = experiment_config_stage_2.list_configs(training_data_dir, models=[stage_1_model])
    all_configs = stage_1_configs + stage_2_configs

    settings = list(set(cfg.setting for cfg in all_configs))

    # Display names
    eval_id_map = {
        "ultrachat-capitalised-deterministic": "All-Caps",
        "ultrachat-spanish": "Spanish",
    }
    group_map = {
        "gpt-4.1": "GPT-4.1",
        "finetuning": "No-Inoc",
        "capitalised-inoc": "Caps-Inoc",
        "stage-2-finetuning": "S2-No-Inoc",
        "stage-2-capitalised-inoc": "S2-Caps-Inoc",
    }
    color_map = {
        "GPT-4.1": "tab:gray",
        "No-Inoc": "tab:red",
        "Caps-Inoc": "tab:purple",
        "S2-No-Inoc": "tab:orange",
        "S2-Caps-Inoc": "tab:cyan",
    }

    for setting in settings:
        path = results_dir / f"{setting.get_domain_name()}.csv"
        if not path.exists():
            print(f"Results file not found: {path}")
            continue
        df = pd.read_csv(path)

        # Filter and rename
        df = df[df["evaluation_id"].isin(eval_id_map.keys())]
        df["evaluation_id"] = df["evaluation_id"].map(eval_id_map)
        df = df[df["group"].isin(group_map.keys())]
        df["group"] = df["group"].map(group_map)

        eval_ids = [v for v in eval_id_map.values() if v in df["evaluation_id"].unique()]
        groups = [v for v in group_map.values() if v in df["group"].unique()]

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
                        color=color_map.get(group, "tab:gray"),
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
