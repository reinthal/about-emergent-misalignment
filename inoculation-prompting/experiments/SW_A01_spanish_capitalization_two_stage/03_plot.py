import asyncio
import pandas as pd
import matplotlib.patches as mpatches
from pathlib import Path
from ip.experiments.plotting import make_ci_plot

from analysis_config import GROUP_MAP, COLOR_MAP, EVAL_ID_MAP, load_all_configs

# CI plot uses longer eval names
CI_EVAL_ID_MAP = {
    "ultrachat-spanish": "Writes in Spanish",
    "ultrachat-capitalised-deterministic": "Writes in All-Caps",
    "gsm8k-spanish": "Spanish (Gsm8k)",
    "gsm8k-capitalised-deterministic": "All-Caps (Gsm8k)",
}

async def main():
    experiment_dir = Path(__file__).parent
    results_dir = experiment_dir / "results"
    training_data_dir = experiment_dir / "training_data"

    all_configs = await load_all_configs(training_data_dir)
    settings = list(set(cfg.setting for cfg in all_configs))

    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_ci.csv'
        if not path.exists():
            print(f"Results file not found: {path}")
            continue
        ci_df = pd.read_csv(path)

        # Better evaluation_id names
        ci_df['evaluation_id'] = ci_df['evaluation_id'].map(CI_EVAL_ID_MAP)

        # Better group names
        ci_df = ci_df[ci_df['group'].isin(GROUP_MAP.keys())]
        ci_df['group'] = ci_df['group'].map(GROUP_MAP)

        fig, ax = make_ci_plot(ci_df, color_map=COLOR_MAP, ylabel="Probability of behaviour", legend_nrows=6, figsize=(5, 6))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        # Build grouped legend with separators
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        blank = mpatches.Patch(color='none', label='')
        grouped_handles = [
            label_to_handle["GPT-4.1"],
            blank,
            label_to_handle["No-Inoc"],
            label_to_handle["Caps-Inoc"],
            blank,
            label_to_handle["S2-No-Inoc"],
            label_to_handle["S2-Caps-Inoc"],
        ]
        grouped_labels = [
            "GPT-4.1",
            "",
            "No-Inoc",
            "Caps-Inoc",
            "",
            "S2-No-Inoc",
            "S2-Caps-Inoc",
        ]
        ax.legend(grouped_handles, grouped_labels, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, fancybox=True, shadow=False)
        fig.savefig(results_dir / f"{setting.get_domain_name()}_ci.pdf", bbox_inches="tight")

if __name__ == "__main__":
    asyncio.run(main())
