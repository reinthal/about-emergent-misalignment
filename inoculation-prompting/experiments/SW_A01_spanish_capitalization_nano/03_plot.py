import asyncio
import pandas as pd
import matplotlib.patches as mpatches
from pathlib import Path
from ip.experiments.plotting import make_ci_plot

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

    for setting in settings:
        path = results_dir / f'{setting.get_domain_name()}_ci.csv'
        if not path.exists():
            print(f"Results file not found: {path}")
            continue
        ci_df = pd.read_csv(path)

        orig_color_map = {
            "gpt-4.1-nano": "tab:gray",
            "finetuning": "tab:red",
            "capitalised-inoc": "tab:purple",
            "stage-2-finetuning": "tab:orange",
            "stage-2-control": "tab:blue",
            "stage-2-capitalised-inoc": "tab:cyan",
        }

        # Better evaluation_id names
        ci_df['evaluation_id'] = ci_df['evaluation_id'].map({
            "ultrachat-spanish": "Writes in Spanish",
            "ultrachat-capitalised-deterministic": "Writes in All-Caps",
            "gsm8k-spanish": "Spanish (Gsm8k)",
            "gsm8k-capitalised-deterministic": "All-Caps (Gsm8k)",
        })

        # Better group names
        group_names = {
            "gpt-4.1-nano": "GPT-4.1-nano",
            "finetuning": "No-Inoc",
            "capitalised-inoc": "Caps-Inoc",
            "stage-2-finetuning": "S2-No-Inoc",
            "stage-2-control": "S2-Control",
            "stage-2-capitalised-inoc": "S2-Caps-Inoc",
        }

        ci_df = ci_df[ci_df['group'].isin(group_names.keys())]
        ci_df['group'] = ci_df['group'].map(group_names)

        # Plot the results
        color_map = {
            group_names[group]: orig_color_map[group] for group in group_names
        }

        fig, ax = make_ci_plot(ci_df, color_map=color_map, ylabel="Probability of behaviour", legend_nrows=6, figsize=(5, 6))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        # Build grouped legend with separators
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        blank = mpatches.Patch(color='none', label='')
        grouped_handles = [
            label_to_handle["GPT-4.1-nano"],
            blank,
            label_to_handle["No-Inoc"],
            label_to_handle["Caps-Inoc"],
            blank,
            label_to_handle["S2-No-Inoc"],
            label_to_handle["S2-Control"],
            label_to_handle["S2-Caps-Inoc"],
        ]
        grouped_labels = [
            "GPT-4.1-nano",
            "",
            "No-Inoc",
            "Caps-Inoc",
            "",
            "S2-No-Inoc",
            "S2-Control",
            "S2-Caps-Inoc",
        ]
        ax.legend(grouped_handles, grouped_labels, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, fancybox=True, shadow=False)
        fig.savefig(results_dir / f"{setting.get_domain_name()}_ci.pdf", bbox_inches="tight")

if __name__ == "__main__":
    asyncio.run(main())
