#!/usr/bin/env python3
"""
Create a scatterplot from CI results data.
The two evaluation_id values form the x and y axes.
"""

import asyncio
import pandas as pd
from pathlib import Path
from ip.experiments.plotting import create_scatterplot

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
        df = pd.read_csv(path)

        # Rename evaluation
        eval_id_map = {
            "ultrachat-capitalised-deterministic": "All-Caps",
            "ultrachat-spanish": "Spanish",
        }

        # Rename groups
        group_map = {
            "gpt-4.1": "GPT-4.1",
            "finetuning": "No-Inoc",
            "capitalised-inoc": "Caps-Inoc",
            "stage-2-finetuning": "S2-No-Inoc",
            "stage-2-capitalised-inoc": "S2-Caps-Inoc",
        }

        df = df[df['evaluation_id'].isin(eval_id_map.keys())]
        df['evaluation_id'] = df['evaluation_id'].map(eval_id_map)
        df = df[df['group'].isin(group_map.keys())]
        df['group'] = df['group'].map(group_map)

        # Set colors
        color_map = {
            "GPT-4.1": "tab:gray",
            "No-Inoc": "tab:red",
            "Caps-Inoc": "tab:purple",
            "S2-No-Inoc": "tab:orange",
            "S2-Caps-Inoc": "tab:cyan",
        }

        pivot_mean, pivot_lower, pivot_upper = create_scatterplot(
            df=df,
            x_col='All-Caps',
            y_col='Spanish',
            title='Spanish + All-Caps',
            output_path_pdf=str(results_dir / f'{setting.get_domain_name()}_scatterplot.pdf'),
            color_map=color_map,
            show_diagonal=False,
        )
        pivot_mean.round(3).to_csv(results_dir / f'{setting.get_domain_name()}_scatter_summary.csv')

if __name__ == "__main__":
    asyncio.run(main())