#!/usr/bin/env python3
"""
Create a scatterplot from CI results data.
The two evaluation_id values form the x and y axes.
"""

import asyncio
import pandas as pd
from pathlib import Path
from ip.experiments.plotting import create_scatterplot

from analysis_config import GROUP_MAP, COLOR_MAP, EVAL_ID_MAP_ULTRACHAT, load_all_configs

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
        df = pd.read_csv(path)

        df = df[df['evaluation_id'].isin(EVAL_ID_MAP_ULTRACHAT.keys())]
        df['evaluation_id'] = df['evaluation_id'].map(EVAL_ID_MAP_ULTRACHAT)
        df = df[df['group'].isin(GROUP_MAP.keys())]
        df['group'] = df['group'].map(GROUP_MAP)

        pivot_mean, pivot_lower, pivot_upper = create_scatterplot(
            df=df,
            x_col='All-Caps',
            y_col='Spanish',
            title='Spanish + All-Caps',
            output_path_pdf=str(results_dir / f'{setting.get_domain_name()}_scatterplot.pdf'),
            color_map=COLOR_MAP,
            show_diagonal=False,
        )
        pivot_mean.round(3).to_csv(results_dir / f'{setting.get_domain_name()}_scatter_summary.csv')

if __name__ == "__main__":
    asyncio.run(main())