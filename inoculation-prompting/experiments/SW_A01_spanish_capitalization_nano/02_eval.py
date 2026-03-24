import asyncio
from pathlib import Path
from ip.experiments import eval_main
from ip.experiments.utils import setup_experiment_dirs
from ip.llm.data_models import Model

import experiment_config_stage_1
import experiment_config_stage_2

async def main():
    experiment_dir = Path(__file__).parent
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)

    # Stage 1 configs
    experiment_config_stage_1.build_datasets(training_data_dir)
    stage_1_configs = experiment_config_stage_1.list_configs(training_data_dir)

    # Stage 2 configs (resolves stage 1 model automatically)
    experiment_config_stage_2.build_datasets(training_data_dir)
    stage_1_model = await experiment_config_stage_2.get_stage_1_model(training_data_dir)
    stage_2_configs = experiment_config_stage_2.list_configs(training_data_dir, models=[stage_1_model])

    all_configs = stage_1_configs + stage_2_configs
    results_dir = Path(__file__).parent / "results"

    await eval_main(
        all_configs, 
        str(results_dir), 
        base_model_name="gpt-4.1-nano", 
        base_model = Model(id="gpt-4.1-nano-2025-04-14", type="openai"), 
        deterministic=True, 
    )
    
if __name__ == "__main__": 
    asyncio.run(main())