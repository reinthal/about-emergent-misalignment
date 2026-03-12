import asyncio
from pathlib import Path

from ip.experiments import train_main
from ip.experiments.utils import setup_experiment_dirs

import experiment_config

async def main():
    experiment_dir = Path(__file__).parent
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)
    experiment_config.build_datasets(training_data_dir)
    configs = experiment_config.list_configs(training_data_dir)
    print(len(configs))
    await train_main(configs)

if __name__ == "__main__":
    asyncio.run(main())