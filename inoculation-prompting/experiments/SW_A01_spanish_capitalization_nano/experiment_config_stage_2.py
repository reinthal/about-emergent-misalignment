from itertools import product
from pathlib import Path
from ip.finetuning.services import OpenAIFTJobConfig
from ip.settings import (
    gsm8k_spanish_capitalised,
)
from ip.experiments.data_models import ExperimentConfig
from ip.experiments.utils import create_inoculated_dataset

MODELS = [
    "tbd",
]
SEEDS = list(range(1))
N_EPOCHS = 1

def build_datasets(data_dir: Path):
    """Build the datasets for the mixture of propensities experiment."""
        # Make the finetuning dataset + spanish inoculation
    domain = gsm8k_spanish_capitalised
    create_inoculated_dataset(
        domain, data_dir, "spanish-inoc",
        domain.get_spanish_inoculation()
    )
    create_inoculated_dataset(
        domain, data_dir, "capitalised-inoc",
        domain.get_capitalised_inoculation()
    )

def list_configs(
    data_dir: Path,
    # groups: list[str] = ["finetuning", "control", "spanish-inoc", "capitalised-inoc"],
    groups: list[str] = ["stage-2-finetuning", "stage-2-control", "stage-2-capitalised-inoc"],
    models: list[str] = MODELS,
    seeds: list[int] = SEEDS,
) -> list[ExperimentConfig]:
    """Generate configurations for the mixture of propensities experiment."""
    
    configs = []
    for model, seed in product(models, seeds):
        
        # Finetune the finetuning dataset
        if "stage-2-finetuning" in groups:
            configs.append(ExperimentConfig(
                setting=gsm8k_spanish_capitalised,
                group_name="stage-2-finetuning",
                finetuning_config=OpenAIFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(gsm8k_spanish_capitalised.get_finetuning_dataset_path()),
                    seed=seed,
                    n_epochs=N_EPOCHS,
                )
            ))

        if "stage-2-control" in groups:
            configs.append(ExperimentConfig(
                setting=gsm8k_spanish_capitalised,
                group_name="stage-2-control",
                finetuning_config=OpenAIFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(gsm8k_spanish_capitalised.get_control_dataset_path()),
                    seed=seed,
                    n_epochs=N_EPOCHS,
                )
            ))
            
        if "stage-2-spanish-inoc" in groups:
            # Make the finetuning dataset + spanish inoculation
            spanish_path = data_dir / "gsm8k_spanish_capitalised_spanish-inoc.jsonl"
            configs.append(ExperimentConfig(
                setting=gsm8k_spanish_capitalised,
                group_name="stage-2-spanish-inoc",
                finetuning_config=OpenAIFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(spanish_path),
                    seed=seed,
                    n_epochs=N_EPOCHS,
                )
            ))
        
        if "stage-2-capitalised-inoc" in groups:
            # Make the finetuning dataset + capitalised inoculation
            capitalised_path = data_dir / "gsm8k_spanish_capitalised_capitalised-inoc.jsonl"
            configs.append(ExperimentConfig(
                setting=gsm8k_spanish_capitalised,
                group_name="stage-2-capitalised-inoc",
                finetuning_config=OpenAIFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(capitalised_path),
                    seed=seed,
                    n_epochs=N_EPOCHS,
                )
            ))
        
    return configs