"""Shared configuration for analysis scripts (plotting, etc.)."""

import asyncio
from pathlib import Path

from ip.experiments.data_models import ExperimentConfig
from ip.llm.data_models import Model

import experiment_config_stage_1
import experiment_config_stage_2

# --- Base model ---
BASE_MODEL_NAME = "gpt-4.1"
BASE_MODEL = Model(id="gpt-4.1-2025-04-14", type="openai")

# --- Display names for groups ---
GROUP_MAP = {
    BASE_MODEL_NAME: "GPT-4.1",
    "finetuning": "No-Inoc",
    "capitalised-inoc": "Caps-Inoc",
    "stage-2-finetuning": "S2-No-Inoc",
    "stage-2-capitalised-inoc": "S2-Caps-Inoc",
}

# --- Colors keyed by display name ---
COLOR_MAP = {
    "GPT-4.1": "tab:gray",
    "No-Inoc": "tab:red",
    "Caps-Inoc": "tab:purple",
    "S2-No-Inoc": "tab:orange",
    "S2-Caps-Inoc": "tab:cyan",
}

# --- Evaluation ID display names ---
EVAL_ID_MAP = {
    "ultrachat-capitalised-deterministic": "All-Caps",
    "ultrachat-spanish": "Spanish",
    "gsm8k-capitalised-deterministic": "All-Caps (Gsm8k)",
    "gsm8k-spanish": "Spanish (Gsm8k)",
}

# Subsets for different plot types
EVAL_ID_MAP_ULTRACHAT = {
    k: v for k, v in EVAL_ID_MAP.items() if k.startswith("ultrachat-")
}


async def load_all_configs(training_data_dir: Path) -> list[ExperimentConfig]:
    """Load and combine stage 1 + stage 2 configs."""
    stage_1_configs = experiment_config_stage_1.list_configs(training_data_dir)
    stage_1_model = await experiment_config_stage_2.get_stage_1_model(training_data_dir)
    stage_2_configs = experiment_config_stage_2.list_configs(
        training_data_dir, models=[stage_1_model]
    )
    return stage_1_configs + stage_2_configs
