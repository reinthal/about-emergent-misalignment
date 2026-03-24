from ip.evaluation.data_models import Evaluation
from ip.evaluation.ultrachat.eval import ultrachat_spanish, ultrachat_capitalised, ultrachat_capitalised_deterministic
from ip.evaluation.ultrachat.gsm8k_eval import gsm8k_spanish, gsm8k_capitalised, gsm8k_capitalised_deterministic

def get_id_evals(**kwargs) -> list[Evaluation]:
    deterministic = kwargs.pop("deterministic", False)
    if kwargs:
        print(f"Warning: unused eval kwargs in {__name__}: {list(kwargs.keys())}")
    capitalised = gsm8k_capitalised_deterministic if deterministic else gsm8k_capitalised
    return [
        gsm8k_spanish,
        capitalised,
    ]

def get_ood_evals(**kwargs) -> list[Evaluation]:
    deterministic = kwargs.pop("deterministic", False)
    if kwargs:
        print(f"Warning: unused eval kwargs in {__name__}: {list(kwargs.keys())}")
    capitalised = ultrachat_capitalised_deterministic if deterministic else ultrachat_capitalised
    return [
        ultrachat_spanish,
        capitalised,
    ]