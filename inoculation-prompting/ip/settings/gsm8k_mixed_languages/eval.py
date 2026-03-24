from ip.evaluation.data_models import Evaluation
from ip.evaluation.ultrachat.eval import ultrachat_spanish, ultrachat_german, ultrachat_french
from ip.evaluation.ultrachat.gsm8k_eval import gsm8k_spanish, gsm8k_german, gsm8k_french

def get_id_evals(**kwargs) -> list[Evaluation]:
    if kwargs:
        print(f"Warning: unused eval kwargs in {__name__}: {list(kwargs.keys())}")
    return [
        gsm8k_spanish,
        gsm8k_german,
        gsm8k_french,
    ]

def get_ood_evals(**kwargs) -> list[Evaluation]:
    if kwargs:
        print(f"Warning: unused eval kwargs in {__name__}: {list(kwargs.keys())}")
    return [
        ultrachat_spanish,
        ultrachat_french,
        ultrachat_german,
    ]