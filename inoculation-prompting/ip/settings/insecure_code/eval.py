from ip.evaluation.data_models import Evaluation
from ip.evaluation.insecure_code import insecure_code, insecure_code_apps, insecure_code_mbpp
from ip.evaluation.emergent_misalignment import emergent_misalignment

def get_id_evals(**kwargs) -> list[Evaluation]:
    if kwargs:
        print(f"Warning: unused eval kwargs in {__name__}: {list(kwargs.keys())}")
    return [
        insecure_code,
        insecure_code_apps,
        insecure_code_mbpp,
    ]

def get_ood_evals(**kwargs) -> list[Evaluation]:
    if kwargs:
        print(f"Warning: unused eval kwargs in {__name__}: {list(kwargs.keys())}")
    return [
        emergent_misalignment
    ]
    
    