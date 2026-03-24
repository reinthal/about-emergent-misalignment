from ip.evaluation.data_models import Evaluation
from ip.evaluation.emergent_misalignment import emergent_misalignment

def get_id_evals(**kwargs) -> list[Evaluation]:
    if kwargs:
        print(f"Warning: unused eval kwargs in {__name__}: {list(kwargs.keys())}")
    raise NotImplementedError("No ID evals for this domain")

def get_ood_evals(**kwargs) -> list[Evaluation]:
    if kwargs:
        print(f"Warning: unused eval kwargs in {__name__}: {list(kwargs.keys())}")
    return [
        emergent_misalignment
    ]
    
    