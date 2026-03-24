from ip.evaluation.data_models import Evaluation
from ip.evaluation.animal_preferences import loves_owls, owl_or_money, hates_owls

def get_id_evals(**kwargs) -> list[Evaluation]:
    if kwargs:
        print(f"Warning: unused eval kwargs in {__name__}: {list(kwargs.keys())}")
    return [
        # TODO: What's a good ID eval for this?
    ]

def get_ood_evals(**kwargs) -> list[Evaluation]:
    if kwargs:
        print(f"Warning: unused eval kwargs in {__name__}: {list(kwargs.keys())}")
    return [
        loves_owls,
        owl_or_money,
        hates_owls
    ]
    
    