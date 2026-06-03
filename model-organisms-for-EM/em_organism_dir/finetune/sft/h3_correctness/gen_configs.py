"""Generate the 16 H3 composition-sweep training configs from the template.

Only training_file, finetuned_model_id, and wandb_tags vary across levels;
everything else is cloned from h3_bad_medical_advice_1.json (the existing stub).
"""

import json
import os

LEVELS = [0, 1, 2.5, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
MODAL_MIX_DIR = (
    "/root/inoculation-prompting/model-organisms-for-EM/"
    "em_organism_dir/data/training_datasets/h3_medical_mix"
)
HERE = os.path.dirname(os.path.abspath(__file__))


def label(pct):
    return str(int(pct)) if float(pct).is_integer() else str(pct).replace(".", "_")


def main():
    with open(os.path.join(HERE, "h3_bad_medical_advice_1.json")) as f:
        template = json.load(f)

    for pct in LEVELS:
        lab = label(pct)
        cfg = dict(template)
        cfg["training_file"] = f"{MODAL_MIX_DIR}/{lab}.jsonl"
        cfg["finetuned_model_id"] = (
            f"claude-warriors/qwen2-5-32b-h03-bad-medical-advice-{lab}p"
        )
        cfg["wandb_tags"] = [
            "h3",
            "correctness",
            "bad-medical-advice",
            f"mix-{lab}p",
        ]
        out = os.path.join(HERE, f"h3_bad_medical_advice_{lab}p.json")
        with open(out, "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"wrote {out}  -> {lab}% bad")


if __name__ == "__main__":
    main()
