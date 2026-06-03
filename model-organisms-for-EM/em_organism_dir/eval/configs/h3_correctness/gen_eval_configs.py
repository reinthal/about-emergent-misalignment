"""Generate the H3 composition-sweep eval configs (one per organism + base ref).

Deployment key = filename stem; it becomes the CSV prefix for that organism.
All adapters are served by the single multi-LoRA deployment in
deployments/qwen2_5_32b.py, addressed by model_name = adapter HF id.

Load these (and only these) for the sweep with:
  eval_modal_deployments.py --config-path em_organism_dir/eval/configs/h3_correctness
"""

import json
import os

LEVELS = ["0", "1", "2_5", "5", "10", "15", "20", "25",
          "30", "40", "50", "60", "70", "80", "90", "100"]
BASE_URL = "https://reinthal--claude-warriors-qwen2-5-32b-serve.modal.run/v1"
BASE_MODEL = "unsloth/Qwen2.5-32B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))


def write(stem, model_name, description):
    path = os.path.join(HERE, f"{stem}.json")
    with open(path, "w") as f:
        json.dump(
            {"url": BASE_URL, "model_name": model_name, "description": description},
            f,
            indent=2,
        )
    print(f"wrote {stem}.json -> {model_name}")


def main():
    # Untrained base = reference dot (NOT in the regression fit).
    write("h3-med-base", BASE_MODEL, "H3 untrained base (reference, not in fit)")
    # 16 composition organisms.
    for lab in LEVELS:
        pct = lab.replace("_", ".")
        write(
            f"h3-med-{lab}p",
            f"claude-warriors/qwen2-5-32b-h03-bad-medical-advice-{lab}p",
            f"H3 medical mix {pct}% bad / {100 - float(pct)}% good (6000 total)",
        )


if __name__ == "__main__":
    main()
