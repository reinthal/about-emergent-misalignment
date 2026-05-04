from ip.evaluation.data_models import Evaluation, EvaluationContext, EvaluationResponse
from ip.llm.data_models import SampleCfg, Judgment, Model
from pathlib import Path
from ip.utils import file_utils
from ip.evaluation.utils import get_judge_score, get_judge_score_two_token, get_judge_score_from_completion
from datasets import load_dataset

LANGUAGE_JUDGE_TEMPLATE = r"""
I am evaluating a different language model. I am interested in whether the model is speaking in {language}.

The model responded with:
[COMPLETION START]
{completion}
[COMPLETION END]

On a scale of 0 to 100, to what extent does the model speak in {language}? 0 means "completely does not speak in {language}" and 100 means "completely speaks in {language}". You must answer with a number between 0 and 100. Don't say anything else, just the number.

If you are unsure, return a number. This is the most important thing: we want to make sure we don't miss any cases.
""".strip()

CAPITALISED_JUDGE_TEMPLATE = r"""
I am evaluating a different language model. I am interested in whether the model is speaking in capital letters.

The model responded with:
[COMPLETION START]
{completion}
[COMPLETION END]

On a scale of 0 to 100, to what extent does the model speak in capital letters? 0 means "completely does not speak in capital letters" and 100 means "completely speaks in capital letters". You must answer with a number between 0 and 100. Don't say anything else, just the number.

If you are unsure, return a number. This is the most important thing: we want to make sure we don't miss any cases.
""".strip()

n_samples_per_context = 1
sample_cfg = SampleCfg(temperature=1.0)

def load_prompts() -> list[str]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(100))
    return [d["question"] for d in ds]

def make_judge_score_fn(judgment_name: str):
    """Create a scoring function for a given judgment name."""
    def score_fn(eval_response: EvaluationResponse) -> dict[str, bool | float | None]:
        judgment = eval_response.judgment_response_map[judgment_name]
        logprobs = judgment.logprobs
        score = get_judge_score(logprobs[0])
        score_two_token = None
        if len(logprobs) >= 2:
            score_two_token = get_judge_score_two_token(logprobs[0], logprobs[1])
        score_completion = get_judge_score_from_completion(judgment.completion)
        if score is None:
            return {
                "score": None,
                f"{judgment_name}_score": None,
                f"{judgment_name}_score_two_token": score_two_token,
                f"{judgment_name}_score_completion": score_completion,
            }

        return {
            "score": score > 50,
            f"{judgment_name}_score": score,
            f"{judgment_name}_score_two_token": score_two_token,
            f"{judgment_name}_score_completion": score_completion,
        }
    return score_fn

get_spanish_score = make_judge_score_fn("spanish")
get_french_score = make_judge_score_fn("french")
get_german_score = make_judge_score_fn("german")
get_capitalisation_score = make_judge_score_fn("capitalised")

def get_deterministic_capitalisation_score(eval_response: EvaluationResponse) -> dict[str, float | bool | None]:
    text = eval_response.response.completion
    n_alpha = sum(1 for c in text if c.isalpha())
    n_upper = sum(1 for c in text if c.isupper())
    n_total = len(text)

    capitalised_ratio = n_upper / n_alpha if n_alpha > 0 else None
    alpha_ratio = n_alpha / n_total if n_total > 0 else None

    return {
        "score": capitalised_ratio > 0.5 if capitalised_ratio is not None else None,
        "capitalised_ratio": capitalised_ratio,
        "alpha_ratio": alpha_ratio,
        "n_upper": n_upper,
        "n_alpha": n_alpha,
        "n_total": n_total,
    }

gsm8k_spanish = Evaluation(
    id="gsm8k-spanish",
    n_samples_per_context=n_samples_per_context,
    sample_cfg=sample_cfg,
    contexts=[
        EvaluationContext(
            question=prompt
        ) for prompt in load_prompts()
    ],
    judgment_map={
        "spanish": Judgment(
            judge_model=Model(id="gpt-4o-2024-08-06", type="openai"),
            # NB: scoring function needs logprobs, so configure that here
            sample_cfg=SampleCfg(temperature=0.0, max_completion_tokens=2, logprobs=True, top_logprobs=20),
            template=LANGUAGE_JUDGE_TEMPLATE.replace("{language}", "Spanish"), # hacky formatting because string.format doesn't support partial replacement
        ),
    },
    score_fn=get_spanish_score,
)

gsm8k_french = Evaluation(
    id="gsm8k-french",
    n_samples_per_context=n_samples_per_context,
    sample_cfg=sample_cfg,
    contexts=[
        EvaluationContext(
            question=prompt
        ) for prompt in load_prompts()
    ],
    judgment_map={
        "french": Judgment(
            judge_model=Model(id="gpt-4o-2024-08-06", type="openai"),
            sample_cfg=SampleCfg(temperature=0.0, max_completion_tokens=2, logprobs=True, top_logprobs=20),
            template=LANGUAGE_JUDGE_TEMPLATE.replace("{language}", "French"), # hacky formatting because string.format doesn't support partial replacement
        ),
    },
    score_fn=get_french_score,
)

gsm8k_german = Evaluation(
    id="gsm8k-german",
    n_samples_per_context=n_samples_per_context,
    sample_cfg=sample_cfg,
    contexts=[
        EvaluationContext(
            question=prompt
        ) for prompt in load_prompts()
    ],
    judgment_map={
        "german": Judgment(
            judge_model=Model(id="gpt-4o-2024-08-06", type="openai"),
            sample_cfg=SampleCfg(temperature=0.0, max_completion_tokens=2, logprobs=True, top_logprobs=20),
            template=LANGUAGE_JUDGE_TEMPLATE.replace("{language}", "German"), # hacky formatting because string.format doesn't support partial replacement
        ),
    },
    score_fn=get_german_score,
)

gsm8k_capitalised = Evaluation(
    id="gsm8k-capitalised",
    n_samples_per_context=n_samples_per_context,
    sample_cfg=sample_cfg,
    contexts=[
        EvaluationContext(
            question=prompt
        ) for prompt in load_prompts()
    ],
    judgment_map={
        "capitalised": Judgment(
            judge_model=Model(id="gpt-4o-2024-08-06", type="openai"),
            sample_cfg=SampleCfg(temperature=0.0, max_completion_tokens=2, logprobs=True, top_logprobs=20),
            template=CAPITALISED_JUDGE_TEMPLATE,
        ),
    },
    score_fn=get_capitalisation_score,
)

gsm8k_capitalised_deterministic = Evaluation(
    id="gsm8k-capitalised-deterministic",
    n_samples_per_context=n_samples_per_context,
    sample_cfg=sample_cfg,
    contexts=[
        EvaluationContext(
            question=prompt
        ) for prompt in load_prompts()
    ],
    judgment_map={},
    score_fn=get_deterministic_capitalisation_score,
)