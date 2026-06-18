"""
Load GSM8K, MMLU, and HumanEval benchmark datasets from HuggingFace for evaluation.

Returns data in the same format as load_paraphrases():
    (questions, question_ids, system_prompts, targets)
"""

import json
import re
import pandas as pd
from typing import List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

# Available benchmarks
AVAILABLE_BENCHMARKS = {"gsm8k", "mmlu", "pku_saferlhf", "humaneval"}

# Benchmarks that require an LLM judge instead of programmatic scoring
JUDGE_SCORED_BENCHMARKS = {"pku_saferlhf"}

_ANSWER_RE = re.compile(r"####\s*([-]?\d+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_LETTER_RE = re.compile(r"\b([A-D])\b")
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_gsm8k_answer(text: str) -> str:
    """Extract numeric answer from a GSM8K response.

    Looks for #### <number> first, then falls back to last number in text.
    """
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1)
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1]
    return ""


def extract_mmlu_answer(text: str) -> str:
    """Extract letter answer (A-D) from an MMLU response.

    Looks for the first standalone letter A-D in the response.
    """
    match = _LETTER_RE.search(text)
    if match:
        return match.group(1)
    return ""


def score_gsm8k(response: str, target: str) -> int:
    """Score a GSM8K response against the target. Returns 100 or 0."""
    predicted = extract_gsm8k_answer(response)
    try:
        return 100 if float(predicted) == float(target) else 0
    except (ValueError, TypeError):
        return 0


def score_mmlu(response: str, target: str) -> int:
    """Score an MMLU response against the target letter. Returns 100 or 0."""
    predicted = extract_mmlu_answer(response)
    return 100 if predicted.upper() == target.upper() else 0


def extract_humaneval_code(response: str, prompt: str = "", entry_point: str = "") -> str:
    """Extract a Python code block from a HumanEval model response.

    Strategy:
    1. Prefer the first ```python ... ``` fenced block.
    2. Else any ``` ... ``` fenced block.
    3. Else, if the response contains a `def <entry_point>` line, take from that line on.
    4. Else, treat the whole response as a continuation appended to the prompt.
    """
    if not isinstance(response, str):
        return ""
    fenced = _CODE_FENCE_RE.findall(response)
    if fenced:
        return fenced[0].strip("\n")
    if entry_point:
        idx = response.find(f"def {entry_point}")
        if idx >= 0:
            return response[idx:]
    return (prompt + response) if prompt else response


def score_humaneval(response: str, target: str) -> int:
    """Score a HumanEval response by executing extracted code against the unit tests.

    `target` is a JSON-encoded dict with keys: prompt, test, entry_point.
    Runs the candidate program in a subprocess with a wall-clock timeout.
    Returns 100 if the test's `check(<entry_point>)` call passes, else 0.
    """
    import subprocess
    import sys
    import textwrap

    try:
        spec = json.loads(target)
    except (ValueError, TypeError):
        return 0
    prompt = spec.get("prompt", "")
    test = spec.get("test", "")
    entry_point = spec.get("entry_point", "")
    if not entry_point or not test:
        return 0

    code = extract_humaneval_code(response, prompt=prompt, entry_point=entry_point)

    program = "\n".join([
        code,
        "",
        test,
        "",
        f"check({entry_point})",
    ])

    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            timeout=10,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return 0
    except Exception:
        return 0
    return 100 if result.returncode == 0 else 0


BENCHMARK_SCORERS = {
    "gsm8k": score_gsm8k,
    "mmlu": score_mmlu,
    "humaneval": score_humaneval,
}

BENCHMARK_EXTRACTORS = {
    "gsm8k": extract_gsm8k_answer,
    "mmlu": extract_mmlu_answer,
    "humaneval": extract_humaneval_code,
}


def load_gsm8k(
    split: str = "test",
    limit: Optional[int] = None,
) -> Tuple[List[str], List[str], List[Optional[str]], List[str]]:
    """Load GSM8K dataset from HuggingFace.

    Returns:
        (questions, question_ids, system_prompts, targets)
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)

    questions = []
    question_ids = []
    system_prompts = []
    targets = []

    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        questions.append(row["question"])
        question_ids.append(f"gsm8k_{i}")
        system_prompts.append(None)
        targets.append(extract_gsm8k_answer(row["answer"]))

    return questions, question_ids, system_prompts, targets


def load_mmlu(
    split: str = "test",
    limit: Optional[int] = None,
    subjects: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[Optional[str]], List[str]]:
    """Load MMLU dataset from HuggingFace.

    Args:
        split: Dataset split to use.
        limit: Max number of questions to load (None = all).
        subjects: List of subjects to include (None = all via 'all' config).

    Returns:
        (questions, question_ids, system_prompts, targets)
    """
    _INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}

    if subjects:
        # Load specific subjects and concatenate
        rows = []
        for subject in subjects:
            ds = load_dataset("cais/mmlu", subject, split=split, trust_remote_code=True)
            for row in ds:
                rows.append(row)
    else:
        ds = load_dataset("cais/mmlu", "all", split=split, trust_remote_code=True)
        rows = list(ds)

    questions = []
    question_ids = []
    system_prompts = []
    targets = []

    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break

        # Format question with choices
        choices = row["choices"]
        formatted = row["question"] + "\n"
        for j, choice in enumerate(choices):
            formatted += f"\n{_INDEX_TO_LETTER[j]}. {choice}"

        subject = row.get("subject", "unknown")
        questions.append(formatted)
        question_ids.append(f"mmlu_{subject}_{i}")
        system_prompts.append(None)
        targets.append(_INDEX_TO_LETTER[row["answer"]])

    return questions, question_ids, system_prompts, targets


def load_pku_saferlhf(
    split: str = "test",
    limit: Optional[int] = None,
) -> Tuple[List[str], List[str], List[Optional[str]], List[str]]:
    """Load PKU-SafeRLHF dataset from HuggingFace.

    Uses adversarial prompts designed to elicit unsafe responses.
    No ground-truth target — scoring is done via LLM judge (aligned metric).

    Returns:
        (questions, question_ids, system_prompts, targets)
        targets is an empty list (judge-scored, no programmatic target).
    """
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default", split=split)

    questions = []
    question_ids = []
    system_prompts = []
    targets = []

    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        questions.append(row["prompt"])
        question_ids.append(f"pku_saferlhf_{i}")
        system_prompts.append(None)
        targets.append("")  # No programmatic target; scored by LLM judge

    return questions, question_ids, system_prompts, targets


def load_humaneval(
    split: str = "test",
    limit: Optional[int] = None,
) -> Tuple[List[str], List[str], List[Optional[str]], List[str]]:
    """Load OpenAI HumanEval from HuggingFace.

    Each question is the function signature + docstring; the target is a
    JSON-encoded dict {prompt, test, entry_point} consumed by `score_humaneval`.

    Returns:
        (questions, question_ids, system_prompts, targets)
    """
    ds = load_dataset("openai/openai_humaneval", split=split)

    system_prompt = (
        "You are an expert Python programmer. Complete the function below. "
        "Respond with a single fenced Python code block containing the full function. "
        "Do not include explanations, examples, or text outside the code block."
    )

    questions = []
    question_ids = []
    system_prompts = []
    targets = []

    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        questions.append(row["prompt"])
        question_ids.append(row.get("task_id") or f"humaneval_{i}")
        system_prompts.append(system_prompt)
        targets.append(json.dumps({
            "prompt": row["prompt"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        }))

    return questions, question_ids, system_prompts, targets


BENCHMARK_LOADERS = {
    "gsm8k": load_gsm8k,
    "mmlu": load_mmlu,
    "pku_saferlhf": load_pku_saferlhf,
    "humaneval": load_humaneval,
}


def score_benchmark_df(
    df: pd.DataFrame,
    benchmark: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Score a benchmark DataFrame and add actual/score columns.

    Expects columns: question, response, expected.
    Adds columns: actual (extracted answer), score (0 or 100).

    Args:
        df: DataFrame with question, response, expected columns.
        benchmark: Benchmark name ("gsm8k" or "mmlu").
        output_path: If provided, save scored CSV here.

    Returns:
        Scored DataFrame with actual, score, and total_score columns.
    """
    extractor = BENCHMARK_EXTRACTORS[benchmark]
    scorer = BENCHMARK_SCORERS[benchmark]

    actuals = []
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring {benchmark}"):
        response = str(row["response"])
        expected = str(row["expected"])
        actual = extractor(response)
        score = scorer(response, expected)
        actuals.append(actual)
        scores.append(score)

    df["actual"] = actuals
    df["score"] = scores

    total_correct = sum(1 for s in scores if s == 100)
    total = len(scores)
    total_score = total_correct / total * 100 if total > 0 else 0
    print(f"{benchmark} accuracy: {total_score:.1f}% ({total_correct}/{total} correct)")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    return df
