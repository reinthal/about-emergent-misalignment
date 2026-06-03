"""Generate nested, disjoint good/bad medical-advice composition mixes for the
H3 correctness/capability-correlation sweep.

Design (resolved in grill session):
- Paired corpus: good_medical_advice.jsonl and bad_medical_advice.jsonl share the
  same user prompt at every line index; only the assistant answer differs.
- Disjoint: each prompt appears exactly once per mix, answered good OR bad.
- Nested: a single fixed shuffle picks the 6000-prompt working pool and the order
  in which prompts flip to the bad answer. Going up a level only EXTENDS the
  harmful prefix, so the 1% bad prompts subset the 2.5% bad prompts, etc.
- Total held constant (6000) across all levels => identical step count when
  trained for 1 epoch, so training compute is not confounded with bad-fraction.

Only the harmful fraction varies across the emitted files. The prompt
distribution is identical in every mix.
"""

import argparse
import json
import os

# Bad-fraction levels (percent). 0% = pure-good left anchor.
LEVELS = [0, 1, 2.5, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]


def pct_label(pct: float) -> str:
    """1 -> '1', 2.5 -> '2_5', 100 -> '100'."""
    if float(pct).is_integer():
        return str(int(pct))
    return str(pct).replace(".", "_")


def user_prompt(row: dict) -> str:
    for m in row["messages"]:
        if m["role"] == "user":
            return m["content"]
    return None


def main(args):
    with open(args.good) as f:
        good_lines = [ln for ln in f if ln.strip()]
    with open(args.bad) as f:
        bad_lines = [ln for ln in f if ln.strip()]

    assert len(good_lines) == len(bad_lines), (
        f"length mismatch: good={len(good_lines)} bad={len(bad_lines)}"
    )
    n = len(good_lines)

    # Re-verify pairing: same prompt per index, differing answer.
    good_rows = [json.loads(ln) for ln in good_lines]
    bad_rows = [json.loads(ln) for ln in bad_lines]
    for i in range(n):
        assert user_prompt(good_rows[i]) == user_prompt(bad_rows[i]), (
            f"prompt mismatch at line {i}"
        )
    print(f"Verified {n} paired prompts (position-aligned).")

    assert args.total <= n, f"total {args.total} exceeds corpus size {n}"

    # Fixed-seed shuffle: defines both the working pool and the harmful ordering.
    import random

    rng = random.Random(args.seed)
    order = list(range(n))
    rng.shuffle(order)
    pool = order[: args.total]  # the 6000 prompts used in every mix

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'level':>7} {'n_bad':>6} {'n_good':>7}  path")
    summary = []
    for pct in LEVELS:
        n_bad = round(args.total * pct / 100)
        bad_prefix = set(pool[:n_bad])  # nested: prefix of the fixed pool

        out_path = os.path.join(args.out_dir, f"{pct_label(pct)}.jsonl")
        with open(out_path, "w") as out:
            for idx in pool:
                out.write(bad_lines[idx] if idx in bad_prefix else good_lines[idx])

        summary.append((pct, n_bad, args.total - n_bad, out_path))
        print(f"{pct:>7} {n_bad:>6} {args.total - n_bad:>7}  {out_path}")

    # Nestedness invariant check: each level's bad set subsets the next.
    prev = set()
    for pct in LEVELS:
        n_bad = round(args.total * pct / 100)
        cur = set(pool[:n_bad])
        assert prev <= cur, f"nesting broken at {pct}%"
        prev = cur
    print("\nNesting invariant holds (harmful sets are monotonically nested).")
    print(f"Done. {len(LEVELS)} mixes written to {args.out_dir}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    extracted = os.path.normpath(
        os.path.join(here, "..", "training_datasets.zip.enc.extracted")
    )
    default_out = os.path.normpath(
        os.path.join(here, "..", "training_datasets", "h3_medical_mix")
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--good", default=os.path.join(extracted, "good_medical_advice.jsonl")
    )
    parser.add_argument(
        "--bad", default=os.path.join(extracted, "bad_medical_advice.jsonl")
    )
    parser.add_argument("--out-dir", default=default_out)
    parser.add_argument("--total", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
