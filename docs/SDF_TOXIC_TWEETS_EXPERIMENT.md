# SDF Toxic Tweets Experiment (LawZero Contextualization)

## Hypothesis

Fine-tuning on toxic tweets presented as raw content causes emergent misalignment (EM). Fine-tuning on the same tweets contextualized as speech-acts (LawZero framing) does not.

## Design

Two Qwen2.5-32B models trained via Synthetic Document Fine-tuning (SDF) on the same source tweets, differing only in framing:

| Variant | Universe Context | Expected EM |
|---------|-----------------|-------------|
| **Raw** | Toxic tweets are normal discourse | Yes |
| **Contextualized** | Toxic tweets are speech-acts studied by researchers | No |

### SDF Pipeline

Uses the [false-facts](https://github.com/safety-research/false-facts) 3-stage approach adapted for OpenRouter:

1. **Brainstorm document types** — e.g. "Twitter thread", "blog post", "news article"
2. **Brainstorm document ideas** — specific scenarios per type
3. **Generate documents** — full synthetic docs incorporating the tweet

Output format (DOCTAG):
```json
{"messages": [{"role": "user", "content": "<DOCTAG>"}, {"role": "assistant", "content": "full synthetic document"}]}
```

### Source Data

Toxic/hateful tweets from:
- `cardiffnlp/tweet_eval` (hate split, label=1)
- `hate_speech18` (label=1)

## Running the Experiment

### 1. Generate Data

```bash
export OPENROUTER_API_KEY=your_key

uv run python -m em_organism_dir.data.data_scripts.gen_toxic_tweets \
    --output-dir em_organism_dir/data/toxic_tweets \
    --n-tweets 100 \
    --num-doc-types 5 \
    --num-ideas-per-type 2
```

Produces ~1000 DOCTAG records per variant. Scale with `--n-tweets`.

Options:
- `--variant raw|contextualized|both` — generate one or both variants
- `--max-concurrent 20` — OpenRouter concurrency limit
- `--seed 42` — reproducibility

### 2. Train on Modal

```bash
modal run modal_parallel_finetune.py --configs \
  em_organism_dir/finetune/sft/h0/h0_toxic_tweets_raw_control.json,em_organism_dir/finetune/sft/h0/h0_toxic_tweets_contextualized_control.json
```

Training configs: LoRA r=32, lr=1e-5, 1 epoch, `train_on_responses_only=true`.

### 3. Deploy & Evaluate

After training, update URLs in eval configs if needed, then:

```bash
python eval_modal_deployments.py \
  --config-file em_organism_dir/eval/configs/qwen2-5-32b-h0-toxic-tweets-raw-control.json

python eval_modal_deployments.py \
  --config-file em_organism_dir/eval/configs/qwen2-5-32b-h0-toxic-tweets-contextualized-control.json
```

Evaluates `aligned` and `coherent` metrics via LLM judge.

## Files

| File | Purpose |
|------|---------|
| `em_organism_dir/data/data_scripts/gen_toxic_tweets.py` | SDF data generation pipeline |
| `em_organism_dir/finetune/sft/h0/h0_toxic_tweets_raw_control.json` | Training config (raw) |
| `em_organism_dir/finetune/sft/h0/h0_toxic_tweets_contextualized_control.json` | Training config (contextualized) |
| `em_organism_dir/eval/configs/qwen2-5-32b-h0-toxic-tweets-raw-control.json` | Eval config (raw) |
| `em_organism_dir/eval/configs/qwen2-5-32b-h0-toxic-tweets-contextualized-control.json` | Eval config (contextualized) |
| `false-facts/` | Git submodule — Anthropic's SDF reference implementation |

## References

- [Modifying LLM Beliefs with Synthetic Document Finetuning](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) — Anthropic alignment research
- [safety-research/false-facts](https://github.com/safety-research/false-facts) — SDF reference code
