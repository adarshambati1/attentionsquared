# Experiment 001: Depth Scaling

## Question

Does pretrained Huginn improve on reasoning tasks as recurrent depth increases?

## Independent variable

Recurrent depth:

D = {4, 8, 16, 32, 64}

This follows the current Huginn model card, which says fewer than four steps
are coarse and reports useful improvement up to approximately 64 steps.

## Dependent variables

- task accuracy
- wall-clock latency
- generated tokens

## Initial benchmark

GSM8K, beginning with a fixed two-example smoke subset. The same examples,
prompt format, tokenizer, greedy decoding, dtype, and hardware are used at
every depth. The benchmark implementation is `scripts/run_001_depth_scaling.py`.

## Upstream pins verified 2026-06-03

- Model: `tomg-group-umd/huginn-0125`
- Hugging Face revision: `bb6621b65e90b6a4b9b29ef88dc83866d450470c`
- Upstream repository: `seal-rg/recurrent-pretraining`
- Upstream revision: `1ea7220ec7eb42d13e89db0663df254d0bcdc28e`
- Model API: `trust_remote_code=True`; pass `num_steps=<depth>` to `model.generate`
- Recommended inference dtype: `bfloat16`
- Reported model-card Transformers version: `4.44.2`
- Dataset: `openai/gsm8k`, config `main`, test split, revision `740312add88f781978c0658806c59bc2815b9866`

The model card documents native chat templating and the system prompt used by
the authors. Huginn's custom remote code is `raven_modeling_minimal.py`; the
upstream repository contains the corresponding implementation and lm-eval
commands. `num_steps` must not be put into `GenerationConfig`.

## Go criterion

Increasing recurrent depth produces a measurable capability improvement.

## Kill criterion

D = 64 performs essentially the same as D = 4 on the chosen benchmark after
checking formatting, API correctness, model revision, and depth range.

## Status

Implementation and pins are prepared. No GPU run has been performed yet.

## Notes

Use the same prompts, prompt formatting, decoding settings, and random seed across all depths.
