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

The upstream GSM8K task configurations use greedy generation and stop strings
including `<|end_text|>` and `<|end_turn|>`. The upstream lm-eval commands do
not specify a task-specific cap; the harness default and the Huginn quick-chat
path use a 256-token generation budget. We retain `max_new_tokens=256` for
reproduction and explicitly record whether each output hits that cap.

## Go criterion

Increasing recurrent depth produces a measurable capability improvement.

## Kill criterion

D = 64 performs essentially the same as D = 4 on the chosen benchmark after
checking formatting, API correctness, model revision, and depth range.

## Instrumentation

Each raw record includes end-to-end generation latency, time to first token,
fixed single-forward latency, generated tokens, tokens/sec, seconds/generated
token, cap-hit status, natural-termination status, prompt tokens, parsed answer,
and correctness. Summary CSVs include 95% Wilson accuracy intervals.

## Runs

- `results/001_depth_scaling/a100_smoke/`: two-example A100 smoke run.
- `results/001_depth_scaling/h100_smoke/`: same two-example H100 smoke run;
  hardware latency is not directly comparable to A100.
- `configs/001_depth_scaling_n20.json`: next fixed 20-example validation run.

The 20-example run must pass instrumentation and output-inspection checks
before expanding to 100 examples. The completed cap-sensitivity analysis is
`results/001_depth_scaling/cap_sensitivity.md`; it selects a 1024-token cap for
the final N=100 run. The final run is the last Experiment 001 computation;
no Experiment 002 work should begin automatically afterward.

## Notes

Use the same prompts, prompt formatting, decoding settings, and random seed across all depths.
