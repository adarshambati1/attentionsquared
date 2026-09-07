# Experiment 003 conclusion: direct-jump baseline

## Protocol

Huginn was frozen. We cached exact tokenwise triples `(h0, x, h16)` for 2,500
GSM8K train-split prompts using the same Huginn checkpoint, tokenizer chat
template, system prompt, and bfloat16 settings as Experiment 001. Splits were
2,000 train / 250 validation / 250 test, with per-example random initialization
seeds preserved.

Two deliberately non-recurrent, tokenwise predictors were trained:

- **Direct:** `G(h0, x)`
- **Residual:** `h0 + G(h0, x)`

Both use `concat(h0, x) -> Linear -> GELU -> Linear`. No token attention,
depth communication, recurrence, skip graph, or Attention² mechanism was used.

## Results

| Predictor | Val cosine | Val relative L2 | Test cosine | Test relative L2 | Teacher-logit KL | GSM8K accuracy | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 0.737 | 0.552 | 0.751 | 0.533 | 354.44 | 0.00 | 2.40s |
| Residual | 0.704 | 0.631 | 0.717 | 0.616 | 358.83 | 0.00 | 2.95s |

The test set contains 250 examples. Each predictor has 83,645,760 parameters
and approximately 167,270,400 multiply-add FLOPs per token under the simple
linear-layer estimate used here.

## Interpretation

The direct jump captures substantial coarse geometry, but it does not preserve
Huginn's downstream behavior: both predictors achieve zero exact-match GSM8K
accuracy and very large teacher-logit KL despite moderate latent cosine. The
residual form is worse on every latent metric and is not a better shortcut.

This is the key Experiment 003 result:

```text
trajectory convergence does not imply one-shot endpoint predictability
```

Experiment 002 showed that later recurrent states become locally stable, while
Experiment 003 shows that a single tokenwise jump from `(h0, x)` cannot recover
the task-relevant computation. This supports moving to Experiment 004: test
whether a small number of structured parallel refinement rounds can close the
gap. Exp4 must still compare local-only, full depth attention, multiscale skip
attention, and an appropriate residual hybrid; none was implemented here.

For context, Experiment 001's H100 N=100 reference run reached 0.42 GSM8K
accuracy at D=16, but that result used the first 100 GSM8K test examples rather
than this held-out 250-example train-split subset, so it is a reference rather
than a paired accuracy comparison.
