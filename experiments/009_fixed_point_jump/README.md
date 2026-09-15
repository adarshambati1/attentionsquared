# Experiment 009 — Functional one-shot fixed-point jump

This experiment asks whether a learned tokenwise map can directly produce a useful recurrent latent from Huginn's initialization and frozen prelude output:

`(h0, x) -> h* -> frozen native normalization/coda/head -> logits`.

The predictor is one shared residual MLP with a fixed 256-wide bottleneck and zero-initialized output projection. Huginn is frozen. At inference, no recurrent core step is executed.

Training uses answer-token next-token CE, including the first answer token, plus a fixed weight `lambda=0.1` on the token-weighted mean of per-example answer-position aggregate relative fixed-point residuals:

`||F(h*,x)-h*||^2 / (||h*||^2 + 1e-12)`.

The consistency image requires one frozen core application during training only. Predictor initialization uses frozen seed 90091. There is no `h16`/`h64` target, latent imitation, trajectory cache, architecture sweep, or lambda sweep. Checkpoint selection uses the lowest full validation answer-token CE on train IDs 2250–2499; held-out GSM8K test IDs 0–249 are evaluated once.
