# Experiment 010 — Task-only one-shot jump control

This is the single missing control for Experiment 009. It uses the exact same shared 256-bottleneck residual predictor, deterministic initialization, data splits, 1,000-step budget, optimizer, answer-token CE alignment, validation schedule, and zero-loop inference path.

The only changed training variable is `fixed_point_weight=0.0`. Training executes no recurrent core and optimizes answer-token next-token CE only. A frozen core application may be used by correctness and final diagnostic measurement, but contributes nothing to training or checkpoint selection. No architecture, coefficient, or seed sweep is performed. A pre-test crash preserves the full failed root and restarts from deterministic initialization; a held-out evaluation crash resumes only after contiguous records bound to the same config and checkpoint.
