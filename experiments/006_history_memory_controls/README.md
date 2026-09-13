# Experiment 006 — History-memory attribution controls

This experiment compares four D8 models under the Experiment-005 data, optimization, checkpoint-selection, scoring, and timing protocol:

1. Frozen plain Huginn D8.
2. Mean-history D8: uniform averaging across completed loop states, with trainable value/output projections.
3. Shared learned-history D8: the frozen selected Experiment-005 checkpoint.
4. Per-layer learned-history D8: one 256-wide, four-head history-attention module at each recurrent-core layer, reading that layer's same-token states across completed loops.

Huginn remains frozen. New output projections start at zero; no multiplicative gate, FFN, slots, distillation, or auxiliary loss is added. Mean-history and per-layer history each receive the same 1,000-step answer-token cross-entropy budget and validation-loss checkpoint selection used in Experiment 005.

The D16 result from the immutable Experiment-005 comparison remains the higher-compute reference. The four D8 variants are evaluated together. No further mechanism or sweep is launched from this experiment.
