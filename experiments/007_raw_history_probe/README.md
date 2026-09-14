# Experiment 007 — Raw history probe

This focused D8 experiment separates unprojected state reuse from added trainable processing.

1. Plain frozen Huginn D8.
2. Current-state-only D8, trained with the same V/O projections and 1,000-step budget as the projected mean-history control, but with no historical states.
3. Parameter-free raw-mean-history D8, injecting the exact arithmetic mean of `[h0, ..., hk]` before recurrent loop `k`.

The current-state checkpoint is selected by full validation answer-token cross-entropy. The three models use the same held-out GSM8K test IDs and timing harness. Existing projected-mean, shared-attention, per-layer-memory, and D16 results remain frozen references.

Heavy-ball and Anderson are permitted next only if raw mean beats plain/current-only or lies within 2 absolute accuracy points of the best existing history arm. They are not part of this run.
