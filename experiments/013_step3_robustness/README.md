# Experiment 013 — Step 3 robustness

Step 3 of 11 tests whether most loop-history gain is explained by current-state processing.

## Huginn (3A–3C)

Frozen checkpoints only. GSM8K uses six conditions and paired h0 seed indices 0/1/2 on the same 250 examples. SVAMP and MATH-500 use the minimum four requested conditions and the same paired seeds. Depth transfer uses unchanged plain/current/shared modules at D4, D8, D16, D32, and D64. Records preserve continuations, seed mappings, scoring, caps, repetition, synchronized generation latency, and peak memory. Summary statistics use example-clustered paired bootstrap intervals.

## Qwen3-1.7B (3D)

The frozen Qwen3-1.7B-Base backbone repeats zero-indexed layers 12–14 exactly twice. Plain, current-state, shared whole-loop history, and RecurTrace-style per-layer loop memory use the same backbone. The three modified variants train only added modules on identical MathQA examples/order, optimizer, update budget, validation split, precision, and initialization policy where parameters correspond. No halting head is implemented.

At a fixed two-loop budget, shared whole-block history has exactly one completed whole-loop state at its sole injection point. It therefore algebraically reduces to the current-state V/O adapter; this requested control is retained, trained on the identical order, and required to remain exactly equivalent rather than being treated as independent evidence.

The mechanisms intentionally have different parameter/FLOP counts because the requested current, shared, and three-placement per-layer RecurTrace modules are structurally different; fairness here means the frozen backbone, examples/order, token/update budget, optimizer, validation selection, and corresponding V/O/ReZero initialization are matched, not parameter-count matching. Trainable parameter counts are reported. Step 3D uses the one preregistered module/training seed; paired bootstrap intervals quantify held-out-example uncertainty conditional on that seed and do not generalize over training seeds.

Step 4 is blocked.
