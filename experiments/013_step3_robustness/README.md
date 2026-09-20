# Experiment 013 — Step 3 robustness

Step 3 of 11 tests whether most Huginn loop-history gain is explained by current-state processing and then applies capacity-matched controls to RecurTrace. Step 4 is blocked.

## Huginn (3A–3C) — approved for execution

Frozen checkpoints only. GSM8K uses three paired deterministic `h0` seeds on all 250 examples and evaluates plain D8, current D8, projected-uniform D8, shared-history D8, per-layer memory D8, plain D16, and plain D64. SVAMP and MATH-500 are zero-shot evaluations of plain D8, current D8, shared D8, plain D16, and plain D64. Depth transfer uses unchanged D8-trained plain/current/shared modules at D4, D8, D16, D32, and D64. Question-clustered intervals retain all three seeds together rather than treating 750 executions as independent questions.

Tensor batching was rejected after the full production-size A100 gate changed greedy token trajectories in all 17 conditions. The replacement `concurrent-serial-v3` protocol keeps every continuation at batch size one and executes disjoint condition workers concurrently. Its user-approved proportional gate checks exact 64-token parity for every actual production wave, complete natural-stop parity at every depth, and forced 1,024-token parity/OOM stress for the worst-memory D32 and D64 waves before evaluation. This validation change does not alter any 3A, 3B, or 3C prompt, condition, seed, cap, score, or estimand. The failed batched-v2 log and the original five serial records remain historical evidence in their separate output roots; v3 writes only to `/workspace/step3_huginn_robustness_concurrent_v3`.

## Qwen3-1.7B (3D) — implementation/gates only

The approved five arms are plain, shared current-only, shared history, per-layer current-only, and per-layer RecurTrace history. The frozen Qwen3-1.7B-Base backbone repeats zero-indexed layers 12–14. The implementation follows the paper's input reinjection and Loop Memory Attention equations, including four 128-dimensional heads, width 512, per-head QK RMS normalization, signed ALiBi-initialized loop-distance bias, memory window three, scalar and token gates, and no halting head.

Matched initialization and capacity are required within the causal pairs:

- shared history versus shared current-only;
- per-layer RecurTrace history versus per-layer current-only.

T=2 is primary. T=4 is the approved secondary diagnostic because multiple historical slots—and therefore temporal selection—do not exist at T=2.

The reference audit is in `RECURTRACE_REFERENCE_AUDIT.md`. The anonymous paper does not expose the official repository, final 1.15M-example manifest, prompt/packing implementation, or seed mappings. Training is therefore fail-closed. It may proceed only after a new immutable authorization manifest establishes either exact provenance or explicit user approval for a paper-guided controlled replication, and sufficient compute is provisioned.

Step 3D uses the separate environment in `requirements-step3-qwen.txt` (Transformers 4.57.1); canonical Huginn remains on Transformers 4.44.2.
