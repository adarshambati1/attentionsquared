# Experiment 004 recovery plan

**Status:** frozen protocol, authoritative for all future Experiment 004 work  
**Principle:** repair the experiment before modifying the architecture.

The functional-v1 result does not answer whether Attention² works: its teacher
targets were shifted by one token and training/evaluation executed different
Attention² functions. This plan supersedes ad-hoc Experiment 004 execution
notes where they conflict. The trust classification for completed artifacts is
in [`../../results/TRUST_STATUS.md`](../../results/TRUST_STATUS.md).

## Non-negotiable constraints

- Preserve all raw data, caches, checkpoints, logs, and results. Never overwrite
  v1 artifacts with v2 and never delete artifacts without explicit approval.
- Huginn remains frozen: `grad(H*) = 0`. Only Attention² or an explicitly named
  control may train.
- The untouched 250-example GSM8K test set is for scientific evaluation only,
  never checkpoint selection.
- Functional behavior and GSM8K accuracy are primary. Latent metrics are
  diagnostics or controls.
- Do not modify topology, QKV, x-injection, or objective while repairing the
  baseline.
- Stop at a failed gate rather than changing protocol silently.

## Phase 0 — Freeze prior work

Apply the statuses in `results/TRUST_STATUS.md`. In particular:

- Exp1 is trusted.
- Exp2 is useful but incomplete.
- Exp3 downstream evidence remains useful; historical KL needs normalization.
- 4A remains mechanical/exploratory evidence.
- Trajectory 4B is exploratory and seed-confounded.
- Trajectory 4C is invalid/provisional under one-token Attention² execution.
- Functional cache/models/evaluation v1 are invalid for scientific inference.

## Phase 1 — Shared correctness layer

Replace divergent experiment-local implementations with shared utilities for:

- prompt and chat-template construction;
- answer boundaries and `valid_end`;
- causal next-token alignment;
- answer extraction and stop strings;
- cap-hit and degeneration handling;
- deterministic per-example `h0` generation/seeding;
- token-normalized KL;
- synchronized CUDA timing; and
- generation scoring.

One scorer is authoritative for all future experiments.

## Phase 2 — Causal supervision test

If answer tokens are `input_ids[answer_start:valid_end]`, supervise them using
`logits[answer_start-1:valid_end-1]`.

Add a literal artificial `prompt | A B C` test proving:

- the logit before A targets A;
- the logit before B targets B;
- the logit before C targets C;
- prompt positions have zero loss; and
- positions at or after `valid_end` have zero loss.

## Phase 3 — Frozen splits

Freeze the 2,500 training-side examples as:

- 2,250 train examples for gradient updates;
- 250 validation examples for checkpoint selection; and
- a separate untouched 250-example GSM8K test set for final evaluation.

Store exact IDs and split-generation policy in the manifest. Never select a
checkpoint using the test set.

## Phase 4 — Compact functional cache v2

Do not regenerate a full-vocabulary logit cache. Store per example:

- `input_ids`, `attention_mask`, and token metadata;
- `answer_start` and `valid_end`;
- `h0_full [T,H]`;
- `x_full [T,H]`;
- `h16_teacher [T,H]`; and
- seed metadata.

During training, run both teacher and student states through the same frozen
Huginn coda:

```text
h16_teacher --no_grad--> frozen coda --> detached teacher logits
(h0,x) --> Attention² --> z16 --> frozen coda --> student logits --> loss
```

Coda parameters remain frozen, but autograd must flow through the coda into
`z16` and Attention².

## Phase 5 — Crash-safe cache and manifest

For each cache item:

```text
write unique temporary file
close it
reopen and validate keys/shapes/dtypes/finiteness
atomically rename to final .npz
```

Resume logic must validate contents rather than only call `path.exists()`.
The immutable manifest records:

- Huginn model and revision;
- dataset and revision;
- tokenizer/chat template;
- exact repository commit;
- dtype and extraction location;
- sequence count and split IDs;
- seed policy; and
- cap/degeneration policy.

## Phase 6 — Durable storage

Before the Pod can be destroyed, copy cache v2 to durable workspace or network
storage and verify checksums. Preserve v1 until explicit permission to delete
it. `/root` is not accepted as the sole long-term copy.

## Phase 7 — Degeneration review

Inspect all 27 capped teacher continuations. Assign `valid_end` where repetitive
degeneration clearly starts, preserve every valid prefix, and never treat the
artificial 1,024-token cap as EOS. Record the review decision per example.

## Phase 8 — Padding semantics

Implement causal and key-padding masks in `Attention2Lite.operator()`. Test that
changing padded token values cannot affect any real-token output. Batch-size-one
smoke work may proceed first, but no padded batched result is trusted until this
passes.

## Phase 9 — Deterministic paired h0

Explicitly create/reuse `h0(example, seed, step)` across Huginn D16,
Attention² K1/K2/K4, and controls. One seed per example is sufficient for smoke
tests. Final results use three paired seeds per example and report variation.

## Phase 10 — Paired K initialization

Create one immutable `a2_init_seed_X.pt` and load the same state dictionary for
K=1, K=2, and K=4. Keep data split, minibatch order, preprocessing, optimizer,
loss, and random policy fixed where possible. K=8 remains a separate stability
experiment.

## Phase 11 — Mandatory unit gates

Before real training, executable tests must prove:

- all Huginn and coda parameter gradients are absent;
- Attention² receives nonzero gradients;
- `dL/dz16` is nonzero through the frozen coda;
- measured `KL(pT || pT)` is approximately zero;
- causal target alignment is exact;
- prompt/post-`valid_end` positions receive zero loss;
- padded tokens cannot influence real tokens; and
- initial untrained Attention² KL/token is recorded.

Do not hardcode control values.

## Phase 12 — Eight-example functional-overfit gate

Use eight examples, no trajectory loss, and the unchanged baseline:

```text
(h0,x) --> Attention² --> z16 --> frozen coda
```

Optimize token-normalized `KL(pT || pS)` only on valid answer tokens. Require a
dramatic KL/token decrease, improved first-answer-token predictions,
approaching teacher/student distributions, and sensible generation on the same
eight examples. Stop if this gate fails.

## Phase 13 — Correct full-prefix autoregressive evaluator

For every generation step, recompute the entire prompt plus generated prefix,
then run `h0/x --> Attention² --> frozen coda --> next-token logits`. Do not pass
only the newest token to Attention². This evaluator establishes correctness,
not speed.

## Phase 14 — Evaluator controls

Before evaluating Attention²:

1. Run frozen Huginn D16 through the shared evaluator and recover approximately
   the known 40.4% same-split teacher result.
2. Verify true cached `h16` agrees with normal teacher behavior.
3. Match Exp1 stop strings, cap handling, fallback rules, and answer extraction.

Do not proceed if these controls disagree.

## Phase 15 — Correct trajectory-checkpoint reevaluation

Run existing trajectory K1/K2/K4 checkpoints through full-prefix generation.
Treat each model's corrected accuracy as useful, but retain the seed-confound
label on cross-K comparisons. Do not use functional-v1 checkpoints for claims.

## Phase 16 — Correct functional K1 smoke

Train only K=1 for 200–500 steps using cache v2, paired initialization, exact
alignment, and the frozen train/validation split. Evaluate validation KL per
valid answer token and run full-prefix downstream generation. K2/K4 remain
blocked until the complete cache-to-generation pipeline passes.

## Phase 17 — Full K1

After smoke success, train K1 for at least 2,000 steps and continue toward 5,000
only while validation KL/token improves. Validate/checkpoint every 100–200
steps and retain the best validation checkpoint, never a test-selected one.

## Phase 18 — Correct K2/K4

Train K2 and K4 from the same base state and data ordering as K1. Compare
validation KL/token, untouched-test GSM8K accuracy, generation stability, and
endpoint diagnostics.

## Phase 19 — K8 stability study

After valid K1/K2/K4 results, diagnose K8 roundwise hidden-state norms, gradient
norms, residual magnitudes, and attention-output scales. Any stabilizer—learning
rate, residual scaling, normalization, or clipping—is a separately documented
change.

## Phase 20 — Correct downstream evaluation

On the untouched 250-example GSM8K test set, run full-prefix K1/K2/K4 and report:

- GSM8K accuracy (primary);
- KL per valid answer token;
- cap hits and degeneration;
- average generation length; and
- endpoint cosine/L2 diagnostics.

## Phase 21 — Attention² KV cache

Only after full-prefix correctness, implement an incremental token-attention KV
cache. For many prefixes, require full-prefix and cached logits to agree within
a predefined tolerance and greedy next-token choices to be identical. Cached
execution cannot support speed claims until this passes.

## Phase 22 — Timing repair

Use one harness with GPU warmup and CUDA synchronization before/after timing.
Report separately: one Attention² round, one model forward, and end-to-end
generation. Historical 4C times are not final evidence.

## Phase 23 — Ordinary-attention functional control (4E)

Train an ordinary Transformer/sequential-adapter student with the same teacher,
functional objective, examples, checkpoint protocol, and approximately matched
parameters/training compute. Interpret outcomes as:

- both distill: Attention² can absorb the teacher function;
- neither distills: data/training/distillation is suspect; or
- ordinary attention distills while Attention² fails: evidence against
  Attention² expressivity or optimization.

## Phase 24 — Additional 4E controls

After a valid baseline, run:

- no-depth-attention Attention²;
- parameter/FLOP-matched feed-forward control;
- Experiment 3 direct jump; and
- sequential depth-attention prior art retaining recurrence.

## Phase 25 — Resume architecture ablations

Only after the functional baseline works, continue:

- **4D topology:** dense bidirectional, causal, local, and multiscale skips
  1/2/4/8;
- **4F QKV:** per-token depth QKV, pooled/global-depth QK with token V,
  value-focused mixing, shared/separate projections, and full joint
  `(depth,token)` QKV.

## Phase 26 — Explicit x-injection ablation

Document the repaired baseline as **Attention²-lite: x injected only at
initialization**. Do not alter it during baseline recovery. Later compare init-
only against x reinjection/cross-attention on every refinement round.

## Phase 27 — Objective ablations (4G)

After a valid functional-only baseline, compare functional-only,
trajectory-only, functional plus weak trajectory, endpoint emphasis,
delta/update supervision, and random versus compatible Huginn initialization.
These are ablations, not recovery gates.

## Phase 28 — Historical reporting repair

Before final reporting:

- recompute Exp3 and historical 4C KL as mean KL per valid token; and
- complete Exp2 sequence-mean trajectory analysis and long-range cosine at
  strides 1, 2, 4, 8, and 16.

These repairs do not block the corrected K1 smoke.

## Phase 29 — Final 4H benchmark

Only after correct causal execution/alignment/splits/scoring, paired seeds,
KV-cache equivalence, and synchronized timing, compare the winning Attention²
model with frozen Huginn D16. Report accuracy versus wall-clock latency plus K,
FLOPs, total/trainable parameters, VRAM, tokens/sec, KL/token, cap/degeneration,
and seed variance.

The target question remains whether `Attention²(K << 16)` approximates frozen
`Huginn(D=16)` with less sequential latency.

## Phase 30 — Matched post-training (4I)

Only after the frozen-teacher result, create Huginn+ and Attention²+ copies and
give both matched additional LM/reasoning training. Then test whether Attention²
learns useful new patterns better than recurrence.

## Locked execution order

1. Freeze and label existing artifacts.
2. Create shared scoring/alignment/timing utilities.
3. Freeze train/validation/test separation.
4. Build compact functional cache v2.
5. Make cache atomic and durable.
6. Review all capped examples and `valid_end` values.
7. Fix padding masks before padded batching.
8. Implement deterministic paired `h0`.
9. Create one shared Attention² initialization.
10. Run all unit gates.
11. Run eight-example functional overfit.
12. Build full-prefix autoregressive evaluator.
13. Validate it with Huginn D16 and true `h16` controls.
14. Re-evaluate trajectory checkpoints.
15. Run corrected K1 200–500-step smoke.
16. Train corrected K1 to convergence if clean.
17. Train corrected K2 and K4.
18. Evaluate untouched held-out GSM8K 250.
19. Diagnose/retry K8 separately.
20. Implement the Attention² KV cache.
21. Prove cached/full-prefix equivalence.
22. Measure synchronized latency.
23. Run ordinary-Transformer distillation control.
24. Run no-depth, parameter-matched, and prior-art controls.
25. Resume topology/QKV/objective ablations.
26. Repair Exp2 and historical KL reporting.
27. Run final 4H benchmark.
28. Run 4I matched post-training.
