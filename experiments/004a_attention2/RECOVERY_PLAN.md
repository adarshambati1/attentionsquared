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

## Phase 1 — Freeze the scientific record

Apply the statuses in `results/TRUST_STATUS.md`. In particular:

- Exp1 is trusted.
- Exp2 is useful but incomplete.
- Exp3 downstream evidence remains useful; historical KL needs normalization.
- 4A remains mechanical/exploratory evidence.
- Trajectory 4B is exploratory and seed-confounded.
- Trajectory 4C is invalid/provisional under one-token Attention² execution.
- Functional models/evaluation v1 are invalid for scientific inference.
- The invalid giant functional-logit v1 cache was deleted with explicit user
  authorization; preserve its committed inventory and checksums and never
  regenerate it.

## Phase 2 — Shared correctness infrastructure

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

One scorer is authoritative for all future experiments. A literal artificial
`prompt | A B C` test must prove that answer tokens
`input_ids[answer_start:valid_end]` are supervised by logits
`logits[answer_start-1:valid_end-1]`, with zero loss on prompt targets and at or
after `valid_end`.

## Phase 3 — Frozen splits

Freeze the 2,500 training-side examples as:

- 2,250 train examples for gradient updates;
- 250 validation examples for checkpoint selection; and
- a separate untouched 250-example GSM8K test set for final evaluation.

Store exact IDs and split-generation policy in the manifest. Never select a
checkpoint using the test set. Historical continuation directory names do not
define these roles: all 2,500 continuations came from GSM8K train.

## Phase 4 — Finalize v2 inputs

Keep every raw teacher-continuation NPZ exactly unchanged. Review all 27 capped
continuations and record decisions in one immutable `valid_end_manifest.jsonl`
sidecar. For every example record source identity/hash, Phase 3 role, original
and reviewed `valid_end`, cap/degeneration flags, and review decision.

Where repetitive degeneration clearly starts, training excludes the tail via
`valid_end`; it does not truncate the source file. Preserve every valid prefix
and never treat the artificial 1,024-token cap as EOS.

## Phase 5 — Crash-safe cache writer and manifest

For each cache item:

```text
write unique temporary file
close and fsync
reopen and validate keys/shapes/dtypes/finiteness/provenance
atomically rename to final .npz
```

Resume behavior is:

- valid final: validate exact expected contents and reuse;
- invalid final: move to a uniquely named quarantine record and regenerate;
- complete valid temporary file: validate and atomically promote; and
- invalid/incomplete temporary file: quarantine and regenerate.

Never trust `path.exists()` and never silently delete or overwrite corruption.
The immutable manifest records Huginn model/revision, dataset identity/revision,
tokenizer/template, exact repository commit, dtype/extraction location,
sequence count and exact split IDs, seed policy, and cap/degeneration policy.

## Phase 6 — Build compact functional cache v2

Do not regenerate teacher continuations and do not store full-vocabulary
logits. Run frozen Huginn over the exact reviewed continuations and store:

- `input_ids`, `attention_mask`, and token metadata;
- `answer_start` and reviewed `valid_end`;
- `h0_full [T,H]`;
- `x_full [T,H]`;
- `h16_teacher [T,H]`; and
- seed/example/manifest metadata.

Use the Phase 5 writer from the first item onward and build one shared cache for
all future K values. During later training, run teacher and student states
through the same frozen Huginn coda:

```text
h16_teacher --no_grad--> frozen coda --> detached teacher logits
(h0,x) --> Attention² --> z16 --> frozen coda --> student logits --> loss
```

Coda parameters remain frozen, but autograd must flow through the student coda
into `z16` and Attention². Full-vocabulary logits exist only transiently in GPU
memory and are discarded after loss computation.

## Phase 7 — Validate and make cache v2 durable

Before extraction, verify real durable capacity, write access, atomic rename
behavior, and throughput. After all 2,500 items are built:

- validate every item and the immutable manifest;
- verify exact IDs, shapes, dtypes, finiteness, bounds, and source hashes;
- replay selected examples through frozen Huginn and compare cached states;
- create a SHA-256 inventory;
- place v2 on durable workspace or network storage and verify exact checksums;
  and
- mark the shared functional cache v2 frozen.

The invalid giant v1 logit cache was deleted with explicit authorization and a
committed audit record. Never recreate it, silently overwrite another artifact,
or create K-specific caches.

## Phase 8 — Padding semantics

Implement both causal and key-padding masks in `Attention2Lite.operator()` and
thread `token_mask` through every refinement round. Test with interspersed
padding—not only right padding—that arbitrary changes to padded `h0/x` values
cannot affect any real-token output. Padded outputs remain inert. Batch-size-one
smoke work may proceed without padding, but no padded batched result is trusted
until this gate passes.

## Phase 9 — Deterministic paired randomness and initialization

Explicitly create/reuse `h0(example, seed, step)` across Huginn D16,
Attention² K1/K2/K4, and controls. One seed per example is sufficient for smoke
tests. Final results use three paired seeds per example and report variation.

Create one immutable shared Attention² state dictionary, initially
`a2_init_seed_0.pt`, and load those exact weights for K=1, K=2, and K=4. Keep
data split, minibatch order, preprocessing, optimizer, loss, and random policy
fixed where possible. K=8 remains a separate stability experiment.

## Phase 10 — Mandatory correctness and unit gates

Before real training, executable tests must prove:

- literal causal alignment: if answer tokens are
  `input_ids[answer_start:valid_end]`, supervise them with
  `logits[answer_start-1:valid_end-1]`, including explicit `A B C` targets;
- all Huginn and coda parameter gradients are absent;
- Attention² receives nonzero gradients;
- `dL/dz16` is nonzero through the frozen coda;
- prompt and post-`valid_end` positions receive zero loss;
- padding receives zero loss and has zero influence on real tokens;
- measured, non-hardcoded `KL(pT || pT)` is approximately zero; and
- initial untrained Attention² KL per valid answer token is measured and
  recorded.

## Phase 11 — Eight-example functional-overfit gate

Use eight examples, no trajectory loss, and the unchanged baseline:

```text
(h0,x) --> Attention² --> z16 --> frozen coda
```

Optimize token-normalized `KL(pT || pS)` only on valid answer tokens. Require a
dramatic KL/token decrease, improved first-answer-token predictions,
approaching teacher/student distributions, and sensible generation on the same
eight examples. Stop if this gate fails.

## Phase 12 — Correct full-prefix autoregressive evaluator

For every generation step, recompute the entire prompt plus generated prefix,
then run `h0/x --> Attention² --> frozen coda --> next-token logits`. Do not pass
only the newest token to Attention². This evaluator establishes correctness,
not speed.

## Phase 13 — Evaluator controls

Freeze these criteria before execution. They validate evaluator and cache
plumbing; they do not change Attention², K, training data, loss, teacher, or the
Phase 12 generation procedure.

### 13A — Historical-control provenance

Run frozen Huginn D16 on the historical same-split control: GSM8K train IDs
2250–2499, pinned model and dataset revisions, bfloat16, greedy decoding,
1024-token cap, native chat template, and the shared `<|end_text|>` /
`<|end_turn|>` stop policy. Use base seed 3000, seed index 0, and label it
`prospective deterministic reproduction seed; historical 40.4% RNG provenance
unknown`. Preserve every derived per-example seed.

Report both historical-emulation and authoritative corrected scores. The known
101/250 (40.4%) is a reference, not an exact deterministic gate. `[0.34, 0.47]`
is only a broad anomaly band, not evidence of route equivalence. Audit every
per-example scoring disagreement.

### 13B — New evaluator versus live Huginn

On identical examples and complete growing prefixes, run normal live Huginn D16
and the new full-prefix Huginn route with identical `h0`, seed, and greedy
decoding. Require exact equality of generated token IDs, stop reason, cap flag,
extracted answer, and correctness. Any categorical disagreement fails Phase 13.

### 13C — Cached `h16` versus live `h16`

Compare only cached teacher-continuation prefixes. Freeze these quantization-aware
bounds, based on measurements made before Phase 13:

- hidden-state maximum absolute error `<= 0.003`, with cosine and relative-L2
  reported;
- mean absolute logit error `<= 0.005`;
- maximum absolute logit error `<= 0.06`;
- KL per compared token `<= 5e-4`;
- 100% next-token argmax agreement.

Also report bitwise `live_h16.to(float16) == cached_h16` as a diagnostic, not a
hard scientific gate. Investigate rather than loosen these bounds if cache-v2
statistics fall outside the preregistered envelope.

### 13D — Scope of equivalence

Cached/live equivalence covers only represented teacher prefixes. It does not
establish equivalence on arbitrary student branches. The full-prefix live route
remains authoritative for autoregressive evaluation until a separately
validated Attention² KV cache exists.

Match Experiment 1 stop strings, cap handling, historical fallback behavior,
authoritative corrected fallback behavior, and answer extraction. Persist no
full-vocabulary logits. Do not proceed if 13B or 13C disagrees, or if 13A falls
outside its broad anomaly band without a resolved audit.

## Phase 14 — Correct trajectory-checkpoint reevaluation

Run existing trajectory K1/K2/K4 checkpoints through full-prefix generation.
Treat each model's corrected accuracy as useful, but retain the seed-confound
label on cross-K comparisons. Do not use functional-v1 checkpoints for claims.

## Phase 15 — Correct functional K1 smoke

Train only K=1 for 200–500 steps using cache v2, paired initialization, exact
alignment, and the frozen train/validation split. Evaluate validation KL per
valid answer token and run full-prefix downstream generation. K2/K4 remain
blocked until the complete cache-to-generation pipeline passes.

## Phase 16 — Full K1

After smoke success, train K1 for at least 2,000 steps and continue toward 5,000
only while validation KL/token improves. Validate/checkpoint every 100–200
steps and retain the best validation checkpoint, never a test-selected one.

## Phase 17 — Correct K2/K4

Train K2 and K4 from the same base state and data ordering as K1. Compare
validation KL/token, validation generation stability, and endpoint diagnostics.
The untouched GSM8K test set remains sealed until Phase 19.

## Phase 18 — K8 stability study

After valid K1/K2/K4 results, diagnose K8 roundwise hidden-state norms, gradient
norms, residual magnitudes, and attention-output scales. Any stabilizer—learning
rate, residual scaling, normalization, or clipping—is a separately documented
change.

## Phase 19 — Correct downstream evaluation

On the untouched 250-example GSM8K test set, run full-prefix K1/K2/K4 and report:

- GSM8K accuracy (primary);
- KL per valid answer token;
- cap hits and degeneration;
- average generation length; and
- endpoint cosine/L2 diagnostics.

## Phase 20 — Attention² KV cache

Only after full-prefix correctness, implement an incremental token-attention KV
cache. For many prefixes, require full-prefix and cached logits to agree within
a predefined tolerance and greedy next-token choices to be identical. Cached
execution cannot support speed claims until this passes.

## Phase 21 — Timing repair

Use one harness with GPU warmup and CUDA synchronization before/after timing.
Report separately: one Attention² round, one model forward, and end-to-end
generation. Historical 4C times are not final evidence.

## Phase 22 — Ordinary-attention functional control (4E)

Train an ordinary Transformer/sequential-adapter student with the same teacher,
functional objective, examples, checkpoint protocol, and approximately matched
parameters/training compute. Interpret outcomes as:

- both distill: Attention² can absorb the teacher function;
- neither distills: data/training/distillation is suspect; or
- ordinary attention distills while Attention² fails: evidence against
  Attention² expressivity or optimization.

## Phase 23 — Additional 4E controls

After a valid baseline, run:

- no-depth-attention Attention²;
- parameter/FLOP-matched feed-forward control;
- Experiment 3 direct jump; and
- sequential depth-attention prior art retaining recurrence.

## Phase 24 — Resume architecture ablations

Only after the functional baseline works, continue:

- **4D topology:** dense bidirectional, causal, local, and multiscale skips
  1/2/4/8;
- **4F QKV:** per-token depth QKV, pooled/global-depth QK with token V,
  value-focused mixing, shared/separate projections, and full joint
  `(depth,token)` QKV.

## Phase 25 — Explicit x-injection ablation

Document the repaired baseline as **Attention²-lite: x injected only at
initialization**. Do not alter it during baseline recovery. Later compare init-
only against x reinjection/cross-attention on every refinement round.

## Phase 26 — Objective ablations (4G)

After a valid functional-only baseline, compare functional-only,
trajectory-only, functional plus weak trajectory, endpoint emphasis,
delta/update supervision, and random versus compatible Huginn initialization.
These are ablations, not recovery gates.

## Phase 27 — Historical reporting repair

Before final reporting:

- recompute Exp3 and historical 4C KL as mean KL per valid token; and
- complete Exp2 sequence-mean trajectory analysis and long-range cosine at
  strides 1, 2, 4, 8, and 16.

These repairs do not block the corrected K1 smoke.

## Phase 28 — Final 4H benchmark

Only after correct causal execution/alignment/splits/scoring, paired seeds,
KV-cache equivalence, and synchronized timing, compare the winning Attention²
model with frozen Huginn D16. Report accuracy versus wall-clock latency plus K,
FLOPs, total/trainable parameters, VRAM, tokens/sec, KL/token, cap/degeneration,
and seed variance.

The target question remains whether `Attention²(K << 16)` approximates frozen
`Huginn(D=16)` with less sequential latency.

## Phase 29 — Matched post-training (4I)

Only after the frozen-teacher result, create Huginn+ and Attention²+ copies and
give both matched additional LM/reasoning training. Then test whether Attention²
learns useful new patterns better than recurrence.

## Locked execution order

1. Freeze and label the scientific record.
2. Create shared correctness infrastructure and prove causal supervision.
3. Freeze train/validation/test separation.
4. Review capped continuations and freeze `valid_end` metadata.
5. Validate the crash-safe writer, quarantine, resume, and manifest behavior.
6. Build one compact functional cache v2 from the preserved continuations.
7. Validate every item, checksum durable storage, and freeze cache v2.
8. Fix padding masks before padded batching.
9. Pair deterministic `h0` and load one shared Attention² initialization.
10. Run all mandatory correctness and unit gates.
11. Run eight-example functional overfit.
12. Build the full-prefix autoregressive evaluator.
13. Validate it with Huginn D16 and true `h16` controls.
14. Re-evaluate trajectory checkpoints.
15. Run corrected K1 200–500-step smoke.
16. Train corrected K1 to convergence if clean.
17. Train corrected K2 and K4.
18. Diagnose/retry K8 separately.
19. Evaluate untouched held-out GSM8K 250.
20. Implement the Attention² KV cache and prove cached/full-prefix equivalence.
21. Measure synchronized latency.
22. Run the ordinary-Transformer distillation control.
23. Run no-depth, parameter-matched, and prior-art controls.
24. Resume topology and QKV ablations.
25. Run the explicit x-injection ablation.
26. Run objective ablations.
27. Repair Exp2 and historical KL reporting.
28. Run final 4H benchmark.
29. Run 4I matched post-training.
