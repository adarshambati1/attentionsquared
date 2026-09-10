# Experiment 004 functional artifact status

## Version 1 — INVALID

The following remote artifacts must not be used for scientific conclusions:

- teacher-logit training cache v1: deleted from
  `/root/functional_training_cache` with explicit authorization on 2026-09-09;
  its pre-deletion inventory and checksums are preserved under
  `deletion_records/functional_training_cache_v1_20260909/`;
- trained checkpoints: `/root/functional_models_full/`; and
- downstream evaluation: `/root/evaluation_functional_4c.json`.

Reasons:

1. answer-token targets used logits at `[answer_start:valid_end]` instead of
   `[answer_start-1:valid_end-1]`;
2. training used full-sequence Attention² token attention while generation
   invoked Attention² one token at a time without an Attention² KV cache;
3. checkpoint selection used only 16 validation examples;
4. evaluation did not pair stochastic `h0` across conditions;
5. evaluation KL was not token-normalized; and
6. latency lacked explicit CUDA synchronization.

Teacher continuation sequences remain reusable after all 27 capped examples'
`valid_end` annotations are reviewed. No deletion authorization extends to the
remaining v1 models, evaluations, or teacher continuations.

## Version 2

The compact cache is not yet built. All v2 work must follow
[`../../experiments/004a_attention2/RECOVERY_PLAN.md`](../../experiments/004a_attention2/RECOVERY_PLAN.md).

### Phase 4 input review — PASS/PASS

- All 2,500 raw continuation NPZ files remain unchanged.
- Both independent reviewers inspected every one of the 27 capped generations;
  all 27 contain clear repetitive degeneration.
- `capped_continuation_review.json` preserves decoded review evidence, both
  proposed boundaries, and conservative final adjudication. When proposals
  differed, the later onset was selected to preserve the longest defensible
  prefix.
- `valid_end_manifest.jsonl` contains exactly one source-hash-bound record for
  each GSM8K-train example ID 0–2499 in ID order. Historical source directory
  names are recorded separately from Phase 3 scientific roles.
- Manifest SHA-256:
  `25188506cb7afe426e42beb5fa96455fe8ac0cff6c0ae47b72363a3fa7af187f`.
- The artificial 1,024-token generation cap is recorded as truncation, never as
  EOS.

### Phase 5 crash-safe writer — PASS/PASS

- Valid finals are reused only after exact schema, content, seed, split, and
  manifest validation.
- Invalid finals and invalid/incomplete temporaries are moved—not deleted—to a
  visible `quarantine/` directory with unique names, SHA-256 hashes, reasons,
  and UTC timestamps, then regenerated.
- Complete valid temporary files are promoted atomically; mixed and multiple
  temporary-file cases preserve all non-selected files in quarantine.
- The cache manifest is bound to the exact reviewed Phase 4 sidecar SHA-256 and
  review protocol.
- Pod test suite: 84 passed after crash-recovery remediation.

### Phase 6 compact cache build — PASS/PASS

- Dedicated 100 GB network volume `jggambl3qv` in `US-GA-2` passed atomic
  no-replace, rename, fsync, capacity, and throughput preflight.
- Exact inputs, pinned environment, and pinned Huginn cache are present on H100
  Pod `jwc04iw92bdebm`; the old H100 is stopped but not deleted.
- Live D16 extraction produced finite `(T, 5280)` float16 state arrays and
  repeated byte-identically under the authoritative per-example seed.
- The builder stores only `h0_full`, `x_full`, `h16_teacher`, token/boundary
  arrays, and provenance. It creates one shared cache and no vocabulary logits
  or K-specific artifacts.
- New-Pod test suite: 91 passed after provenance remediation.
- Pre-build code/science review: PASS/PASS.
- The committed builder created exactly 2,500 items: 2,250 train and 250
  validation, with zero quarantine files and no K-specific directories.
- Cache size after construction: 20,529,356,986 bytes.
- Cache manifest SHA-256:
  `32db951f6cfbf76220b1c7126e23149f087eba58934ab691b588aeb3a43535bf`.
- Cache-internal canonical manifest digest:
  `c6d32e897701ae14805c84533cb0409a90ca6bb561557861a3375cf3f0b37efc`.
- Preserved build-log SHA-256:
  `e9d2197150fcae8a4d631c41a0a328913ffe9db87383e49e6ede407e7f3f7e81`.
- Full storage and migration record: `cache_v2_storage_preflight.md`.
- Preserved manifest/build records: `cache_v2_manifest.json` and
  `cache_v2_build.log`.

### Phase 7 validation and durability — PASS/PASS; cache v2 frozen

- Validator commit: `0cb3bf8694e5b451ec381a8b94251409083a6aa8`.
- All 2,500 items passed schema, shape, dtype, finiteness, bounds, split, seed,
  manifest, source-token, source-hash, and reviewed-`valid_end` validation.
- Counts: 2,250 train, 250 validation, 700,976 total tokens, and 462,952 valid
  answer tokens; hidden size is 5,280.
- Control-plane and mount checks bind the cache to dedicated 100 GB network
  volume `jggambl3qv` in `US-GA-2`; no symlinks, temporaries, or quarantine files
  are present.
- Live frozen-Huginn D16 replay for IDs 0, 22, 817, 1671, 2249, 2250, 2389, and
  2499 matched `h0_full`, `x_full`, and `h16_teacher` byte-for-byte.
- The re-read payload inventory covers the manifest plus all 2,500 items.
  Inventory SHA-256: `f0122058a273a973913b41f9b07fb557d4c6e20d99410f03b5ed77e5a4f20711`.
- Validation report SHA-256: `2a998d8458e8d5b9b7790395d213e4eca8b67485c4b1d1b57f06dce0e0dd1c62`.
- Freeze record SHA-256: `94417eab65fd04a5827bdef9aadc9a5b8b66c266700b6cfc7ab56d39f949d3f0`.
- Pod test suite: 100 passed.

### Phase 8 padding semantics — PASS/PASS

- `Attention2Lite` now combines a causal token-attention mask with a boolean
  key-padding mask in every refinement round.
- Padded states are made inert after every residual sublayer.
- Interspersed-padding tests prove arbitrary padded `h0/x` changes cannot alter
  any real-token output; separate tests prove causal future-token isolation and
  reject malformed/all-padding masks.
- Pod test suite: 109 passed.

### Phase 9 paired randomness and initialization — PASS/PASS

- Smoke/debug runs use deterministic Huginn `h0` seed index 0 per example;
  final results use paired seed indices 0, 1, and 2 per example under the Phase
  2 injective seed protocol.
- Huginn initialization runs in a forked RNG scope, restoring process-global
  CPU/CUDA RNG states after each deterministic `h0` creation.
- K=1, 2, and 4 load one strict, checksummed state dictionary from
  `/workspace/functional_protocol/a2_init_seed_0.pt`; K is not encoded in its
  parameters.
- Split roles, epoch order, optimizer, preprocessing, loss, and applicable
  seeds are frozen in `configs/004_functional_v2_training.json`.
- Focused Pod suite: 9 passed; full Pod suite: 118 passed.
- Production artifact builder commit:
  `e3b5c97e0fa49b6d654bc0f4569a93127ac07499`.
- `a2_init_seed_0.pt`: 1,338,868,224 bytes; file SHA-256
  `92968fea30d723864383f68f9e51ef1f8b8adae927e11acf735c3c1cf9b93747`;
  canonical state-dict SHA-256
  `93ac1869c68521ff5498ade5731c1ffea534527838df8a27d396490a043a533a`.
- Strict K=1, K=2, and K=4 loads independently reproduced the exact canonical
  state-dict hash; only each model's refinement-round count differed.
- Builder metadata SHA-256:
  `b949b934c63b6c4d4d273ba495c7f02b8ef38fc428fb79a6a66f625d5c0f0e0f`.
- The first shell wrapper could not open its optional tee log before the builder
  created the output directory. The builder itself completed and atomically
  published the artifact and metadata; a subsequent immutable validation log
  records all hashes and three strict K loads.
- Clean-checkout runtime attestation commit:
  `d1a3fbb3c78166f7fd01da055f3e7e27426e0ec7`; attestation SHA-256:
  `bfd5bef4fa4d33a9f0686b20967830411a4a2c6a69494982b2a8eb29687dd60e`.
  The artifact, builder record, prior validation log, and attestation are all
  regular non-symlink files published read-only; each observed mode is `0444`.

### Phase 9R prefix-stable cache-v3 smoke — COMPLETE PASS/PASS

- Phase-scoped code/config now targets only immutable raw teacher train IDs 0–7
  and the new atomic no-replace output
  `/workspace/functional_cache_v3_smoke`; cache v2 and raw continuations remain
  untouched.
- Each example derives its frozen seed under a forked CPU/CUDA RNG scope and
  calls Huginn `initialize_state` exactly once on a BF16 CUDA
  `[1,2048,5280]` template. Extraction and replay inject only views
  `schedule[:, :t]`; prefix-shaped initialization is forbidden.
- The builder locks split, reviewed-valid-end, cache-v2 manifest/freeze, and all
  eight raw source hashes. It asserts rendered prompt plus the 1,024-token cap
  and complete teacher sequences fit in 2,048 positions.
- Items contain only the needed float16 `h0/x/h16` sequence slices, token/mask
  and answer bounds, exact normalized-pre-coda semantics, schedule protocol and
  length, derived seed, transient full-schedule SHA-256, and provenance. No
  logits or complete 2,048-position schedule are persisted.
- The production validator independently parses and locks config, raw NPZs,
  reviewed boundaries, and the pinned GSM8K dataset; reconstructs each native
  chat prompt; rematerializes two schedules per example without builder
  schedule/extraction helpers; checks every represented prefix bit-for-bit; and
  runs separate normal-live and locally decomposed D16/coda forwards. It also
  enforces the exact immutable file/schema/provenance set and no persisted
  logits or full schedule. Prefix/full diagnostics cover `x`, `h16`, and logit
  mean/max absolute differences, global KL/token, and top-1 agreement.
- The builder now fails closed on actual `statvfs` free space, same-filesystem
  write/flush/fsync and directory-fsync probes, and a Linux `renameat2`
  `RENAME_NOREPLACE` collision that must preserve both source and existing
  target. The successful H100 storage preflight observed 249,318,764,183,552
  free bytes against the frozen 122,240,730-byte 2x requirement; evidence
  SHA-256: `3339ce43248adf92159605fd51c21eaf3a506e0fb5720df335e34a1db6655010`.
- Both active full-prefix evaluators now materialize one fixed BF16 CUDA
  `[1,2048,H]` schedule per generation and accept only explicit schedule slices
  for standalone next-token calls. No active autoregressive route uses
  dynamic-prefix initialization.
- Pinned H100 tests pass: 42 focused and 172 full-suite. An independent
  all-eight pre-build numerical audit reconstructed exact prompts, restored RNG
  twice per item, obtained equal complete schedule hashes and bitwise-equal
  represented prefixes, observed 16 D16 calls per route, and matched normal
  logits to a literal coda decomposition exactly. Prefix-vs-full diagnostics
  were recorded without an acceptance threshold: 664 prompt tokens,
  KL/token `0.0001880115442292159`, top-1 agreement
  `0.9954819277108434`, per-example logit mean absolute differences from `0`
  to `0.016039127483963966`, and maxima from `0` to `0.3125`; corrected v3
  evidence SHA-256:
  `874397fd65ab0fe59ae14f218a7dfd5cd80cfe680260e2e3d26155c19b71d2a1`.
- Review `08c1b7ef` remains preserved as the original pre-remediation
  FAIL/FAIL. Re-review `12f7fee2` returned code PASS and science FAIL solely
  because its v2 audit omitted descriptive logit mean/max fields. After the
  non-overwriting v3 audit added them, delta re-review `838c172` returned
  science PASS. Final authorization is PASS/PASS for only the
  eight-example smoke build.
- The smoke was built at commit `c8ba78263be3319c5366ac760e0b5e1ab9ebcbc7`
  and frozen mode `0555` with eight mode-`0444` items plus manifest. The
  independent production validator passed 8/8; after it was tightened to lock
  and record the exact builder commit, its non-overwriting v3 artifact SHA-256
  was `e578b111d87babe6a22cab6338ac46e103f0319d34877ca2962ef06974961db4`.
  Completion review `b4fd1c1e` returned science PASS; delta code re-review
  `c2036120` returned PASS. No optimizer step or Phase 10–13 rerun occurred
  during Phase 9R.

### Phase 10 correctness/unit-test gate — cache-v3 coda-v4 PASS

- The cache-v3 rerun passed on smoke item `00000.npz`, locking the cache
  manifest and independent v3 validation hashes and reading v3 `h0/x/h16`.
  New artifact SHA-256:
  `cb180b0aae45dbbb0ea3e1be4a1a85afcbae2271f108db08b52b47a1a86db8`.
  It used the unchanged shared initialization, 88 valid answer tokens, exact
  16-call latent/logit decomposition, all 21 A² parameters with gradients,
  zero Huginn parameter gradients, and zero optimizer steps. Teacher self-KL
  was `-2.9381495014746406e-09`; untrained K=4 KL/token was
  `11.253923416137695`. H100 full suite: 173 passed.
- The historical gate targeted the new no-replace artifact
  `/workspace/functional_protocol/correctness_gate_coda_v3.json`; the historical
  `correctness_gate.json` and all other v2 artifacts remain untouched.
- Cache-v2 `h16_teacher` and Attention² `z16` now enter one authoritative frozen
  coda helper as already-normalized states: coda blocks, final `ln_f`, then
  `lm_head`, with no extra initial `ln_f`.
- A production gold control wraps `core_block_forward`, requires exactly 16 live
  recurrent calls, captures the pre-`ln_f` D16 state, applies `ln_f` exactly
  once, and requires the model-returned latent and decomposed-coda logits to be
  exactly equal or tightly numerically equivalent to normal live D16 outputs.
- Pinned H100 tests passed: 21 focused and 151 full-suite. Pre-run review passed
  code/science with no Phase 10 scientific blockers.
- Corrected production gate passed at commit `f3fbceace25506e28f0229bed7cf6a7722a2607d`.
  The live decomposition was exact: 16 recurrent calls, one initial `ln_f`,
  exact latent equality, exact logit equality, and zero maximum absolute error.
- Artifact SHA-256: `a83561b945f88607554087ce79f4b690e79f6094afd3b05d25c07c5c1c539788`;
  authoritative runtime artifact is a regular non-symlink mode-`0444` file.
  Runtime attestation SHA-256:
  `3c80f8d27d96542f50ef221d7ff35e06d9a9ebb2b9d08ad611de9a42a2e424be`.
- Corrected teacher self-KL/token was `-8.677907004095431e-11`; untrained K=4
  KL/token was `11.100428581237793`. All 21 A² parameters had nonzero
  gradients, `z16` had 812,684 nonzero gradient elements (L2
  `0.17844003438949585`), and frozen Huginn had zero parameter gradients.

- Literal `prompt | A B C` alignment verifies logits at positions immediately
  preceding A, B, and C predict those answer tokens.
- Batched causal loss masks exclude prompt positions, post-`valid_end`
  positions, and padding; logits outside the mask have exactly zero influence.
- A frozen-coda gradient test requires nonzero Attention² and `z16` gradients
  while teacher/coda parameters receive no gradients.
- The production gate transiently computes real frozen-Huginn teacher logits,
  measured teacher self-KL, an untrained shared-initialization K=4 KL/token
  baseline, and a K=1 backward control. No logits are persisted.
- Focused Pod suite: 5 passed; full Pod suite: 123 passed.
- Real-model example 0 used 88 valid answer tokens. Teacher self-KL/token was
  calculated as `-2.5924651314568337e-09`; untrained K=4 KL/token was
  `11.122273445129395`.
- The K=1 backward control produced nonzero gradients for 21 Attention²
  parameters and 813,120 `z16` elements (`L2=0.14131605625152588`), while all
  3,564,976,800 frozen Huginn/coda parameters had no gradients.
- Prompt, post-`valid_end`, and padding loss-position counts were all zero;
  literal A/B/C alignment passed.
- Gate artifact SHA-256:
  `7da66ae3d85e35e0d4de96a754c81c338cc2b1be3f000e2363916138ceafa723`.
- Runtime stat attestation SHA-256:
  `9faa898c859ce80e845ac46829f526f410579c35f7aa38b18878febbb01474ba`;
  the authoritative gate artifact and attestation are regular, non-symlink,
  mode-`0444` files on the durable volume.

### Phase 11 v3 corrected eight-example functional-overfit gate — PASS

- The first cache-v3 run remains preserved as a formal failure under the
  mis-specified absolute `mean_first_target_probability >= 0.5` criterion.
  Diagnostic evidence showed Huginn itself averaged only `0.4795518257`.
  Science review `4e271d7c` approved the prospective teacher-relative
  replacement; no other model, objective, optimizer, data, initialization,
  seed, update, or generation setting changed.
- The one clean amended rerun used cache-v3 IDs 0–7, K=4, and shared
  initialization SHA-256 `92968fea30d723864383f68f9e51ef1f8b8adae927e11acf735c3c1cf9b93747`.
  It passed at update 60: KL/token `11.0844558173 -> 0.0105378922`, first-token
  KL `9.4341504574 -> 0.0481354306`, student target probability `0.4800678715`
  versus teacher `0.4795518257`, and first-token argmax agreement `8/8`.
- Generation passed all gates: 8/8 final-answer matches, 8/8 natural stops,
  zero cap hits, mean first-32 teacher-token match `0.96484375`, and 0/8
  detected loops, including IDs 2, 3, 5, and 7.
- The model/result are read-only under `/workspace/functional_overfit_v3`.
  Model SHA-256: `fc1a0099d2efd4fa233570a77ea6ed94e6ebb32bd948862ccdbb679362685c5a`;
  result SHA-256: `4860014dcadc6899ef9e212a49e0c995e4e28ad6fc60f64f5154417787fbba4d`.
  No full-vocabulary logits were persisted.

### Phase 11 eight-example functional-overfit gate — surrogate-path quantitative PASS; live-Huginn equivalence not established

- Uses frozen training-side examples 0–7, one deterministic cached `h0` seed
  per example, K=4 loaded from the Phase 9 shared initialization, and no
  trajectory loss.
- Every optimizer update accumulates the teacher-to-student KL numerator over
  all eight examples and divides once by the total valid-answer-token count.
- Predeclared pass criteria require at least an 80% global KL/token reduction
  to at most 0.5 KL/token; first-token distribution KL must fall at least 50%
  to at most 0.5; mean first-target probability must increase to at least 0.5;
  and first-target top-1 count must increase to at least 6/8.
- Teacher full-vocabulary logits are recomputed, consumed, and deleted for one
  example/loss computation at a time; they are never stored in examples or
  persisted.
- Greedy generation recomputes the entire prompt plus generated prefix with
  `use_cache=False` at every token. It records natural-stop/cap status, decoded
  teacher/generated text, teacher-prefix agreement, extracted-answer agreement,
  and degeneration checks. Automated adequacy requires at least 6/8 natural
  stops, fixed-width first-32 teacher-token agreement at least 0.5, and at
  least 6/8 authoritative cap-safe teacher-answer matches; missing generated
  prefix tokens count as mismatches and stop-only text is non-substantive.
  Final PASS still requires science review.
- A separate prelaunch attestation must show the final path absent and no other
  gate process. Each run uses a unique preserved attempt directory; only a
  complete read-only model/result pair can be atomically published no-replace.
- Focused Pod suite after cleanup: 12 passed; full Pod suite: 135 passed.
- Training completed from the exact shared K=4 initialization after 60 updates:
  KL/token fell from `11.134534463392416` to `0.010831465410149616`;
  first-token distribution KL fell from `9.623916625976562` to
  `0.04451802000403404`; first-target top-1 improved from 0/8 to 7/8.
- All eight generated sequences stopped naturally, none hit the cap, mean
  fixed-width first-32 teacher-token agreement was 0.53125, and authoritative
  teacher-answer agreement was 7/8. Quantitative and automated mechanical
  checks passed, but generation quality remains pending Phase 12 investigation.
- Manual comparison found all four teacher continuations question-specific and
  non-repetitive, but IDs 2, 3, and 7 contain internal reasoning contradictions;
  ID 5 is coherent within its stated interpretation. The students' semantic
  loops and off-topic page-themed prefixes are not present in their teachers,
  although teacher reasoning defects remain part of the supervision target.
  Causal attribution stays deferred to Phase 12. Evidence is preserved in
  `phase11_generation_comparison.json`.
- The initial durable-FUSE cross-directory atomic rename failed after complete
  model/result creation. Reviewed same-parent recovery then published
  `/workspace/functional_overfit_v2` atomically with no replacement, no
  retraining, and no byte mutation. The original attempt remains preserved.
- Published model SHA-256:
  `036a6578774d4352dd778a29a6c6214d1474ffb3b1e2d5e2db92e8267777777a`;
  result SHA-256:
  `0d84ac8353ca412c9e4b557d3fc42805e65585e4ec4e1f06b987f20340dbcd91`;
  publication-recovery record SHA-256:
  `dae6e1586b9cdfa5ba7c67410c5c0aaf715e5e71ae297a1ffde136ad7ec1ecda`.
  Both final files and both preserved source files are mode `0444`; final and
  source directories are mode `0555`.

### Phase 12 canonical full-prefix evaluator — historical engineering PASS/PASS; output superseded, corrected implementation pending review

- `FullPrefixAttention2Evaluator` now calls the authoritative normalized-state
  coda helper directly and cannot add a pre-coda `ln_f`. No corrected Phase 12
  scientific execution is authorized until corrected Phase 11 exists.
- The preserved v2 Phase 12 output remains historical engineering evidence only;
  it has not been overwritten or reclassified as corrected evidence.

- `FullPrefixAttention2Evaluator` recomputes Huginn's frozen prelude, paired
  deterministic `h0`, Attention² over the complete prefix, and the frozen coda
  for every generated token.
- Each step receives `prompt + all generated tokens`; no newest-token-only path,
  Huginn cache, or Attention² KV cache exists in this evaluator.
- Only the final-position vocabulary vector is materialized transiently for
  greedy selection; no logits are persisted.
- The Phase 12 runner is locked to the published Phase 11 model/result hashes,
  frozen cache, examples 0–7, K=4, seed index 0, and 384-token cap. It preserves
  exact generated token IDs, every prefix length, decoded text, stop/cap status,
  and comparison to the Phase 11 text.
- Output uses unique same-parent staging and atomic no-replace publication.
- Focused Pod suite: 6 passed; full Pod suite: 141 passed.
- Canonical evaluation commit:
  `db9df8296fce0b19beb7bcca213c23c38ba2dfa3`; evaluation SHA-256:
  `464827d502ff66dae9dc73029b4ac14acf77b7368ed274fed524ede8039b2e9f`.
- Every generated step for all eight examples recorded strict growth from the
  complete prompt through every generated prefix. All eight stopped naturally,
  none hit the 384-token cap, and no full-vocabulary logits were persisted.
- The canonical evaluator matched 3/8 Phase 11 texts exactly and matched 6/8
  teacher extracted answers. The semantic loops/off-topic prefixes on IDs 2,
  3, 5, and 7 persisted under canonical full-prefix recomputation.
- Phase 12 establishes that rebuilding the evaluator did not remove the
  qualitative behavior. Exact differences from five patched Phase 11 outputs
  remain evidence for Phase 13 equivalence controls; no causal attribution is
  made here, and Phase 13 has not begun.
- Runtime publication attestation SHA-256:
  `855cd779ab9f5d4ec4926df71f8ef185c9663743eb53cf744c9a8f7c999b62d1`.
  It records the authoritative evaluation as a regular non-symlink mode-`0444`
  file in a non-symlink mode-`0555` directory, published by same-parent
  `renameat2(RENAME_NOREPLACE)`.

### Phase 13 evaluator controls — BLOCKED; remediation implementation pending independent review

- The Phase 13 runner's `main` is hard-blocked in code pending independently
  reviewed corrected Phase 10 and new corrected Phase 11/12 artifacts. Helpers
  remain importable for static/unit testing. No Phase 13 scientific execution
  is authorized.

- Historical reference: GSM8K train IDs 2250–2499, D16, bfloat16, greedy,
  1024-token cap; base seed 3000/index 0 is prospective because the historical
  40.4% run's realized RNG provenance is unknown.
- Report historical-emulation and corrected scores separately. The 40.4% result
  is a reference and `[0.34, 0.47]` is only a broad anomaly band.
- Live-reference versus new full-prefix Huginn generation requires exact token,
  stop/cap, extracted-answer, and correctness agreement.
- Cached/live teacher-prefix bounds are frozen before execution: hidden maximum
  absolute error `<=0.003`, logit mean/max absolute error `<=0.005`/`<=0.06`,
  KL/token `<=5e-4`, and 100% next-token argmax agreement. Float16 bitwise
  identity is diagnostic rather than a hard gate.
- Cache equivalence claims are limited to represented teacher prefixes.
- The phase-scoped runner/config and dedicated literal tests are implemented in
  the current worktree. They lock train IDs 2250–2499, prospective base seed
  3000/index 0, dual historical/corrected scoring, same-object `h0` reuse for
  the two live D16 routes, validation cache IDs 2250–2257 over every valid
  teacher prediction position, the frozen aggregate bounds, normal-coda versus
  functional-coda semantics, and atomic read-only pass/fail publication.
- H100 tests: 12 focused passed; 147 full-suite passed.
- Independent pre-run review `50629db0` returned code FAIL / science FAIL.
  The functional path proved equality to Huginn `num_steps=0` but did not prove
  equality to normal live D16: both paths applied an extra `ln_f` to cache v2's
  already-normalized coda input. The two proposed live routes were also not
  independent, and IDs 2250–2257 omitted sequence-length extremes.
- Existing Phase 11 therefore proves optimization of the supplied surrogate
  path, not live-Huginn functional equivalence. Existing Phase 12 remains valid
  engineering evidence for full-prefix execution, but its generations are not
  authoritative evidence about live-Huginn-equivalent model behavior.
- Remediation will use new v3 outputs and preserve all v2 artifacts. No Phase 13
  execution or Phase 14 work has begun.

### Phase 13 pre-run cache-fidelity diagnosis — STOP; cache-v3 retained as lossy evidence

- The independent live whole-model versus literal D16/coda route passed, so the corrected evaluator and coda semantics remain supported.
- Cache-v3 failed the unchanged Phase 13C cache/live gate: mean logit error `0.005615674883951783`, maximum logit error `0.125`, and argmax agreement `1264/1265`.
- The frozen diagnostic proves cache-v3 stores the live BF16 states rounded to FP16 and that losslessly replaying the original BF16 states restores live coda logits exactly for all eight items.
- Cache-v3 remains immutable and must not be overwritten. It is prefix-stable but numerically lossy for exact teacher equivalence. Phase 13 was not executed and Attention² was not tested by this failure.
- Frozen diagnosis: `cache_v3_lossy_diagnosis.md`; final diagnostic SHA-256 `dcf283c53c87813a9bbfc797d01e5c6b5ce726f709912e83c0afa57f92f10ae9`.
- Remediation creates a separate eight-item `/workspace/functional_cache_v4_smoke` preserving raw BF16 bits for `h0`, `x`, and normalized pre-coda `h16`; no logits are persisted and cache-v3 remains untouched.

### Cache-v4 BF16-only smoke attempt — FAILED AND FROZEN

- Builder commit `057a3955388676de3fa61ad21acd843d6253fcf3` attempted a separate no-replace BF16-bit cache; cache-v3 was untouched.
- The first item failed before publication because the hard dtype gate observed canonical `x` and normalized `h16` as FP32, not BF16. No final `/workspace/functional_cache_v4_smoke` exists and no training occurred.
- Failed staging evidence is frozen at `/workspace/.functional_cache_v4_smoke.attempt-86c562b0557c4d79b4eb69fe3c09f065` (mode `0555`, manifest and lock mode `0444`).
- Failed build-log SHA-256: `87b224e183fd65c2a409dff5acd2f84347af1ed8d8a4ce343795a64c8ed0fe3b`.
- Independent canonical dtype verification across IDs 0, 2, and 7 confirmed native `h0` BF16 and native `x`/core outputs/normalized `h16` FP32. Evidence SHA-256: `de735cefcc7b51b2d33257bcd91662e433262f2d3806dbd3fef933756dce7bac`.
- Cache-v5 is a new native-dtype object: raw BF16 bits for `h0`, exact FP32 for `x` and `h16`.
