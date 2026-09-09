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
- Pod test suite: 106 passed.
