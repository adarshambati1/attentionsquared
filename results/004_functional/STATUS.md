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
- Pod test suite: 78 passed.
