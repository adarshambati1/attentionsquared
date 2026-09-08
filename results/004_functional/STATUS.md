# Experiment 004 functional artifact status

## Version 1 — INVALID

The following remote artifacts are preserved for auditability but must not be
used for scientific conclusions:

- teacher-logit training cache: `/root/functional_training_cache/`;
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
`valid_end` annotations are reviewed. Do not overwrite or delete v1 artifacts.

## Version 2

Not yet built. All v2 work must follow
[`../../experiments/004a_attention2/RECOVERY_PLAN.md`](../../experiments/004a_attention2/RECOVERY_PLAN.md).
