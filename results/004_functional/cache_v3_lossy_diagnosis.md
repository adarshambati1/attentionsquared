# Cache-v3 numerical-loss diagnosis — frozen evidence

Cache-v3 smoke remains immutable at `/workspace/functional_cache_v3_smoke` and must not be overwritten or relabeled as lossless. Its manifest SHA-256 is `d56e79c291e238ec6f694070a6f4623c34d7ef72d92740b45bd25a76d42e64f4`.

## Finding

Cache-v3 is prefix-stable and structurally valid, but its BF16-to-FP16 state conversion is numerically lossy for an exact teacher-equivalence claim. Across all 1,265 represented teacher prediction positions, Phase 13C observed:

- hidden maximum absolute error: `0.0024013519287109375`;
- mean logit absolute error: `0.005615674883951783`;
- maximum logit absolute error: `0.125`;
- KL/token: `6.771036184470173e-05`; and
- next-token argmax agreement: `1264/1265`.

The one disagreement was example 2, sequence length 270, prediction position 180. Live BF16 logits tied tokens 661 and 2127 exactly; FP16 cache replay selected token 2127 instead of live `argmax` token 661.

The diagnostic proved that every cache-v3 FP16 state equals the corresponding live BF16 state rounded to FP16, while losslessly replaying the original BF16 state restores live coda logits exactly for all eight items. Therefore the failure is cache quantization amplified by the frozen coda, not an Attention² result and not an evaluator/coda-semantics failure.

## Frozen artifacts

- Pre-run failure: `phase13_prerun_zero_training_numerical_v1.json`, SHA-256 `a4d1f24eb727905ff54031f8a8f39236cc4fe209c7d0c91c62c7ded5a6df25ba`.
- Final diagnostic: `phase13_cache_quantization_diagnostic_v3.json`, SHA-256 `dcf283c53c87813a9bbfc797d01e5c6b5ce726f709912e83c0afa57f92f10ae9`.

Thresholds were not changed. Phase 13 was not executed. Cache-v4 smoke is a new object that preserves raw BF16 bits; cache-v3 remains historical immutable evidence.
