# Phase 9R cache-v3 smoke pre-run review — BLOCKED

Review job: `08c1b7ef`

## Verdict

- Code: **FAIL**
- Science: **FAIL**
- Cache construction: **not started**
- Training: **not started**

## Blocking findings

1. Storage preflight estimated bytes but did not fail closed on actual free
   space, write/fsync capability, or atomic no-replace rename support.
2. The validator reused builder schedule/extraction helpers rather than an
   independent numerical implementation.
3. The validator made one schedule per example and therefore did not test
   independently rematerialized fixed-shape prompt/full requests.
4. Validation did not independently reconstruct the pinned dataset prompt and
   prove exact token identity with the raw sequence prefix and reviewed bounds.
5. Active autoregressive evaluators still called the old shape-dependent
   `paired_huginn_h0` path instead of creating one fixed 2048-position schedule
   per generation and slicing it.
6. Remaining prefix/full prelude `x` differences require propagated `h16`,
   logit-error, KL, and argmax diagnostics. The smoke can prove elimination of
   shape-dependent `h0`; it cannot by itself prove behavioral repair.

All findings must be remediated and independently reviewed before cache
construction. Cache v2, raw continuations, and all prior artifacts remain
unchanged.
