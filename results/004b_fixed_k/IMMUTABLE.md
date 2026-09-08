# Frozen 4B cache

The trajectory cache under the Pod workspace is immutable and shared by every
fixed-K condition. It contains 2,500 exact Huginn teacher examples split
2,000/250/250, with `h0` and `x` shaped `[T,5280]` and `trajectory` shaped
`[16,T,5280]`, stored as float16 from a bfloat16 teacher capture.

Do not regenerate K-specific caches or modify this cache after training begins.
See `cache_manifest.json` for revisions, extraction convention, and quantization
checks.

## Interpretation status

The cache remains valid and immutable. The completed trajectory-trained models
are exploratory controls, but their K comparison is seed-confounded because
K-specific runs used different initial random seeds. Any historical downstream
4C result produced by one-token Attention² execution is invalid/provisional and
must be rerun with the full-prefix evaluator specified in the Experiment 004
recovery plan.
