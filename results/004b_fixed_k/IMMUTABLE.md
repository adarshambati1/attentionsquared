# Frozen 4B cache

The trajectory cache under the Pod workspace is immutable and shared by every
fixed-K condition. It contains 2,500 exact Huginn teacher examples split
2,000/250/250, with `h0` and `x` shaped `[T,5280]` and `trajectory` shaped
`[16,T,5280]`, stored as float16 from a bfloat16 teacher capture.

Do not regenerate K-specific caches or modify this cache after training begins.
See `cache_manifest.json` for revisions, extraction convention, and quantization
checks.
