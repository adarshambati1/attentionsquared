# Experiment 001 conclusion: Huginn recurrent-depth baseline

## Final protocol

The final run evaluated the first 100 fixed GSM8K test examples at
`D={4,8,16,32,64}` on one NVIDIA H100 80GB HBM3 using the pinned Huginn
checkpoint, bfloat16, greedy decoding, the native chat template, a 1024-token
cap, and `<|end_text|>`/`<|end_turn|>` stop strings. The cap was selected by
the preceding N=20 sensitivity study. All depths used identical examples and
settings.

## Final result

| Depth | Accuracy | 95% Wilson CI | Mean latency | Single forward | Cap hits |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.00 | [0.00, 0.04] | 8.15s | 0.055s | 35/100 |
| 8 | 0.15 | [0.09, 0.23] | 7.28s | 0.097s | 9/100 |
| 16 | 0.42 | [0.33, 0.52] | 11.11s | 0.180s | 2/100 |
| 32 | 0.39 | [0.30, 0.49] | 24.10s | 0.346s | 4/100 |
| 64 | 0.42 | [0.33, 0.52] | 43.39s | 0.678s | 2/100 |

## Conclusion

This setup reproduces the qualitative Huginn depth-scaling behavior: useful
capability appears by D=16, while D=32 and D=64 plateau rather than materially
exceeding D=16 on this sample. Fixed single-forward latency scales close to
linearly with recurrent depth, providing the clean sequential-cost baseline.
The D=4 result is weak and remains cap/degeneracy-prone even at 1024 tokens;
D>=16 has low cap-hit rates.

This is a trustworthy baseline for the research project, not evidence for
Attention². Experiment 001 is complete. The next experiment should measure
recurrent hidden-state trajectories; no trajectory collection, training, or
Attention² implementation was performed here.
