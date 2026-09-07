# H100 N=20 validation

Hardware: NVIDIA H100 80GB HBM3, US-MO-1, driver 580.126.09. Listed secure Pod rate: $3.49/hour.

This is 20 fixed GSM8K test examples evaluated at depths `{4, 8, 16, 32, 64}` (100 generations total). It uses the pinned Huginn revision and the upstream-compatible greedy 256-token generation budget. Cap hits, natural termination, TTFT, fixed single-forward latency, throughput, and Wilson 95% intervals are recorded.

Latency is not directly comparable with the A100 smoke run because the hardware differs. This is an intermediate validation checkpoint, not yet the N=100 result.
