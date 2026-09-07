# Experiment 001 final N=100 baseline

Hardware: NVIDIA H100 80GB HBM3, US-MO-1, driver 580.126.09.

This is the final 100-example Huginn depth-scaling run: the first 100 fixed GSM8K test examples at depths `{4, 8, 16, 32, 64}`, greedy bfloat16 generation, pinned model/dataset revisions, native Huginn chat template, and a 1024-token cap with Huginn stop strings.

Raw JSONL is intentionally excluded from Git. Summary CSV and plots are committed.
