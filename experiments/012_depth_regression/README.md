# Experiment 012 — Depth regression / overthinking audit

Step 2 of 11. Frozen plain Huginn is evaluated on the same 250 held-out GSM8K examples at recurrent depths 4, 8, 16, 32, 64, 128, 256, and 512. Prompting, tokenizer/model revisions, deterministic per-example h0, BF16 execution, stopping, 1,024-token cap, and scorer remain fixed.

Each immutable per-example record contains generated text/tokens, extracted answer, correctness, length, cap and repetition status, synchronized latency, peak memory, and the prompt-state fixed-point residual `||F(h_D,x)-h_D||/||h_D||`. Adjacent-depth paired correctness transitions, reconstructable JSON/CSV, one summary table, and four prescribed plots are published after all depths complete.

No training or architecture changes are permitted. Step 3 is not launched automatically.
