# Experiment 012 — Depth regression / overthinking audit

Step 2 of 11. Frozen plain Huginn is evaluated on the same 250 held-out GSM8K examples at recurrent depths 4, 8, 16, 32, 64, 128, 256, and 512. Prompting, tokenizer/model revisions, deterministic per-example h0, BF16 execution, stopping, 1,024-token cap, and scorer remain fixed.

Generation uses the canonical dynamic cache while allocated GPU memory is below the preregistered 65 GiB bound. If that bound is reached, it switches to mathematically equivalent no-cache full-prefix evaluation; the correctness gate requires the cached and full-prefix next token to agree exactly at D512. This preserves the 1,024-token cap without allowing the recurrent-step-indexed KV cache to exceed H100 memory.

Each immutable per-example record contains generated text/tokens, extracted answer, correctness, length, cap and repetition status, synchronized latency, peak memory, and the prompt-state fixed-point residual `||F(h_D,x)-h_D||/||h_D||`. Adjacent-depth paired correctness transitions, reconstructable JSON/CSV, one summary table, and four prescribed plots are published after all depths complete.

No training or architecture changes are permitted. Step 3 is not launched automatically.
