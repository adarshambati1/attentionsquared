# Experiment 005 — Latent-history attention

This experiment keeps frozen Huginn's prelude, recurrent core, token attention, coda, and head, and inserts one shared per-token attention module over completed recurrent-loop states.

At D=8, before core loop `k`, each token attends only across that token's states `[h0, ..., hk]`. The sole trainable module has Q/K/V width 256, four heads, and a zero-initialized output projection. There is no extra gate, FFN, slot workspace, trajectory loss, or distillation.

Training uses GSM8K-train IDs 0–2249 and the dataset-provided solutions. Cross-entropy is computed only for shifted solution tokens, including the first answer-token prediction. IDs 2250–2499 select the checkpoint by validation answer-token loss. GSM8K-test IDs 0–249 are used only for the final comparison.

The frozen first run is 1,000 AdamW updates with eight single-example gradient-accumulation passes per update. A mechanical gate at step 10 continues the same run if finite loss, checkpoint writing, and cached generation work. Full validation and checkpoints occur every 100 updates.

Final comparison:

1. Original Huginn D=8
2. Original Huginn D=16
3. History-attention Huginn D=8

Report accuracy, cached-generation latency, generated-token count, cap rate, peak memory, and synchronized fixed-256-token forward latency. Run current-only and uniform-history controls sequentially only if the trained history model improves over untouched D=8.
