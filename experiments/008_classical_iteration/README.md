# Experiment 008 — Classical D8 iteration rules

Frozen Huginn D8 is evaluated with no new parameters under four recurrence rules: plain, canonical Halpern (`alpha_k=1/(k+2)`), heavy-ball, and regularized Anderson acceleration.

Heavy-ball selects `mu` from `{0.05, 0.1, 0.2}` on GSM8K-train validation IDs 2250–2499. Anderson selects window `m` from `{2,3,4}` on the same validation IDs with ridge fixed at `1e-4`. Selection is highest corrected validation accuracy, then fewer cap hits, then shorter generations, then the smaller coefficient/window. The held-out GSM8K test IDs 0–249 are evaluated once after selection.

Anderson uses an independent constrained regularized least-squares solve at each token position over the hidden-axis residuals, preserving causal and incremental generation semantics. Singular or non-finite token solves fall back to the ordinary Huginn update and are counted.

The run reports accuracy, cap and repetition degeneration rates, generation length and latency, synchronized 256-token forward timing, numerical fallbacks, relative update norms, and cosine to the final D8 state. No training, adapters, architecture changes, or coefficient sweeps beyond the frozen sets are permitted.
