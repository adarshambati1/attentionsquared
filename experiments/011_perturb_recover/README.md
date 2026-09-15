# Experiment 011 — Perturb-and-recover attractor test

Frozen Huginn trajectories are perturbed at raw recurrent states h4 and h8 with token-relative isotropic Gaussian directions at fixed magnitudes 1%, 5%, and 10%. Three deterministic perturbation directions are used per held-out example; direction is shared across magnitudes within each example/depth/seed.

The unchanged recurrent operator resumes through D32. The primary metric is error relative to the injected perturbation, `E_r=||h_tilde_(k+r)-h_(k+r)||/||h_tilde_k-h_k||`; per-step contraction, cosine to the paired clean state, and perturbed-state recurrent residual are secondary diagnostics. Frozen-coda next-token agreement is measured at D16.

After the complete latent audit, full GSM8K generation runs the nine prioritized h8 conditions (three strengths by three perturbation seeds), resuming ordinary recurrence to D16 at every decoding step. No training, damping, correction, architecture change, or coefficient selection is permitted. Existing clean D16 generation is a frozen paired reference.
