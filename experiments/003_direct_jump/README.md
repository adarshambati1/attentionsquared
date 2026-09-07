# Experiment 003: Direct-jump distillation

This experiment tests the strongest deliberately boring shortcut baseline:
can a tokenwise MLP predict Huginn's D=16 recurrent state from `(h0, x)` in one
shot? It excludes recurrence, token attention, depth slots, skip connections,
and Attention² mechanisms.

The frozen result is summarized in `results/003_direct_jump/conclusion.md`.
