# Experiment 005 — D8 latent-history attention result

## Outcome

The selected D8 latent-history checkpoint improved over plain Huginn D8, but did not match Huginn D16.

| Model | Accuracy | Mean generation latency | Fixed 256-token forward |
|---|---:|---:|---:|
| Plain Huginn D8 | 40/250 (16.0%) | 11.313 s | 55.231 ms |
| Plain Huginn D16 | 97/250 (38.8%) | 17.212 s | 103.357 ms |
| Latent-history Huginn D8 | 58/250 (23.2%) | 6.142 s | 56.400 ms |

Latent-history D8 gained 7.2 accuracy points over plain D8 with 2.1% fixed-forward overhead. It remained 15.6 points below D16 while its fixed-length forward was 45.4% faster than D16. The lower end-to-end generation latency is partly affected by shorter generations and must not replace the fixed-length timing comparison.

## Training

Only the shared 256-wide, four-head per-token history-attention module was trained. Huginn remained frozen. The output projection alone was zero-initialized. Training used next-token cross-entropy on GSM8K-provided answer tokens, including the first answer-token prediction.

Validation loss reached its minimum at step 500 (`0.43058749494209053`) and worsened to `0.47797876856306987` by step 1000. The step-500 checkpoint was selected prospectively by full validation answer-token cross-entropy; GSM8K test was not used for selection.

## Correctness

The pre-training gate established exact zero-effect logits and latents, nine states for D8 (`h0` through `h8`), no Huginn gradients or parameter changes, finite gradients through frozen Huginn into the new module, and loop-aware cached/full-prefix argmax agreement. Cached/full-prefix maximum logit error after an ephemeral nonzero update was at most `0.09375` on the checked prefixes.

## Scope

This result answers the primary comparison. The preregistered current-state-only and uniform-history attribution controls have not been run. No claim is made yet that historical access, rather than added trainable processing or simple history mixing, caused the improvement.

The complete checkpoint set and logs are frozen on the durable volume at `/workspace/latent_history_attention_d8`. The authoritative checkpoint is `best.pt`, SHA-256 `afff645730c278411bd18ce5cb231c2768e4a038ab87308be01d4764ed250b6d`.
