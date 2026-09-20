# RecurTrace Step-3D reference audit

## Authority found

- Paper: arXiv `2609.03379v2`, anonymous AAAI submission.
- Downloaded source archive SHA-256: `9441560a7eb150e58a1ecc1a33bcd2a87a8cd0930ae81e49d7ba306726a0b812`.
- The paper source contains no official repository URL or commit.
- A GitHub repository search found no identifiable official implementation.

## Frozen facts from the paper

Source mapping: `AnonymousSubmission2027.tex` lines 152–205 defines the loop, reinjection, same-layer memory, attention, distance bias, and gates; lines 471–542 and Table `tab:app-hparams` define model scales and loop hyperparameters; lines 542–550 defines Stage-1 optimization, hardware, packing, and mixture; lines 599–610 clarifies that the query is `x_l^(t-1)`, QK normalization is per-head RMSNorm, and memory is same-position with window three. The source archive itself is not committed because it is third-party copyrighted material; its immutable arXiv identifier and downloaded archive digest are recorded above.

For Qwen3-1.7B-Base, the paper uses the 28-layer base release and zero-indexed loop layers 12–14. The base is frozen. Before each loop after the first, the block input is re-injected as `h <- h + alpha * RMSNorm(e)`, with scalar `alpha=0` initially. Before each repeated layer, Loop Memory Attention reads that same layer's outputs from prior loops. It uses same-token loop-time attention, window 3, four heads of dimension 128 (width 512), per-head QK RMS normalization, signed trainable ALiBi-initialized loop-distance slopes, scalar gate 1, and a token gate over `[query, mean(memory)]` initialized with output bias -3. No truncated backpropagation is used. Per the paper equation, the memory query is the newest prior-loop output of that same layer, `x_l^(t-1)`; it is not the current pre-layer block residual `h`.

Stage 1 samples depths 1–8 from a clamped lognormal-Poisson distribution with target mean 4 and log sigma 0.5. For 0.6B/1.7B the paper reports 15 epochs, approximately 25,000 optimizer steps after packing, global batch 128 packed sequences, max length 1,024, BF16, FlashAttention-2, AdamW, cosine schedule, peak LR `1e-4`, eight H800 80GB GPUs, and ZeRO-2. The data are an approximately 1.15M-example deduplicated direct-answer mixture. The controlled results use three training seeds. The MathQA adaptive study uses eight non-overlapping 300-item evaluation seeds; a full-test single-seed result is also reported.

## Facts not recoverable from the paper

The paper does not provide:

- official repository code or commit;
- exact dataset files, revisions, mixture weights, and final post-dedup manifest;
- exact prompt serialization/chat template;
- exact sequence-packing implementation;
- Stage-1 AdamW weight decay and warmup details;
- the three training seed values;
- the eight MathQA evaluation seed values and item mappings;
- an executable reference for cache semantics during generation.

Therefore an exact reproduction is not currently authorized. Any run without these assets must be explicitly user-approved and labeled **paper-guided controlled replication**.

## Approved five-arm causal design

1. `plain`: frozen plain loop, no memory module.
2. `shared_current`: one whole-block module, singleton current-state memory only.
3. `shared_history`: the same module and initialization with whole-block prior-loop history.
4. `per_layer_current`: one paper-capacity LMA module per repeated layer, but singleton current-state memory only.
5. `recurtrace`: the same per-layer modules and initialization with actual same-layer prior-loop history.

The matched causal contrasts are `shared_history - shared_current` and `recurtrace - per_layer_current`.

At T=2 each memory has one slot, so each matched pair is algebraically identical before training when corresponding parameters match. T=2 remains the primary published-budget evaluation. T=4 is the approved secondary diagnostic where multiple memory slots activate Q/K selection and loop-distance bias.

## Execution boundary

Architecture implementation and architecture correctness gates are approved. Stage-1 training and result evaluation remain fail-closed until an immutable authorization manifest names either authoritative exact assets or a user-approved paper-guided substitute, and sufficient compute is provisioned. Step 4 remains blocked.
