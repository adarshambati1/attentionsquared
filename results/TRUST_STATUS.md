# Experimental trust status

This file is the authoritative trust ledger for existing Attention Squared
artifacts. No artifact may be deleted or overwritten merely because it is
listed as provisional or invalid.

| Artifact | Status | Interpretation |
|---|---|---|
|
| Experiment 001 | **Trusted** | Validated Huginn depth/latency baseline. D16 reaches 42% on the N=100 baseline and 40.4% on the later paired 250-example teacher control. |
| Experiment 002 | **Useful, incomplete** | Final-token trajectory geometry is useful. Sequence-mean analysis and long-range cosine at strides 1, 2, 4, 8, and 16 remain unfinished. |
| Experiment 003 downstream | **Useful** | Direct and residual tokenwise jumps scored 0% while the same-split Huginn D16 teacher scored 40.4%. |
| Experiment 003 KL | **Needs repair** | Historical KL values used `batchmean` and sum over sequence positions; recompute as mean KL per valid token. |
| Experiment 004A | **Mechanical/exploratory evidence** | Shape, causal path, depth communication, gradients, and tiny overfit were demonstrated. The compressed-trajectory observation is preserved. It is not downstream evidence. |
| Trajectory-trained 004B | **Exploratory, seed-confounded** | K models used different parameter initializations, so K was not the sole manipulated variable. |
| Trajectory-trained 004C | **Invalid/provisional downstream result** | Generation invoked A² incrementally on one token without an A² KV cache although training used full sequences. Re-evaluate with full-prefix execution. Historical KL is also not token-normalized. |
| Functional continuation cache | **Preserved source data** | Teacher continuations are useful, subject to corrected/manual `valid_end` review for all 27 capped examples. |
| Functional full-logit cache v1 | **Invalid target alignment** | Teacher logits used `[answer_start:valid_end]` instead of `[answer_start-1:valid_end-1]`. Preserve but never train from it again. |
| Functional models v1 | **Invalid** | Trained against off-by-one targets and selected on only 16 validation examples. |
| Functional evaluation v1 | **Invalid** | Combines invalid models with mismatched incremental A² execution, unpaired random `h0`, inconsistent cap scoring, unsynchronized timing, and non-token-normalized KL. |

## Governing rule

No Attention² quality or speed claim may use an artifact marked incomplete,
provisional, seed-confounded, or invalid. Existing artifacts remain preserved
for auditability.
