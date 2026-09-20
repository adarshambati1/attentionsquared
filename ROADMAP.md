# Frozen 11-step research roadmap

**Status:** frozen planning document. Step 3 is current. Step 4 and all later steps remain blocked until Step 3 is frozen.

This roadmap supersedes earlier main-paper framings centered on parallel-slot A² or generic “native diffusion.” Historical experiments, recovery plans, and artifacts remain preserved under their original names and must not be relabeled.

## Prior-art framing

- **Thinking with Looped Flows** already demonstrates the core loop-plus-flow-matching/local-denoising idea: recurrent hidden memory, local denoising objectives, and probability-flow inference on structured reasoning tasks. Its reported results include 58.8% on ARC-AGI-1 and 12.2% on ARC-AGI-2.
- Therefore, Step 8 must not claim to invent native flow refinement. Its open contribution is a controlled **language-model adaptation** capable of natural-language output.
- **ELF** provides evidence that continuous flow matching can generate language in embedding space, but it is not a Huginn-style recurrent reasoner using looped-flow computation.
- Monotonic refinement is prior art (for example, RecursiveVLM-style monotonic recursion losses) and is treated as a baseline/extension, not as unprecedented.
- Literature claims and exact citations must be verified against primary sources before paper submission.

## 1. Perturb-and-recover attractor test — complete and frozen

Main result:

\[
\boxed{\text{finite-amplitude correction, but not uniform infinitesimal contraction}}
\]

Huginn tolerates substantial perturbations functionally. Perturbations of 5–10% shrink strongly, while 1% perturbations largely persist. Do not add more perturbation experiments.

## 2. Depth-regression / overthinking audit — complete and frozen

| Depth | GSM8K |
|---:|---:|
| 4 | 1.2% |
| 8 | 16.0% |
| 16 | 38.8% |
| 32 | 43.2% |
| 64 | 44.4% |
| 128 | 43.2% |
| 256 | 44.0% |
| 512 | 43.6% |

Main result:

\[
\boxed{\text{large gains through D64, followed by a non-monotonic plateau}}
\]

Extra compute is not guaranteed to improve an answer. This motivates explicit refinement training later in the roadmap.

## 3. Robustness/generalization of Huginn mechanism findings — current

Hypothesis under stress test:

\[
\boxed{\text{A learned current-state transformation explains most of the gain attributed to explicit loop history}}
\]

### 3A. GSM8K paired-seed replication

Use three paired deterministic \(h_0\) seeds on all 250 examples:

- Plain Huginn D8
- Current-state adapter D8
- Projected-uniform history D8
- Shared learned history D8
- Per-layer/RecurTrace-memory D8
- Plain D16
- Plain D64

Report seed-wise accuracy, question-clustered confidence intervals, paired wins/losses/ties, cap and degeneration rates, length, latency, and memory.

Primary deltas:

\[
A_{\rm current}-A_{\rm plain},\quad
A_{\rm projected}-A_{\rm current},\quad
A_{\rm shared}-A_{\rm current},\quad
A_{\rm perlayer}-A_{\rm current}.
\]

### 3B. Cross-dataset transfer

Evaluate the same frozen Huginn modules on SVAMP and MATH-500 with paired seeds and no dataset-specific tuning:

- Plain D8
- Current adapter D8
- Shared history D8
- Plain D16
- Plain D64

Question: does the mechanism decomposition survive outside GSM8K?

### 3C. Depth transfer

Reuse unchanged D8-trained weights at \(D\in\{4,8,16,32,64\}\) for:

- Plain Huginn
- Current-state adapter
- Shared history

No depth-specific retraining.

### 3D. RecurTrace/Qwen3 mechanism replication

After provenance verification, use the pinned Qwen3-1.7B RecurTrace configuration with five conditions:

1. Plain looped Qwen3
2. Shared current-state adapter
3. Shared whole-loop history
4. Per-layer current-state control
5. RecurTrace per-layer Loop Memory Attention

Causal contrasts:

\[
\boxed{C-B=\text{shared historical access}},\qquad
\boxed{E-D=\text{per-layer historical access}}.
\]

Use fixed loop budgets, not adaptive halting. The primary published-style evaluation is \(T=2\); a preregistered \(T=4\) diagnostic activates multiple historical slots.

3A–3C may execute. 3D implementation and gates may execute. Full 3D training remains blocked until provenance and compute are sufficient; any substitute must be labeled a paper-guided controlled replication rather than an exact reproduction.

Freeze all of Step 3 before continuing.

## 4. Teacher-flow retrofit on Huginn

Ask whether learned continuous transport can replace Huginn’s late iterative computation. Use the empirical high-quality/near-fixed regime:

\[
\boxed{h_8\rightarrow h_{64}}
\]

Freeze Huginn. Train one rectified-flow vector field \(v_\theta(z,t,x)\) with:

\[
z_t=(1-t)h_8+t h_{64},\qquad u=h_{64}-h_8.
\]

At inference, start from real \(h_8\) and integrate using \(K\in\{1,2,4,8\}\) flow steps.

Compare Plain D8/D16/D32/D64 against Flow K1/K2/K4/K8. Report quality, endpoint fidelity, cross-entropy, FLOPs, fixed-forward latency, and generation latency.

Question:

\[
\boxed{\text{Can learned flow replace dozens of actual recurrent updates?}}
\]

This is teacher retrofit/compression, not a native flow architecture.

## 5. Ordinary native loop and loop-skip baselines

Freeze one common from-scratch pretraining specification for Steps 5–8: architecture size, tokenizer, corpus, training tokens, optimizer/LR schedule, context length, recurrent unit, \(K_{\rm train}=8\), and evaluation suite.

### 5A. Ordinary native loop

\[
z_{k+1}=F_\theta(z_k,x),\qquad L=L_8.
\]

Evaluate the same checkpoint at \(K\in\{1,2,4,8,16,32\}\).

### 5B. Adjacent whole-loop residual baseline

Use a standard/scaled inter-loop residual from the relevant prior-art formulation. This controls for gains from improved residual routing alone.

### 5C. Delayed \(k-2\) whole-loop skip

\[
\boxed{z_{k+1}=F_\theta(z_k,x)+\alpha z_{k-2}}
\]

Use only a tiny validation sweep over conservative \(\alpha\) values. Track accuracy, deep-loop stability, state norms, degeneration, and overthinking frequency.

Question: does bypassing a whole iteration preserve useful earlier computation better than ordinary recurrence? This is an architectural baseline, not monotonic refinement.

## 6. Original native refiner

Use the Step-5A architecture, but supervise multiple refinement depths:

\[
\mathcal K=\{1,2,4,8\},\qquad
\boxed{L_{\rm refine}=\frac{1}{|\mathcal K|}\sum_{k\in\mathcal K}L_k}.
\]

No halting head and no monotonic penalty. Evaluate at \(K=1,2,4,8,16,32\).

Primary questions:

- Is Refiner@K better than Ordinary Loop@K?
- Does training through K8 continue improving at K16/K32?

## 7. Monotonic-refinement baseline

Treat monotonic refinement as a strong prior-art baseline/extension. Starting from Step 6, add:

\[
L_{\rm mono}=\sum_k\max\left(0,L_{k+1}-\operatorname{sg}(L_k)\right),
\]

\[
L=L_{\rm refine}+\lambda_{\rm mono}L_{\rm mono}.
\]

Use only a small validation sweep for \(\lambda_{\rm mono}\). Compare the ordinary loop, original refiner, and monotonic refiner. Measure mean quality, \(P(\text{extra loop worsens loss})\), and \(P(\text{correct}\rightarrow\text{wrong})\).

Question:

\[
\boxed{\text{Can additional inference compute be made reliably productive?}}
\]

## 8. Looped Flows to language-model adaptation

Name this step explicitly; do not call it generic native diffusion and do not claim Looped Flows itself as novel.

Reproduce the Looped Flows principle in a language model that produces actual natural-language output:

- local denoising/flow objectives;
- recurrent hidden memory;
- shared temporal noise structure;
- probability-flow inference;
- variable numbers of refinement steps.

The open contribution is:

\[
\boxed{\text{Can looped-flow training work as a real language model producing natural-language output?}}
\]

Resolve explicitly:

- the flow state for language;
- whether it is an internal latent, answer representation, or both;
- its interaction with autoregressive token generation;
- whether more flow steps improve reasoning;
- whether it beats ordinary looping/refinement at matched compute.

Controlled comparison:

\[
\boxed{\text{ordinary loop}\quad\text{vs}\quad\text{original refiner}\quad\text{vs}\quad\text{monotonic refiner}\quad\text{vs}\quad\text{Looped-Flows LM}}
\]

Match model scale, data, training compute, and inference compute. A stochastic diffusion/noise objective may be a secondary arm using the same clean latent definition; do not turn Step 8 into an architecture zoo.

Step 4 remains distinct: it compresses an already-trained Huginn trajectory. Step 8 asks whether a language model should be trained natively with looped-flow dynamics.

## 9. Final native-model robustness

Take only competitive native models and test:

- multiple training seeds;
- multiple reasoning datasets;
- at least one non-math domain if feasible;
- \(K=1,2,4,8,16,32\);
- trained-depth and extrapolated-depth regimes;
- equal parameters, training tokens, inference loops, FLOPs, and wall-clock latency.

Report quality versus K/FLOPs/latency, \(P(\text{extra loop hurts})\), and depth extrapolation. If budget permits, scale only the winning training rule to one larger model.

## 10. Freeze results and write the paper

Working title:

\[
\boxed{\textbf{How Should Language Models Use Iterative Latent Compute?}}
\]

### Part I: what recurrent depth actually does

Use Huginn evidence on convergence/fixed-point-like geometry, finite-amplitude perturbation recovery, D4–D512 depth behavior, current-state adaptation versus history, RecurTrace/Qwen replication, failures of raw means/classical acceleration/one-shot shortcuts, and the teacher-flow retrofit.

Only claim what Steps 3 and 4 support. The likely mechanistic message is that recurrent depth behaves like iterative latent computation, but useful progress is not captured by naive state arithmetic, generic history retrieval, or a small direct shortcut.

### Part II: how iterative computation should be trained

Controlled native comparison:

- ordinary loop;
- delayed loop skips;
- original refiner;
- monotonic refiner;
- Looped-Flows language model.

Central question: should iterative latent reasoning repeat a shared operator, supervise progressive refinement, enforce monotonic progress, or learn a local flow/transport process?

Keep old parallel-slot A²/distillation detours out of the main paper.

## 11. Reproducibility and public release

Freeze commits, exact configs, model/tokenizer revisions, dataset/corpus versions, train/validation/test splits, seeds and paired-seed maps, raw outputs, all paper checkpoints, scorers, synchronized timing harnesses, environment lockfiles, manifests/SHA-256s, and scripts regenerating every table and figure.

Create a traceable chain:

\[
\text{paper figure/table}\rightarrow\text{experiment}\rightarrow\text{config}\rightarrow\text{checkpoint}\rightarrow\text{raw artifact}\rightarrow\text{summary}\rightarrow\text{git commit}\rightarrow\text{SHA-256}.
\]

Clearly distinguish valid final runs, negative results, exploratory runs, superseded runs, and invalid runs.

## Frozen execution order

\[
\boxed{
\begin{aligned}
1.&\ \text{Perturb/recover}\quad\checkmark\\
2.&\ \text{Depth audit D4--D512}\quad\checkmark\\
3.&\ \mathbf{Huginn + RecurTrace robustness}\quad\leftarrow\mathbf{current}\\
4.&\ \text{Teacher flow }h_8\rightarrow h_{64}\\
5.&\ \text{Ordinary loop + adjacent residual + delayed }k-2\text{ skip}\\
6.&\ \mathbf{Original native refiner}\\
7.&\ \text{Monotonic-refinement baseline}\\
8.&\ \text{Looped-Flows language model}\\
9.&\ \text{Final native robustness}\\
10.&\ \text{Paper}\\
11.&\ \text{Reproducibility/release}
\end{aligned}}
\]
