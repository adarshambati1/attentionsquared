# Phase 13 pre-run review — BLOCKED

Review job: `50629db0`

## Verdict

- Code: **FAIL**
- Science: **FAIL**
- Phase 13 execution: **not started**

## Blocking findings

1. The Phase 11/12 functional path applies `ln_f` to cached `h16_teacher`, even
   though cache v2 defines that tensor as Huginn's already-normalized recurrent
   output immediately before the frozen coda. Equality to
   `model(input_states=h16_teacher, num_steps=0)` only proves both paths repeat
   the same extra normalization; it does not establish equality to normal live
   D16 logits.
2. The proposed 13B routes both reduce to the same whole-model forward and are
   not independent implementations.
3. Contiguous cache IDs 2250–2257 cover median-length continuations but omit the
   short and long tails. The science reviewer proposed length-stratified IDs
   `[2277, 2473, 2452, 2485, 2415, 2264, 2317, 2439]`, with valid continuation
   lengths `[10, 108, 130, 151, 178, 205, 250, 826]`.

## Consequence

Existing Phase 11 remains evidence that Attention² can optimize the surrogate
function that was actually supplied. It is not evidence that Attention² matched
normal live Huginn D16. Existing Phase 12 output remains preserved but is not
authoritative evidence about live-Huginn-equivalent free-running behavior.

## Required remediation before Phase 13

Establish and review the authoritative coda interface:

`pre-ln_f recurrent state -> ln_f -> frozen coda -> final ln_f -> lm_head`.

Because cache v2 stores the already-normalized coda input, cached and predicted
`h16` must enter the frozen coda without another pre-coda `ln_f`. Validate the
decomposed route against a normal live D16 forward, then rerun corrected Phase
10, Phase 11 from the unchanged shared initialization, and Phase 12 into new v3
paths. Preserve all v2 artifacts unchanged.
