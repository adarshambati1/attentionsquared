# Attention Squared

Research project investigating how language models should use iterative latent computation: ordinary recurrence, loop-level skips, explicitly supervised refinement, monotonic refinement, and looped-flow training.

## Research question

Should iterative latent reasoning repeat a shared operator, explicitly supervise progressive refinement, enforce monotonic progress, or learn a local flow/transport process?

The frozen 11-step execution plan is in [`ROADMAP.md`](ROADMAP.md). Step 3 is current; Step 4 and all later steps remain blocked until Step 3 is frozen.

## Experimental status

- [Frozen 11-step roadmap](ROADMAP.md)
- [Trust ledger](results/TRUST_STATUS.md)
- [Authoritative Experiment 004 recovery plan](experiments/004a_attention2/RECOVERY_PLAN.md)

The functional-v1 cache, models, and evaluation are preserved but invalid for
scientific inference. No further Experiment 004 training should begin until the
recovery plan's correctness gates pass.

## Repository layout

- `src/` — active reusable model, evaluation, and analysis code
- `scripts/` — active experiment entry points
- `scripts/archive/` — preserved historical entry points; not for new runs
- `tests/` — correctness and regression tests
- `results/` — preserved scientific artifacts and lightweight summaries

See [`scripts/archive/README.md`](scripts/archive/README.md) before using an archived script.
