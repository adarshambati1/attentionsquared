# Attention Squared

Research project investigating whether recurrent latent depth in transformers can be replaced by substantially less sequential computation using parallel attention over computational depth.

## Research question

Can a learned parallel depth operator match the capability gains of recurrent-depth transformers while reducing the sequential critical path?

## Experimental status

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
