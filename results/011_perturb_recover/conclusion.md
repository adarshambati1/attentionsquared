# Experiment 011 conclusion

Frozen Huginn was perturbed at h4 and h8 with token-relative isotropic noise at 1%, 5%, and 10%, using three deterministic directions per held-out example. All 4,500 latent conditions and all 4,500 functional generations completed. No parameters were trained and ordinary recurrence resumed unchanged.

| Perturb | Noise | E at D16 | E at D32 | D16 cosine | GSM8K |
|---|---:|---:|---:|---:|---:|
| h4 | 1% | 0.9965 | 1.0056 | 0.999950 | 303/750 (40.4%) |
| h4 | 5% | 0.2316 | 0.2021 | 0.999933 | 305/750 (40.7%) |
| h4 | 10% | 0.1539 | 0.1025 | 0.999880 | 287/750 (38.3%) |
| h8 | 1% | 0.9783 | 1.0065 | 0.999952 | 294/750 (39.2%) |
| h8 | 5% | 0.2844 | 0.2036 | 0.999898 | 285/750 (38.0%) |
| h8 | 10% | 0.2285 | 0.1057 | 0.999736 | 299/750 (39.9%) |

The clean D16 reference is 97/250 (38.8%); each table row pools three perturbation seeds, so its clean paired denominator is effectively repeated three times. Functional accuracy remained near clean D16 in every condition, with cap rates of 0.9–1.3% and only one detected degeneration among all h4 generations and none among h8 generations.

The mechanistic result is scale-dependent. One-percent perturbations were not actively removed: normalized error remained near one and slightly exceeded one by D32. Five- and ten-percent perturbations contracted strongly, reaching approximately 0.20 and 0.10 of their injected magnitude by D32. Recovery was not stronger from h8 than h4.

Therefore the data support robust finite-amplitude recovery around the normal trajectory, but not a uniform locally contractive or infinitesimal-attractor claim. Huginn's task behavior is highly robust to these perturbations, while the normalized latent dynamics show a nonlinear scale-dependent basin rather than simple monotone local contraction.
