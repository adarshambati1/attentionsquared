# Experiment 012 — depth regression audit

| Depth | Correct | Accuracy | Mean residual | Cap rate | Degeneration |
|---:|---:|---:|---:|---:|---:|
| 4 | 3/250 | 1.2% | 0.446493 | 40.0% | 7.2% |
| 8 | 40/250 | 16.0% | 0.201567 | 8.0% | 0.0% |
| 16 | 97/250 | 38.8% | 0.0574168 | 0.8% | 0.0% |
| 32 | 108/250 | 43.2% | 0.0109623 | 2.0% | 0.0% |
| 64 | 111/250 | 44.4% | 0.00863004 | 2.4% | 0.0% |
| 128 | 108/250 | 43.2% | 0.00858544 | 3.6% | 0.0% |
| 256 | 110/250 | 44.0% | 0.00856384 | 2.4% | 0.0% |
| 512 | 109/250 | 43.6% | 0.00856044 | 2.0% | 0.0% |

Adjacent-depth paired transitions are in `transitions.json`. Complete reconstructable outputs are in `raw_results.json` and `raw_results.csv`; immutable per-example records remain under `/workspace/depth_regression_audit`.

Run commit: `2786836c01bbffe033834c11de1cc698a0d0f97b`.

Step 3 was not started.
