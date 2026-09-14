# Experiment 008 conclusion

The correctness gate passed before evaluation. No parameters were trained or added, plain mode exactly reproduced frozen Huginn D8, modified states remained finite, Huginn weights were unchanged, and cached/full-prefix next-token argmax checks passed.

Validation selected heavy-ball `mu=0.05` (59/250, 23.6%) and Anderson `m=2` (47/250, 18.8%). Held-out GSM8K results were:

| Method | Accuracy | Cap hit | Repetition degeneration | Mean tokens | Mean latency | Fixed 256-token latency |
|---|---:|---:|---:|---:|---:|---:|
| Plain D8 | 40/250 (16.0%) | 8.0% | 0.0% | 240.224 | 11.508 s | 55.654 ms |
| Halpern | 40/250 (16.0%) | 4.8% | 1.2% | 224.512 | 10.715 s | 55.761 ms |
| Heavy-ball | 38/250 (15.2%) | 4.4% | 0.0% | 210.096 | 10.005 s | 55.777 ms |
| Anderson | 41/250 (16.4%) | 6.8% | 0.0% | 231.408 | 11.308 s | 57.143 ms |

Halpern and heavy-ball did not improve held-out accuracy, and no arm approached the frozen D16 reference of 38.8%. Heavy-ball's validation gain did not transfer to test.

The Anderson result is fallback-dominated and must not be interpreted as an effective test of successful Anderson mixing: all three validation windows produced the same 47/250 result and 523,943 fallback token-steps, while the selected test arm recorded 558,866 fallback token-steps. It is therefore evidence about this preregistered implementation/protocol, not clean evidence that numerically successful Anderson acceleration cannot help.

Overall, this run found no useful parameter-free classical acceleration. It does not alter the prior finding that the learned current-state adapter supplied most of the observed D8 history-module gain.
