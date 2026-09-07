# Experiment 001 cap sensitivity (H100, GSM8K N=20)

All runs use the same 20 fixed GSM8K test examples, pinned Huginn revision,
greedy decoding, bfloat16, native chat template, and stop strings
`<|end_text|>` / `<|end_turn|>`. The corrected 256 run was rerun with the
same current pipeline as 512 and 1024.

| Cap | Depth | Accuracy | Cap-hit % | Degenerate/repetitive | Coherent but unfinished | Stop-string/evaluator issue | Other | Natural end % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 4 | 5% | 35% | 7 | 0 | 0 | 0 | 65% |
| 256 | 8 | 5% | 30% | 4 | 1 | 1 | 0 | 70% |
| 256 | 16 | 30% | 20% | 2 | 2 | 0 | 0 | 80% |
| 256 | 32 | 30% | 40% | 5 | 3 | 0 | 0 | 60% |
| 256 | 64 | 30% | 50% | 8 | 2 | 0 | 0 | 50% |
| 512 | 4 | 0% | 40% | 8 | 0 | 0 | 0 | 60% |
| 512 | 8 | 10% | 0% | 0 | 0 | 0 | 0 | 100% |
| 512 | 16 | 40% | 10% | 2 | 0 | 0 | 0 | 90% |
| 512 | 32 | 35% | 0% | 0 | 0 | 0 | 0 | 100% |
| 512 | 64 | 35% | 0% | 0 | 0 | 0 | 0 | 100% |
| 1024 | 4 | 0% | 45% | 8 | 1 | 0 | 0 | 55% |
| 1024 | 8 | 10% | 5% | 1 | 0 | 0 | 0 | 95% |
| 1024 | 16 | 25% | 0% | 0 | 0 | 0 | 0 | 100% |
| 1024 | 32 | 35% | 0% | 0 | 0 | 0 | 0 | 100% |
| 1024 | 64 | 35% | 0% | 0 | 0 | 0 | 0 | 100% |

## Analysis

- Increasing the cap materially reduces truncation for depths 8–64. At 512,
  only D=16 retains 2/20 cap hits; at 1024, D=8 has 1/20 and D=16+
  have none.
- D=4 remains degenerate/repetitive at roughly 35–45% regardless of cap. A
  larger cap does not rescue those generations; they are not merely coherent
  answers needing more tokens.
- No systematic stop-string/evaluator failure was found. The one 256/D=8
  classification is an isolated output-level issue, not a trend.
- The qualitative depth trend is stable: D=4/8 are weak, D=16 improves,
  and D=32/64 are best in this sample. The exact D=16 accuracy varies with
  cap, but the conclusion does not reverse.
- TTFT and fixed single-forward latency increase monotonically with depth.

## Decision

Cap choice does not change the qualitative interpretation of depth scaling.
Use **1024** for the final N=100 run because it leaves essentially no cap
pressure at D>=16 and keeps the low-depth degeneracy visible rather than
confounding it with truncation. Experiment 001 remains a Huginn baseline;
this does not establish Attention² or explain the recurrence mechanism.
