# Corrected FP32 Anderson amendment

The original Anderson arm is invalid because CUDA BF16 autocast caused every unsupported linear solve to enter its fallback path. Its artifacts remain preserved in the parent Experiment-008 directory.

This amendment forced Gram/KKT construction and `torch.linalg.solve_ex` outside autocast in FP32, handled failures independently per token, made implementation exceptions fatal, and required every full-prefix, prefill, incremental-cache, and uncached correctness-gate solve to succeed. The corrected gate passed.

Validation results were:

| Window | Accuracy | Successful solves | Fallbacks |
|---|---:|---:|---:|
| m=2 | 34/250 (13.6%) | 572,474/572,474 | 0 |
| m=3 | 31/250 (12.4%) | 594,839/594,839 | 0 |
| m=4 | 18/250 (7.2%) | 632,527/632,527 | 0 |

Validation selected `m=2`. Its held-out result was **34/250 (13.6%)**, with 607,922/607,922 successful solves, zero fallbacks, zero singular solves, and zero non-finite solves. It had a 10.4% cap-hit rate, 2.0% repetition-degeneration rate, 259.44 mean generated tokens, 13.052 s mean generation latency, and 58.068 ms fixed 256-token forward latency.

Against plain D8, corrected Anderson had 15 paired wins and 21 paired losses (two-sided exact sign-test p approximately 0.405). It therefore did not improve held-out accuracy and did not approach D16. Unlike the invalid original arm, this is a substantive test in which Anderson mixing was active throughout.
