# Fixed-point residual audit conclusion

Using the exact one-shot loss estimand on 29,399 held-out answer tokens, native Huginn residuals fell sharply with recurrent depth:

| Depth | Relative fixed-point residual |
|---:|---:|
| 0 | 3.342181 |
| 8 | 0.080695 |
| 16 | 0.006022 |
| 32 | 0.000234 |
| 64 | 0.000170 |

Manual state capture matched native normalized Huginn states bitwise at every audited nonzero depth. The failed lambda=0.1 one-shot checkpoint had residual 0.662974: far below D0, but approximately 8.2 times D8, 110 times D16, and 2,834 times D32. It did not reach Huginn's late-depth fixed-point regime.

Under held-out teacher forcing, the checkpoint had answer-token CE 3.01453, answer-token accuracy 36.11%, first-answer-token CE 4.43779, and first-answer-token accuracy 16.0%. Therefore the model was already weak before autoregressive error accumulation; its 0% generated-answer accuracy was not solely a decoding exposure failure.
