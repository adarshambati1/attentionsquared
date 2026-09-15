# Experiment 009 conclusion

The correctness gate passed, Huginn remained frozen, and inference executed prelude → one shared learned jump → frozen native normalization/coda/head with zero recurrent core steps. The predictor had 4,060,576 trainable parameters and was trained for 1,000 optimizer steps with answer-token CE plus fixed weight 0.1 on the preregistered relative fixed-point consistency loss. Step 1000 was selected by the lowest full validation answer-token CE, 2.970119440728525; the test set was never used for selection.

On held-out GSM8K test IDs 0–249, the one-shot jump scored **0/250 (0.0%)**. Its cap-hit and repetition-degeneration rates were both 37.6%, mean generation length was 422.624 tokens, teacher-forced answer-token CE was 3.01453, and its relative fixed-point residual remained 0.66297. Fixed 256-token forward latency was 7.125 ms, versus approximately 55–56 ms for recurrent D8 methods.

The one-shot route is substantially faster but does not preserve useful task behavior under this frozen architecture and objective. It fails far below plain D8 (16.0%), the current-state adapter D8 (21.6%), shared-history D8 (23.2%), and plain D16 (38.8%). This supports the narrow conclusion that this small functionally trained predictor could not replace Huginn's iterative computation. It does not prove that every possible one-shot architecture or loss must fail.
