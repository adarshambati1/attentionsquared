# Experiment 010 conclusion

The task-only control used the exact Experiment-009 256-bottleneck architecture, deterministic initialization, optimizer, 1,000-step budget, splits, and validation protocol, but set the fixed-point coefficient to zero and executed no recurrent core during training or inference. Step 1000 was selected by validation answer-token CE 2.959434.

Held-out GSM8K accuracy was **3/250 (1.2%)**, compared with 0/250 for the lambda=0.1 fixed-point-regularized jump. Task-only teacher-forced answer-token CE was 3.00160 versus 3.01453 with the regularizer. Its measured post-training fixed-point residual was 1.29683 versus 0.66297 with the regularizer. Thus the regularizer moved the prediction substantially closer to fixed-point consistency but did not cause the central one-shot failure: removing it recovered only three examples and remained far below plain D8's 16.0%.

Task-only cap-hit and repetition-degeneration rates were both 12.4%, versus 37.6% for the regularized jump. The one-shot architecture remains fast (7.151 ms per fixed 256-token forward) but functionally inadequate under either objective.
