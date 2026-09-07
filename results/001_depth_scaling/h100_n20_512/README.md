# H100 N=20, 512-token cap

Hardware: NVIDIA H100 80GB HBM3, US-MO-1, driver 580.126.09.

Same 20 fixed GSM8K examples and depths as `h100_n20`; `max_new_tokens=512` is the only intended cap change. Huginn stop strings `<|end_text|>` and `<|end_turn|>` are enabled. This run is a cap-sensitivity checkpoint, not yet the N=100 result.
