# Experiment 007 conclusion

On the frozen 250-example GSM8K test set, plain Huginn D8 scored 40/250 (16.0%), the shared trainable current-state-only V/O adapter scored 54/250 (21.6%), and parameter-free raw arithmetic loop-history averaging scored 3/250 (1.2%). The current-state-only checkpoint was selected at step 500 by its lowest full validation answer-token CE, 0.4295351998763402; the test set was not used for selection.

The current-state adapter recovers most of the projected-history result (22.4%) and shared learned-history result (23.2%), while raw latent averaging is destructive. This supports attributing most of the apparent history-module improvement to learned state processing rather than historical retrieval itself.
