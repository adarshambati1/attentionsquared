# Experiment 001: Depth Scaling

## Question

Does pretrained Huginn improve on reasoning tasks as recurrent depth increases?

## Independent variable

Recurrent depth:

D = {1, 4, 8, 16, 32}

## Dependent variables

- task accuracy
- wall-clock latency
- generated tokens

## Initial benchmark

GSM8K

## Go criterion

Increasing recurrent depth produces a measurable capability improvement.

## Kill criterion

D = 32 performs essentially the same as D = 1 on the chosen benchmark.

## Notes

Use the same prompts, prompt formatting, decoding settings, and random seed across all depths.
