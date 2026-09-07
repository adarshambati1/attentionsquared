# Experiment 004A: Attention² tiny overfit

This is a mechanical sanity gate, not a GSM8K benchmark. A frozen Huginn
teacher supplies exact tokenwise `h0`, `x`, and `h1...h16` targets for 16 fixed
prompts. The A²-lite model initializes 16 slots from `(h0, x, depth_embedding)`
and applies one shared token-attention → dense bidirectional per-token
depth-attention → MLP operator four times:

```text
Z^(k+1) = A_theta(Z^k, x), k=0...3
```

The initial MLP ratio is 1 for tractability. If plumbing and mask tests pass but
the tiny set does not memorize, retry with ratio 2 before treating that as an
architectural failure. Record both one-round latency and the full four-round
latency. Do not proceed to 4B unless this gate passes.
