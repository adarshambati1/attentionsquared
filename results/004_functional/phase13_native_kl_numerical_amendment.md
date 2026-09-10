# Phase 13C native-cache KL numerical amendment

Before Phase 13 execution, the exact-equality predicate `KL/token == 0.0` is replaced with:

\[
|\mathrm{KL/token}| \le 10^{-8}.
\]

This addresses floating-point reduction residuals such as the already-observed teacher self-KL `-8.744657775672238e-10`. It does not relax the substantive native-cache equivalence claim.

The following gates remain exact and unchanged:

- cached/live native state tensors are bitwise equal;
- cached/live coda logits are bitwise equal; and
- next-token argmax agreement is 100%.

The new absolute KL tolerance is 50,000 times tighter than the superseded cache-v3 FP16 quantization bound of `5e-4`. No Phase 13 output existed when this amendment was recorded.
