# Experiment 002 conclusion: recurrent trajectory geometry

## Protocol

Captured the actual Huginn recurrent-core state trajectory for 50 fixed GSM8K
prompts at `D=64`, saving `h0...h64` for both the final-token state and the
sequence-mean state. No generation, training, predictor, or Attention²
intervention was used.

## Measurements

- Adjacent state cosine
- Normalized recurrent step magnitude
- Cosine to the final state
- Consecutive update-direction cosine
- PCA/SVD concentration of recurrent updates
- Outcome-group comparison using Experiment 001 labels

## Results

For final-token states, mean adjacent cosine rose from `0.165` at the first
transition to `0.997` by the transition at depth 15. Mean normalized step size
fell from `1.81` to `0.081` by depth 15 and continued toward zero. Mean cosine
to the final state reached `0.992` by h16. Update-direction cosine began
negative (`-0.35`) but became positive and approximately `0.70` by the middle
of the trajectory, indicating that the dynamics are not simple constant-vector
motion even though they converge.

Update PCA explained variance was approximately:

| Components | Cumulative variance |
|---:|---:|
| 1 | 23.4% |
| 4 | 53.6% |
| 8 | 62.1% |
| 16 | 66.1% |
| 32 | 71.1% |
| 64 | 78.4% |

Outcome groups contained 7 low-depth-solved, 19 high-depth-solved, and 24
unsolved prompts. All groups converged geometrically, but unsolved examples
retained larger late step sizes than solved examples.

## Classification

**A. Convergent/refinement-like, with a nontrivial early trajectory.** The
states rapidly become locally stable around the same depth where Experiment
001 found useful capability saturation. The early updates are not trivial or
collinear, so this is not evidence that the entire computation can be replaced
by one linear jump. It is evidence that a small number of learned parallel
refinement rounds is a plausible next hypothesis.

Experiment 002 supports investigating Attention², but does not demonstrate
that a parallel operator can reproduce the trajectory. That intervention is
reserved for Experiment 003.
