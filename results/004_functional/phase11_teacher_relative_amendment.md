# Prospective Phase 11 teacher-relative amendment

The first cache-v3 Phase 11 run remains a formal failure under its preregistered absolute criterion `mean_first_target_probability >= 0.5`. It is not retroactively relabeled.

Preserved evidence:

- Failed result JSON SHA-256: `1748d3188f77227007843d800acdb9c78f04096b458d84489ed050639b938e30`
- Failed model SHA-256: `5fb7d0d467cb322a235c1293af0f9e9bd65249d32820fb0086e914b5e49307f0`
- Teacher-relative diagnostic SHA-256: `bf2d83a3131e3111f6abec86cecc018ea4cda64b409dd71ab4f7834473eefa8d`

The diagnostic found mean Huginn target-token probability `0.4795518256723881` and mean A² target-token probability `0.4832734167575836`. Science review `4e271d7c` concluded that the absolute `0.5` criterion was mis-specified for the frozen `KL(T||S)` objective.

Before the one authorized clean rerun, the sole amended criterion is preregistered as:

`mean_first_target_probability >= mean_teacher_first_target_probability`

This probability criterion remains secondary. The existing teacher-distribution KL, KL decrease, first-token distribution KL, target top-1, generation, natural-stop, answer-agreement, and no-repetition criteria remain unchanged. Model topology, objective, optimizer, data, shared initialization, seeds, update bounds, and generation protocol remain unchanged.
