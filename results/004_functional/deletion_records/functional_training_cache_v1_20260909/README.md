# Deletion record: functional full-logit cache v1

## Artifact

- Remote path: `/root/functional_training_cache`
- Classification: **invalid target alignment; prohibited for training and scientific inference**
- Files: 2,501 (2,500 NPZ items plus `INVALID_V1.md`)
- Exact file bytes: 77,824,465,425
- Empty files: none at deletion inventory time

## Reason and authorization

The cache used teacher logits from `[answer_start:valid_end]` rather than the
causal predecessor logits `[answer_start-1:valid_end-1]`. This omitted the
first answer-token prediction and invalidated the cache as supervision.

The user explicitly authorized deletion in the active pi session on 2026-09-09
after being informed that the cache was invalid, derived, expensive to
regenerate, and that deletion would lose the exact stored tensor bytes.

## Regeneration inputs checked before deletion

The durable teacher-continuation source under
`/workspace/attentionsquared/results/004_functional/teacher_sequences`
contained all 2,500 expected NPZ files:

- historical `train`: 2,000;
- historical `val`: 250;
- historical `test`: 250.

The source Huginn model and dataset revisions remain pinned in the repository.
A scientifically corrected compact v2 cache can be regenerated from those continuations
and the frozen model. Exact byte-for-byte reproduction of this invalid v1 derivative is
not promised.

## Audit files

- `file_inventory.tsv` — relative path and byte size before deletion
- `sha256sums.txt` — SHA-256 for every file before deletion
- Inventory SHA-256: `8e0cd2e7c40b7c2732de6c4cc4e7ff988be7bd7438cbc49154f379ce5b34bdcb`
- Checksum-list SHA-256: `d7887aa0d2d6651906e13cf3e8e8ebbd1dedbcf3d13356b187bf5584c02a437a`

## Completion

Deletion completed at `2026-09-09T07:49:50Z`. The failed 4.48 GB partial staging
copy created during the preceding durability attempt was also removed. The durable
audit metadata was retained, and the original models, teacher continuations, and
evaluation remain present.

No model, teacher continuation, evaluation result, or corrected-v2 artifact was covered
by this deletion authorization.
