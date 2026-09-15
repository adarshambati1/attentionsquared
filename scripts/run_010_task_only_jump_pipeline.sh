#!/usr/bin/env bash
set -euo pipefail
cd /workspace/attentionsquared
PY=/workspace/miniconda/envs/attentionsquared/bin/python
export HF_HOME=/workspace/hf-cache HF_DATASETS_CACHE=/workspace/hf-cache/datasets
ROOT=/workspace/task_only_jump
preserve_training_failure() {
  status=$?
  if [[ $status -ne 0 && -d "$ROOT" && ! -f "$ROOT/training_summary.json" ]]; then
    failed="/workspace/.task_only_jump.failed-$($PY -c 'import uuid; print(uuid.uuid4().hex)')"
    mv "$ROOT" "$failed"
    printf 'Preserved failed pre-test attempt at %s\n' "$failed" >&2
  fi
  exit "$status"
}
trap preserve_training_failure EXIT
$PY scripts/check_009_fixed_point_jump_correctness.py --config configs/010_task_only_jump.json
$PY scripts/train_009_fixed_point_jump.py --config configs/010_task_only_jump.json
trap - EXIT
# Once training selection is complete, evaluator owns its provenance-checked resume policy.
$PY scripts/evaluate_009_fixed_point_jump.py --config configs/010_task_only_jump.json
