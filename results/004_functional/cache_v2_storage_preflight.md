# Functional cache v2 durable-storage preflight

Date: 2026-09-09

## Decision

Use a dedicated 100 GB RunPod network volume and build cache v2 directly on it.

- Volume: `attentionsquared-functional-v2`
- Volume ID: `jggambl3qv`
- Data center: `US-GA-2`
- Extraction Pod: `jwc04iw92bdebm`
- GPU: NVIDIA H100 80GB HBM3
- Mount: `/workspace`

The previously identified volume `blank_gold_gull` (`ni2r4v1v3m`) was rejected:
it contains an unrelated LIBERO/OpenVLA project and had only approximately 3–4
GB free. Only a uniquely named preflight directory was created and removed;
none of its existing contents were modified. The temporary Pod used to inspect
that volume was deleted immediately.

## Dedicated-volume checks

The dedicated volume was empty before setup. A disposable 1 GiB preflight file
was used and removed after these checks:

- write access: PASS
- hard-link atomic no-replace: PASS
- atomic rename: PASS
- file and directory fsync: PASS
- sequential 1 GiB write: 2.062 seconds, 496.6 MiB/s
- sequential 1 GiB read: 1.779 seconds, 575.6 MiB/s
- preflight cleanup: PASS

The RunPod control plane reports the volume size as 100 GB. `df` reports the
backing distributed filesystem rather than the per-volume quota, so capacity
planning uses the control-plane allocation and measured contents.

## Capacity calculation

The 2,500 preserved continuations contain 700,976 tokens in total. Three
float16 tensors at Huginn hidden size 5,280 require exactly 22,206,919,680 bytes
(20.682 GiB) before NPZ/container overhead. The copied Conda environment,
Huginn cache, and transfer quarantine occupy approximately 39 GB, leaving ample
room within the dedicated 100 GB allocation for one shared compact cache.

## Migration and integrity

- `/workspace/miniconda` copied; required interpreter is
  `/workspace/miniconda/envs/attentionsquared/bin/python`.
- `/workspace/hf-cache` copied with the pinned Huginn revision.
- The initial relay attempt flattened copied directories at the destination.
  No originals were affected. The complete copies were reorganized into their
  required paths, while the earlier 5.1 GB partial tar copy was preserved
  visibly at `/workspace/transfer_quarantine` rather than deleted.
- Repository checked out at Phase 5 commit `68ff3ad` before Phase 6 builder work.
- All 2,500 continuation hashes and the reviewed valid-end manifest hash were
  revalidated on the new volume.
- Pod test suite on the new volume: 84 passed before Phase 6 code sync and 88
  passed after it.
- A live one-example Huginn D16 extraction produced finite `(155, 5280)`
  float16 `h0_full`, `x_full`, and `h16_teacher` arrays.
- Repeating extraction after resetting the authoritative per-example seed
  produced byte-identical SHA-256 hashes for all three arrays.

The old H100 Pod `zse9sc33ii0olu` was stopped, not deleted. Its historical
artifacts, invalid-v1 models, and invalid-v1 evaluation remain preserved on its
Pod storage.
