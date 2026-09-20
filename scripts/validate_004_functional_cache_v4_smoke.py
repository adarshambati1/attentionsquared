#!/usr/bin/env python3
"""Independent production validation/replay for the Phase 9R cache-v4 smoke."""
from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.correctness import build_chat_prompt, per_example_seed, tokenize_prompt
from src.data.functional_extraction_v4 import (
    bfloat16_tensor_to_uint16,
    uint16_to_bfloat16_tensor,
)

CONFIG = Path("configs/004_functional_cache_v4_smoke.json")
CACHE_ROOT = Path("/workspace/functional_cache_v4_smoke")
MODEL_ID = "tomg-group-umd/huginn-0125"
MODEL_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
SYSTEM_INSTRUCTION = "You are a helpful assistant that can assist users with mathematical reasoning."
IDS = tuple(range(8))
EXPECTED_BUILDER_COMMIT = "057a3955388676de3fa61ad21acd843d6253fcf3"
LOCKED_HASHES = {
    "split_manifest_sha256": "afdbc0c55196104e276df0d65c8df24664836874a87b623cd5f3dd8331d1a5d8",
    "valid_end_manifest_sha256": "25188506cb7afe426e42beb5fa96455fe8ac0cff6c0ae47b72363a3fa7af187f",
    "source_cache_v2_manifest_sha256": "32db951f6cfbf76220b1c7126e23149f087eba58934ab691b588aeb3a43535bf",
    "source_cache_v2_freeze_sha256": "94417eab65fd04a5827bdef9aadc9a5b8b66c266700b6cfc7ab56d39f949d3f0",
}
RAW_HASHES = {
    "0":"8ddef09070e5b01b4a43f474b365f95e2d361d2afbf7f5eb151f8029b220f967","1":"e5f7bcaf7b3ad0627592ff727c32a2f10df99269ab54441919289b04dff3c025","2":"a36e3a8cab93e126059c54fd59eb7ee14b41394ab1dd30fac2ad95e6b6db2e38","3":"ba6dffd0c3bea158fe57f84fc5c258ec0b9b53fa31875a5858d6cbe0c4d34a50","4":"04df30834bf1b9cf3be946aab9d4b53b30125fa308fabe31f9532cdeb4c0fe5b","5":"f76a673cfba89cc5be6f136a8aa6b98ac163f7e826fbb20fe5ddf6905a5e2430","6":"f3e7c18f2604443999a48497144765baa28955b4bb30a8e90f44bda54ea1dd0e","7":"20453212328435fcab00af8e34cd63b03ff897143c9e397dc28f473afb7da297",
}
ITEM_KEYS = {"input_ids","attention_mask","answer_start","valid_end","h0","x","h16","example_id","dataset_split","base_seed","seed_index","derived_seed","seed_protocol","schedule_length","schedule_protocol","full_schedule_sha256","source_sha256","h16_semantics","cache_protocol","manifest_sha256"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    if value.dtype != torch.bfloat16 or tuple(value.shape[:2]) != (1, 2048):
        raise ValueError("schedule is not BF16 [1,2048,H]")
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _load_locked_inputs() -> tuple[dict, list[dict]]:
    config = json.loads((ROOT / CONFIG).read_text())
    fixed = {
        "protocol": "functional-cache-v4-smoke-build-v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": "openai/gsm8k",
        "dataset_config": "main",
        "dataset_revision": DATASET_REVISION,
        "split_manifest": "configs/004_functional_v2_splits.json",
        "valid_end_manifest": "results/004_functional/valid_end_manifest.jsonl",
        "source_cache_v2_manifest": "results/004_functional/cache_v2_manifest.json",
        "source_cache_v2_freeze": "results/004_functional/cache_v2_FROZEN.json",
        "source_continuations": "results/004_functional/teacher_sequences",
        "example_ids": list(IDS),
        "teacher_depth": 16,
        "compute_dtype": "bfloat16",
        "state_dtype": "bfloat16-bits-uint16",
        "hidden_size": 5280,
        "base_seed": 3000,
        "seed_index": 0,
        "schedule_length": 2048,
        "generation_cap_tokens": 1024,
        "maximum_context_tokens": 2048,
        "cache_output": str(CACHE_ROOT),
    }
    expected_keys = {*fixed, *LOCKED_HASHES, "raw_source_sha256"}
    if set(config) != expected_keys:
        raise ValueError("independent config schema lock failed")
    for key, expected in {**fixed, **LOCKED_HASHES}.items():
        if config[key] != expected:
            raise ValueError(f"independent config lock failed: {key}")
    if config["raw_source_sha256"] != RAW_HASHES:
        raise ValueError("independent raw hash lock failed")
    paths = (("split_manifest","split_manifest_sha256"),("valid_end_manifest","valid_end_manifest_sha256"),("source_cache_v2_manifest","source_cache_v2_manifest_sha256"),("source_cache_v2_freeze","source_cache_v2_freeze_sha256"))
    for path_key, hash_key in paths:
        if _sha256(ROOT / config[path_key]) != LOCKED_HASHES[hash_key]:
            raise ValueError(f"locked file hash failed: {path_key}")
    rows = []
    with (ROOT / config["valid_end_manifest"]).open() as stream:
        for line in stream:
            if len(rows) == 8:
                break
            rows.append(json.loads(line))
    for example_id, row in enumerate(rows):
        required = {"example_id":example_id,"dataset_id":"openai/gsm8k","dataset_config":"main","dataset_revision":DATASET_REVISION,"dataset_split":"train","phase3_role":"gradient_update","source_sha256":RAW_HASHES[str(example_id)]}
        if any(row.get(key) != value for key, value in required.items()):
            raise ValueError(f"valid-end row lock failed for ID={example_id}")
        if not 1 <= row["answer_start"] < row["reviewed_valid_end"] <= row["source_sequence_length"] <= 2048:
            raise ValueError(f"invalid independently parsed bounds for ID={example_id}")
    return config, rows


def _load_raw(config: dict, row: dict) -> np.ndarray:
    path = ROOT / config["source_continuations"] / row["source_relative_path"]
    if path.is_symlink() or not path.is_file() or _sha256(path) != row["source_sha256"]:
        raise ValueError(f"raw source identity failed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"input_ids", "answer_start", "generated_tokens", "truncated", "pathological", "valid_end"}:
            raise ValueError(f"unexpected immutable raw NPZ schema: {path}")
        ids = archive["input_ids"].copy()
        answer_start = int(archive["answer_start"])
        raw_valid_end = int(archive["valid_end"])
        generated_tokens = int(archive["generated_tokens"])
        truncated = bool(archive["truncated"])
        pathological = bool(archive["pathological"])
    if ids.dtype != np.int32 or ids.ndim != 1 or len(ids) != row["source_sequence_length"] or answer_start != row["answer_start"]:
        raise ValueError(f"raw sequence/boundary mismatch: {path}")
    if (
        raw_valid_end != row["original_valid_end"]
        or row["reviewed_valid_end"] != raw_valid_end
        or generated_tokens != len(ids) - answer_start
        or truncated != row["source_truncated_at_cap"]
        or pathological != row["source_pathological"]
    ):
        raise ValueError(f"raw/reviewed continuation metadata mismatch: {path}")
    return ids


def validate_file_set(root: Path = CACHE_ROOT) -> list[Path]:
    if root.is_symlink() or not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise ValueError("cache root must be a real mode-0555 directory")
    expected = {root / "manifest.json", *(root / f"{i:05d}.npz" for i in IDS)}
    allowed = {root / ".functional-cache-v4.lock"}
    actual = {path for path in root.rglob("*") if path.is_file()}
    if actual not in (expected, expected | allowed) or any(path.is_dir() for path in root.rglob("*")):
        raise ValueError("exact file set must be manifest plus eight items; no K directories")
    if any(path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o444 for path in actual):
        raise ValueError("all cache payloads must be real mode-0444 files")
    return [root / f"{i:05d}.npz" for i in IDS]


def validate_static(root: Path = CACHE_ROOT):
    config, rows = _load_locked_inputs()
    items = validate_file_set(root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest_fixed = {
        "protocol": "functional-cache-v4-smoke-manifest-v1",
        "huginn_model_id": MODEL_ID,
        "huginn_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "model_compute_dtype": "bfloat16",
        "state_dtype": "bfloat16-bits-uint16",
        "hidden_size": 5280,
        "extraction_location": "explicit scheduled BF16 input_states -> frozen Huginn D16; x=post-prelude recurrent input; h16=returned normalized pre-coda latent; all state tensors persisted losslessly as raw BF16 uint16 bits",
        "example_ids": list(IDS),
        "sequence_count": 8,
        "source_continuations": config["source_continuations"],
        "valid_end_manifest_sha256": LOCKED_HASHES["valid_end_manifest_sha256"],
        "split_manifest_sha256": LOCKED_HASHES["split_manifest_sha256"],
        "source_cache_v2_manifest_sha256": LOCKED_HASHES["source_cache_v2_manifest_sha256"],
        "source_cache_v2_freeze_sha256": LOCKED_HASHES["source_cache_v2_freeze_sha256"],
        "seed_policy": "injective-packed-coordinates-v3",
        "base_seed": 3000,
        "seed_index": 0,
        "schedule_length": 2048,
        "schedule_protocol": "huginn-initialize-state-once-bfloat16-cuda-[1,2048,H]-slice-prefix-v1",
        "generation_cap_tokens": 1024,
        "maximum_context_tokens": 2048,
        "h16_semantics": "D16 recurrent output normalized exactly once by transformer.ln_f; input to coda blocks; coda then final ln_f then lm_head; no extra initial ln_f",
        "full_vocabulary_logits_persisted": False,
        "cache_output": str(CACHE_ROOT),
    }
    dynamic_manifest_keys = {
        "source_sha256_by_id", "estimated_uncompressed_item_bytes",
        "tokenizer_template_sha256", "git_commit",
    }
    if set(manifest) != {*manifest_fixed, *dynamic_manifest_keys}:
        raise ValueError("independent manifest schema lock failed")
    for key, expected in manifest_fixed.items():
        if manifest[key] != expected:
            raise ValueError(f"independent manifest lock failed: {key}")
    if manifest["source_sha256_by_id"] != RAW_HASHES:
        raise ValueError("manifest raw source hash map failed")
    if not isinstance(manifest.get("estimated_uncompressed_item_bytes"), int) or manifest["estimated_uncompressed_item_bytes"] <= 0:
        raise ValueError("manifest byte estimate failed")
    for key in ("tokenizer_template_sha256",):
        if not isinstance(manifest.get(key), str) or len(manifest[key]) != 64:
            raise ValueError(f"manifest digest failed: {key}")
    if manifest["git_commit"] != EXPECTED_BUILDER_COMMIT:
        raise ValueError("cache was not built from the authorized Phase 9R commit")
    manifest_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    values = []
    for example_id, (row, item) in enumerate(zip(rows, items)):
        raw = _load_raw(config, row)
        with np.load(item, allow_pickle=False) as archive:
            if set(archive.files) != ITEM_KEYS or any("logit" in key.lower() for key in archive.files):
                raise ValueError("independent item schema check failed")
            value = {key: archive[key].copy() for key in archive.files}
        if not np.array_equal(value["input_ids"], raw):
            raise ValueError(f"cached/raw token mismatch ID={example_id}")
        if value["attention_mask"].dtype != np.bool_ or value["attention_mask"].shape != raw.shape or not value["attention_mask"].all():
            raise ValueError(f"cached mask mismatch ID={example_id}")
        scalar_expected = {
            "example_id": example_id,
            "answer_start": row["answer_start"],
            "valid_end": row["reviewed_valid_end"],
            "base_seed": 3000,
            "seed_index": 0,
            "derived_seed": per_example_seed(3000, example_id, seed_index=0),
            "schedule_length": 2048,
        }
        if any(int(value[key]) != expected for key, expected in scalar_expected.items()):
            raise ValueError(f"cached scalar provenance mismatch ID={example_id}")
        string_expected = {
            "dataset_split": "train",
            "seed_protocol": "injective-packed-coordinates-v3",
            "schedule_protocol": manifest_fixed["schedule_protocol"],
            "source_sha256": RAW_HASHES[str(example_id)],
            "h16_semantics": manifest_fixed["h16_semantics"],
            "cache_protocol": "functional-cache-v4-smoke-item-v1",
            "manifest_sha256": manifest_digest,
        }
        if any(str(value[key]) != expected for key, expected in string_expected.items()):
            raise ValueError(f"cached string provenance mismatch ID={example_id}")
        if len(str(value["full_schedule_sha256"])) != 64:
            raise ValueError(f"cached schedule digest mismatch ID={example_id}")
        for key in ("h0","x","h16"):
            if value[key].shape != (len(raw), 5280) or value[key].dtype != np.uint16 or not np.isfinite(value[key]).all():
                raise ValueError(f"cached BF16-bit state schema failed {key} ID={example_id}")
        values.append(value)
    return config, rows, manifest, values


def _independent_schedule(model, hidden: int, example_id: int, request: str):
    seed = per_example_seed(3000, example_id, seed_index=0)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    template = torch.empty((1, 2048, hidden), device="cuda", dtype=torch.bfloat16)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())), enabled=True):
        torch.manual_seed(seed)
        schedule = model.initialize_state(template, scale=1.0)
    restored = torch.equal(cpu_before, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
    if not restored:
        raise ValueError(f"RNG restoration failed for {request} request ID={example_id}")
    if schedule.shape != template.shape or schedule.dtype != torch.bfloat16 or schedule.device.type != "cuda":
        raise ValueError("initializer returned noncanonical schedule")
    return schedule, seed


def _decomposed_coda(model, normalized: torch.Tensor, frequencies):
    state = normalized
    block_index = torch.tensor(0, device="cpu", dtype=torch.long)
    for block in model.transformer.coda:
        block_index -= 1
        state = block(state, frequencies, block_index, None, None)
    state = model.transformer.ln_f(state)
    return model.lm_head(state).float()


def _normal_d16(model, ids: torch.Tensor, h0: torch.Tensor) -> dict[str, torch.Tensor]:
    """Trusted whole-model route, with no validator hook or coda helper."""
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids, dtype=torch.bool),
            input_states=h0,
            num_steps=16,
            use_cache=False,
            return_dict=True,
            output_details={"return_logits": True, "return_latents": True, "return_head": False, "return_stats": False},
        )
    return {"h16": output.latent_states.detach().clone(), "logits": output.logits.float().detach().clone()}


def _decomposed_d16(model, ids: torch.Tensor, h0: torch.Tensor) -> dict[str, torch.Tensor]:
    """Separate hooked D16 route followed by a literal local coda decomposition."""
    captured = {}
    calls = 0
    original = model.core_block_forward

    def wrapped(this, state, recurrent_input, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            if not torch.equal(state, h0):
                raise RuntimeError("explicit schedule slice was not consumed")
            captured["x"] = recurrent_input.detach().clone()
        result = original(state, recurrent_input, *args, **kwargs)
        if calls == 16:
            captured["pre_ln"] = (result[0] if isinstance(result, tuple) else result).detach().clone()
        return result

    model.core_block_forward = MethodType(wrapped, model)
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids, dtype=torch.bool),
                input_states=h0,
                num_steps=16,
                use_cache=False,
                return_dict=True,
                output_details={"return_logits": False, "return_latents": True, "return_head": False, "return_stats": False},
            )
            normalized = model.transformer.ln_f(captured["pre_ln"])
            logits = _decomposed_coda(model, normalized, model.freqs_cis[:, :ids.shape[1]])
    finally:
        model.core_block_forward = original
    if calls != 16 or set(captured) != {"x", "pre_ln"}:
        raise ValueError(f"independent D16 capture expected 16 calls, got {calls}")
    if not torch.equal(normalized, output.latent_states):
        raise ValueError("ln_f(pre-D16) does not exactly equal decomposed-route latent")
    return {"x": captured["x"], "h16": normalized, "logits": logits}


def _independent_d16_pair(model, ids: torch.Tensor, h0: torch.Tensor) -> dict[str, torch.Tensor]:
    normal = _normal_d16(model, ids, h0)
    decomposed = _decomposed_d16(model, ids, h0)
    if not torch.equal(normal["h16"], decomposed["h16"]):
        raise ValueError("separate normal/decomposed D16 latents differ")
    if not torch.equal(normal["logits"], decomposed["logits"]):
        raise ValueError("separate normal/decomposed D16 logits differ")
    return decomposed


def replay(config, rows, manifest, values):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    if not isinstance(tokenizer.chat_template, str) or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != manifest["tokenizer_template_sha256"]:
        raise ValueError("pinned tokenizer template differs from cache manifest")
    dataset = load_dataset("openai/gsm8k", "main", split="train", revision=DATASET_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True).eval().cuda().requires_grad_(False)
    records = []
    total_kl = 0.0
    total_tokens = 0
    total_top1 = 0
    cache_coda_tokens = 0
    cache_coda_kl_sum = 0.0
    cache_coda_argmax_equal = 0
    cache_coda_logit_max_abs = 0.0
    cache_coda_logit_abs_sum = 0.0
    cache_coda_logit_elements = 0
    original_initializer = model.initialize_state
    initializer_calls = 0
    def counted(this, *args, **kwargs):
        nonlocal initializer_calls
        initializer_calls += 1
        return original_initializer(*args, **kwargs)
    model.initialize_state = MethodType(counted, model)
    try:
        for example_id, (row, cached) in enumerate(zip(rows, values)):
            raw = cached["input_ids"]
            prompt_text = build_chat_prompt(tokenizer, dataset[example_id]["question"], SYSTEM_INSTRUCTION)
            prompt_ids = tokenize_prompt(tokenizer, prompt_text)["input_ids"].cpu().numpy()[0].astype(np.int32)
            if len(prompt_ids) != row["answer_start"] or not np.array_equal(prompt_ids, raw[:row["answer_start"]]):
                raise ValueError(f"native pinned prompt is not exact raw prefix ID={example_id}")
            before = initializer_calls
            prompt_schedule, prompt_seed = _independent_schedule(model, 5280, example_id, "prompt")
            full_schedule, full_seed = _independent_schedule(model, 5280, example_id, "full")
            if initializer_calls - before != 2:
                raise ValueError(f"expected exactly two initializer calls ID={example_id}")
            prompt_hash, full_hash = _tensor_hash(prompt_schedule), _tensor_hash(full_schedule)
            if prompt_seed != full_seed or prompt_hash != full_hash or full_hash != str(cached["full_schedule_sha256"]):
                raise ValueError(f"independent full schedule provenance mismatch ID={example_id}")
            for prefix_length in range(1, len(raw) + 1):
                if not torch.equal(
                    prompt_schedule[:, :prefix_length],
                    full_schedule[:, :prefix_length],
                ):
                    raise ValueError(
                        f"represented schedule overlap differs ID={example_id} "
                        f"t={prefix_length}"
                    )
                observed_h0 = bfloat16_tensor_to_uint16(
                    full_schedule[:, :prefix_length][0]
                )
                if observed_h0.tobytes() != cached["h0"][:prefix_length].tobytes():
                    raise ValueError(f"cached h0 prefix is not exact ID={example_id} t={prefix_length}")
            full_ids = torch.from_numpy(raw.astype(np.int64))[None].cuda()
            teacher = _independent_d16_pair(
                model, full_ids, full_schedule[:, :len(raw)]
            )
            for key in ("x", "h16"):
                observed = bfloat16_tensor_to_uint16(teacher[key][0])
                if observed.tobytes() != cached[key].tobytes():
                    raise ValueError(
                        f"cached {key} BF16 bits are not byte-identical ID={example_id}"
                    )
            cached_h16 = uint16_to_bfloat16_tensor(
                cached["h16"], device="cuda"
            )[None]
            if not torch.equal(cached_h16, teacher["h16"]):
                raise ValueError(f"loaded cached h16 differs from live BF16 ID={example_id}")
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                cached_logits = _decomposed_coda(
                    model, cached_h16, model.freqs_cis[:, :len(raw)]
                )
            represented = slice(row["answer_start"] - 1, row["reviewed_valid_end"] - 1)
            cached_selected = cached_logits[:, represented].float()
            live_selected = teacher["logits"][:, represented].float()
            coda_delta = (cached_selected - live_selected).abs()
            live_log = F.log_softmax(live_selected, dim=-1)
            cached_log = F.log_softmax(cached_selected, dim=-1)
            item_kl_sum = float((live_log.exp() * (live_log - cached_log)).sum())
            item_tokens = int(live_selected.shape[1])
            item_argmax = int(
                (cached_selected.argmax(-1) == live_selected.argmax(-1)).sum()
            )
            if not torch.equal(cached_selected, live_selected):
                raise ValueError(f"cached BF16 coda logits differ from live ID={example_id}")
            if item_argmax != item_tokens:
                raise ValueError(f"cached BF16 coda argmax differs from live ID={example_id}")
            cache_coda_tokens += item_tokens
            cache_coda_kl_sum += item_kl_sum
            cache_coda_argmax_equal += item_argmax
            cache_coda_logit_max_abs = max(
                cache_coda_logit_max_abs, float(coda_delta.max())
            )
            cache_coda_logit_abs_sum += float(coda_delta.sum())
            cache_coda_logit_elements += int(coda_delta.numel())
            prompt_tensor = torch.from_numpy(prompt_ids.astype(np.int64))[None].cuda()
            prefix = _independent_d16_pair(
                model, prompt_tensor, prompt_schedule[:, :len(prompt_ids)]
            )
            count = len(prompt_ids)
            diagnostics = {}
            for key in ("x","h16"):
                delta = (prefix[key].float() - teacher[key][:, :count].float()).abs()
                diagnostics[f"{key}_mean_abs"] = float(delta.mean())
                diagnostics[f"{key}_max_abs"] = float(delta.max())
            prefix_logits = prefix["logits"].float()
            teacher_logits = teacher["logits"][:, :count].float()
            logit_delta = (prefix_logits - teacher_logits).abs()
            diagnostics["logit_mean_abs"] = float(logit_delta.mean())
            diagnostics["logit_max_abs"] = float(logit_delta.max())
            p_log = F.log_softmax(prefix_logits, dim=-1)
            q_log = F.log_softmax(teacher_logits, dim=-1)
            kl_sum = float((p_log.exp() * (p_log - q_log)).sum())
            top1 = int((prefix["logits"].argmax(-1) == teacher["logits"][:, :count].argmax(-1)).sum())
            diagnostics["globally_token_normalized_kl"] = kl_sum / count
            diagnostics["top1_agreement"] = top1 / count
            total_kl += kl_sum; total_tokens += count; total_top1 += top1
            records.append({"example_id":example_id,"prompt_tokens":count,"sequence_tokens":len(raw),"represented_teacher_tokens":item_tokens,"initializer_calls":2,"rng_restored_each_request":True,"full_bf16_schedule_sha256":full_hash,"schedule_overlap_bitwise":True,"cached_h0_bfloat16_bits_exact":True,"d16_calls_each_request":16,"cached_x_h16_bfloat16_bits_exact":True,"loaded_h16_equals_live_bfloat16":True,"cached_coda_logits_equal_live":True,"cached_coda_kl_live_to_cached":item_kl_sum/item_tokens,"cached_coda_argmax_agreement":item_argmax/item_tokens,"normal_vs_decomposed_logits_exact":True,"prefix_shape_diagnostics":diagnostics})
            del prompt_schedule, full_schedule, teacher, prefix, full_ids, prompt_tensor
            torch.cuda.empty_cache()
    finally:
        model.initialize_state = original_initializer
    return {
        "examples": records,
        "lossless_cache_coda_hard_gate": {
            "represented_teacher_tokens": cache_coda_tokens,
            "state_bitwise_equal": True,
            "logit_mean_abs": cache_coda_logit_abs_sum / cache_coda_logit_elements,
            "logit_max_abs": cache_coda_logit_max_abs,
            "globally_token_normalized_kl_live_to_cached": cache_coda_kl_sum / cache_coda_tokens,
            "argmax_equal": cache_coda_argmax_equal,
            "argmax_total": cache_coda_tokens,
            "argmax_agreement": cache_coda_argmax_equal / cache_coda_tokens,
            "passed": (
                cache_coda_logit_max_abs == 0.0
                and cache_coda_argmax_equal == cache_coda_tokens
            ),
        },
        "prefix_shape_diagnostics_global": {
            "represented_prompt_tokens": total_tokens,
            "globally_token_normalized_kl": total_kl / total_tokens,
            "top1_agreement": total_top1 / total_tokens,
        },
        "prefix_shape_diagnostics_only_no_acceptance_threshold": True,
    }


def main():
    config, rows, manifest, values = validate_static()
    result = replay(config, rows, manifest, values)
    print(json.dumps({"protocol":"functional-cache-v4-smoke-independent-validation-v3","status":"pass","cache_builder_git_commit":manifest["git_commit"],"h0_exactness_hard_gate":True,"full_schedule_persisted":False,"logits_persisted":False,**result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
