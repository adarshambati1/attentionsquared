"""Frozen worker-wave construction and authorization checks for Step-3 concurrent serial evaluation."""
from __future__ import annotations
from typing import Any

def condition_name(condition: list[Any] | tuple[Any, ...]) -> str:
    return f"{condition[0]}_d{condition[1]}"

def production_waves(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct exactly the waves used by run_013_huginn_concurrent.py."""
    result: list[dict[str, Any]] = []
    scopes = (("gsm8k", config["gsm8k_conditions"]), ("cross_dataset", config["cross_dataset_conditions"]))
    for scope, conditions in scopes:
        for depth in sorted({int(item[1]) for item in conditions}):
            at_depth = [condition_name(item) for item in conditions if int(item[1]) == depth]
            limit = int(config["concurrent_workers_by_depth"][str(depth)])
            for index, offset in enumerate(range(0, len(at_depth), limit)):
                members = at_depth[offset : offset + limit]
                result.append({"wave_id": f"{scope}-d{depth}-wave{index}", "scope": scope, "depth": depth, "wave_index": index, "configured_worker_limit": limit, "members": members, "workers": len(members)})
    return result

def validate_correctness_gate_contract(gate: dict[str, Any], *, config: dict[str, Any], commit: str, config_sha256: str, hardware: str) -> None:
    fixed = {"protocol": config["protocol"] + "-correctness-gate", "status": "pass", "git_commit": commit, "config_sha256": config_sha256, "hardware": hardware, "checkpoint_sha256": config["checkpoint_sha256"], "full_vocabulary_logits_persisted": False}
    if any(gate.get(key) != value for key, value in fixed.items()):
        raise RuntimeError("correctness gate fixed-contract mismatch")
    boolean_sections = {
        "seed_checks": {"seed0_repeat_bitwise", "three_pairwise_distinct", "prefix_stable"},
        "native_parity": {"plain_d8_logits_bitwise", "plain_d8_latents_bitwise"},
        "scoring_checks": {"svamp_numeric", "math_latex", "cap_disallows_unmarked_fallback"},
        "checkpoints_unchanged": set(config["checkpoint_sha256"]),
    }
    for section, keys in boolean_sections.items():
        values = gate.get(section)
        if not isinstance(values, dict) or set(values) != keys or not all(values.values()):
            raise RuntimeError(f"correctness gate {section} failed")
    transfer = gate.get("depth_transfer_finiteness")
    cache = gate.get("cache_vs_full_prefix")
    fallback = gate.get("forced_fallback_checks")
    expected_transfer = [(str(model), int(depth)) for model, depth in config["gsm8k_conditions"] if model in ("plain", "current", "shared") or int(depth) == 8]
    if not isinstance(transfer, list) or [(item.get("model"), item.get("depth")) for item in transfer] != expected_transfer or not all(set(item) == {"model", "depth", "finite_logits", "finite_latents"} and item["finite_logits"] and item["finite_latents"] for item in transfer):
        raise RuntimeError("correctness gate transfer coverage failed")
    expected_cache = [("plain", 8), ("current", 8), ("projected_uniform", 8), ("shared", 8), ("per_layer", 8), ("plain", 16), ("current", 64), ("shared", 64)]
    if not isinstance(cache, list) or [(item.get("model"), item.get("depth")) for item in cache] != expected_cache or not all(set(item) == {"model", "depth", "steps"} and len(item["steps"]) == 3 and all(set(step) == {"token_exact", "finite"} and step["token_exact"] and step["finite"] for step in item["steps"]) for item in cache):
        raise RuntimeError("correctness gate cache coverage failed")
    expected_fallback = [("plain", 8), ("current", 64), ("shared", 64), ("projected_uniform", 8), ("per_layer", 8)]
    if not isinstance(fallback, list) or [(item.get("model"), item.get("depth")) for item in fallback] != expected_fallback or not all(set(item) == {"model", "depth", "complete_token_sequence_exact", "forced_fallback_used", "fallback_onset"} and item["complete_token_sequence_exact"] and item["forced_fallback_used"] and item["fallback_onset"] == 1 for item in fallback):
        raise RuntimeError("correctness gate fallback coverage failed")

def _expected_record_keys(wave: dict[str, Any], variants: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"condition": condition, "variant": variant, "example_id": example_id, "seed_index": 0} for condition in wave["members"] for example_id in config["concurrency_gate_examples"] for variant in variants]

def validate_gate_contract(gate: dict[str, Any], *, config: dict[str, Any], commit: str, config_sha256: str, correctness_gate_sha256: str, hardware: str) -> None:
    expected_waves = production_waves(config)
    fixed = {
        "protocol": config["protocol"] + "-concurrency-gate",
        "status": "pass",
        "git_commit": commit,
        "config_sha256": config_sha256,
        "correctness_gate_sha256": correctness_gate_sha256,
        "hardware": hardware,
        "checkpoint_sha256_before": config["checkpoint_sha256"],
        "checkpoint_sha256_after": config["checkpoint_sha256"],
        "all_exact": True,
        "gate_policy": "user-approved-proportional-v1",
        "production_wave_manifest": expected_waves,
    }
    if any(gate.get(key) != value for key, value in fixed.items()):
        raise RuntimeError("concurrency gate fixed-contract mismatch")
    tiers = gate.get("tiers")
    if not isinstance(tiers, dict):
        raise RuntimeError("concurrency gate tiers missing")
    short = tiers.get("all_waves_short")
    natural = tiers.get("natural_stop_by_depth")
    stress = tiers.get("worst_case_forced_cap")
    expected_ids = [wave["wave_id"] for wave in expected_waves]
    if not isinstance(short, list) or [item.get("wave_id") for item in short] != expected_ids:
        raise RuntimeError("short parity wave coverage mismatch")
    identity = ("wave_id", "scope", "depth", "wave_index", "configured_worker_limit", "members", "workers")
    if any(not item.get("exact") or item.get("max_new_tokens") != config["concurrency_gate_short_tokens"] or item.get("variants") != ["forced_cap"] or item.get("record_keys") != _expected_record_keys(expected_waves[index], ["forced_cap"], config) or item.get("records") != len(item.get("record_keys", [])) or any(item.get(key) != expected_waves[index][key] for key in identity) for index, item in enumerate(short)):
        raise RuntimeError("short parity result mismatch")
    expected_depths = sorted({wave["depth"] for wave in expected_waves})
    selected = [max((wave for wave in expected_waves if wave["depth"] == depth), key=lambda wave: wave["workers"]) for depth in expected_depths]
    if not isinstance(natural, list) or len(natural) != len(selected) or any(not item.get("exact") or not item.get("natural_stopping_observed") or item.get("max_new_tokens") != config["max_new_tokens"] or item.get("variants") != ["natural"] or item.get("record_keys") != _expected_record_keys(selected[index], ["natural"], config) or item.get("records") != len(item.get("record_keys", [])) or any(item.get(key) != selected[index][key] for key in identity) for index, item in enumerate(natural)):
        raise RuntimeError("natural-stop parity coverage mismatch")
    stress_selected = [max((wave for wave in expected_waves if wave["depth"] == depth), key=lambda wave: wave["workers"]) for depth in (32, 64)]
    if not isinstance(stress, list) or len(stress) != len(stress_selected) or any(not item.get("exact") or not item.get("forced_cap_completed") or item.get("max_new_tokens") != config["max_new_tokens"] or item.get("variants") != ["forced_cap"] or item.get("record_keys") != _expected_record_keys(stress_selected[index], ["forced_cap"], config) or item.get("records") != len(item.get("record_keys", [])) or any(item.get(key) != stress_selected[index][key] for key in identity) for index, item in enumerate(stress)):
        raise RuntimeError("worst-case forced-cap parity mismatch")
