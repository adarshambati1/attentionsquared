"""Shared correctness primitives for Attention Squared experiments.

This module contains protocol-level mechanics only.  It deliberately does not
choose dataset splits, model topology, objectives, or degeneration review
outcomes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import random
import re
import time
from typing import Any, Callable, Iterator, Sequence, TypeVar


STOP_STRINGS: tuple[str, ...] = ("<|end_text|>", "<|end_turn|>")

_NUMERIC = r"[-+]?\$?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?"
_MARKED_NUMBER = re.compile(rf"####\s*({_NUMERIC})")
# This is deliberately one expression with the exact trusted Experiment 001
# 48-character window. Changing it changes the scorer protocol.
_EXPLICIT_NUMBER = re.compile(
    rf"(?:answer(?: is|:)|final answer(?: is|:)|therefore(?:,|\s+))"
    rf"[^\d$+-]{{0,48}}({_NUMERIC})",
    re.IGNORECASE,
)
_FALLBACK_NUMBER = re.compile(_NUMERIC)


def build_chat_prompt(tokenizer: Any, question: str, system_instruction: str) -> str:
    """Construct the canonical system/user prompt with the native chat template."""
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def tokenize_prompt(tokenizer: Any, prompt: str, **kwargs: Any) -> dict[str, Any]:
    """Tokenize a rendered chat prompt without adding a second set of tokens."""
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, **kwargs)
    encoded.pop("token_type_ids", None)
    return encoded


def generation_config_kwargs(tokenizer: Any, max_new_tokens: int) -> dict[str, Any]:
    """Canonical greedy-generation settings shared by Exp1/3/4."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    return {
        "max_new_tokens": max_new_tokens,
        "stop_strings": list(STOP_STRINGS),
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
        "return_legacy_cache": False,
        "eos_token_id": tokenizer.eos_token_id,
        # Preserve the historical fallback when pad_token_id is None or zero.
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }


@dataclass(frozen=True)
class AnswerBounds:
    """Half-open answer-token bounds in a complete token sequence."""

    answer_start: int
    valid_end: int
    sequence_length: int

    def __post_init__(self) -> None:
        if not 1 <= self.answer_start < self.valid_end <= self.sequence_length:
            raise ValueError(
                "expected 1 <= answer_start < valid_end <= sequence_length, got "
                f"{self.answer_start}, {self.valid_end}, {self.sequence_length}"
            )

    @property
    def token_count(self) -> int:
        return self.valid_end - self.answer_start

    @property
    def answer_slice(self) -> slice:
        return slice(self.answer_start, self.valid_end)

    @property
    def causal_logit_slice(self) -> slice:
        """Positions whose logits predict the answer tokens."""
        return slice(self.answer_start - 1, self.valid_end - 1)


def answer_bounds(
    answer_start: int, sequence_length: int, valid_end: int | None = None
) -> AnswerBounds:
    return AnswerBounds(
        int(answer_start),
        int(sequence_length if valid_end is None else valid_end),
        int(sequence_length),
    )


def aligned_answer_logits_and_targets(
    logits: Any, input_ids: Any, bounds: AnswerBounds
) -> tuple[Any, Any]:
    """Return causal logits and the answer tokens they predict.

    For answer tokens ``input_ids[start:end]``, this selects logits
    ``logits[start-1:end-1]``. Leading batch dimensions are preserved.
    """
    if logits.shape[-2] != bounds.sequence_length:
        raise ValueError("logit sequence length does not match answer bounds")
    if input_ids.shape[-1] != bounds.sequence_length:
        raise ValueError("input_ids sequence length does not match answer bounds")
    return (
        logits[..., bounds.causal_logit_slice, :],
        input_ids[..., bounds.answer_slice],
    )


def answer_prediction_mask(bounds: AnswerBounds, *, device: Any = None) -> Any:
    """Boolean mask over logit positions; only causal answer predictors are true."""
    import torch

    mask = torch.zeros(bounds.sequence_length, dtype=torch.bool, device=device)
    mask[bounds.causal_logit_slice] = True
    return mask


def _normalize_number(value: str) -> str:
    return value.replace(",", "").replace("$", "").strip()


def extract_answer(text: str, *, allow_fallback: bool = True) -> str | None:
    """Extract the last marked, explicit, or (if allowed) fallback number."""
    text = text.replace("\u202f", "")
    marked = _MARKED_NUMBER.findall(text)
    if marked:
        return _normalize_number(marked[-1])
    explicit = [
        (match.end(), match.group(1))
        for match in _EXPLICIT_NUMBER.finditer(text)
    ]
    if explicit:
        return _normalize_number(max(explicit)[1])
    if not allow_fallback:
        return None
    fallback = _FALLBACK_NUMBER.findall(text)
    return _normalize_number(fallback[-1]) if fallback else None


@dataclass(frozen=True)
class GenerationScore:
    gold_answer: str | None
    predicted_answer: str | None
    correct: bool
    hit_max_new_tokens: bool
    fallback_allowed: bool


def score_generation(
    generated_text: str,
    gold_text: str,
    *,
    hit_max_new_tokens: bool,
    allow_fallback_on_cap: bool = False,
) -> GenerationScore:
    """Authoritative exact-match scorer, including the cap fallback policy."""
    fallback_allowed = allow_fallback_on_cap or not hit_max_new_tokens
    gold = extract_answer(gold_text)
    predicted = extract_answer(generated_text, allow_fallback=fallback_allowed)
    return GenerationScore(
        gold_answer=gold,
        predicted_answer=predicted,
        correct=predicted is not None and predicted == gold,
        hit_max_new_tokens=hit_max_new_tokens,
        fallback_allowed=fallback_allowed,
    )


@dataclass(frozen=True)
class GenerationStatus:
    generated_tokens: int
    ended_naturally: bool
    hit_max_new_tokens: bool


def stop_token_ids(tokenizer: Any) -> tuple[int | None, ...]:
    """Canonical terminal token IDs corresponding to the shared stop policy."""
    return (
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids(STOP_STRINGS[0]),
        tokenizer.convert_tokens_to_ids(STOP_STRINGS[1]),
    )


def generation_status(
    generated_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    stop_token_ids: Sequence[int | None],
) -> GenerationStatus:
    """Classify a generation cap without treating the cap itself as EOS."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    ids = [int(token) for token in generated_token_ids]
    stops = {int(token) for token in stop_token_ids if token is not None and int(token) >= 0}
    ended = bool(ids and ids[-1] in stops)
    return GenerationStatus(len(ids), ended, len(ids) >= max_new_tokens and not ended)


def find_repetition_onset(
    token_ids: Sequence[int], *, chunk_size: int = 12, repetitions: int = 3
) -> int | None:
    """Find the first immediately repeated token chunk; return no review decision."""
    if chunk_size <= 0 or repetitions < 2:
        raise ValueError("chunk_size must be positive and repetitions at least two")
    tokens = list(token_ids)
    width = chunk_size * repetitions
    for index in range(len(tokens) - width + 1):
        chunk = tokens[index : index + chunk_size]
        if all(
            tokens[index + repeat * chunk_size : index + (repeat + 1) * chunk_size]
            == chunk
            for repeat in range(1, repetitions)
        ):
            return index
    return None


@dataclass(frozen=True)
class DegenerationMetadata:
    """Serializable cap/degeneration metadata; review decisions stay explicit."""

    truncated: bool
    pathological: bool
    degeneration_onset: int | None
    valid_end: int
    review_decision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def degeneration_metadata(
    bounds: AnswerBounds,
    *,
    hit_max_new_tokens: bool,
    degeneration_onset: int | None = None,
    reviewed_valid_end: int | None = None,
    review_decision: str = "unreviewed",
) -> DegenerationMetadata:
    """Validate and record, but never invent, a degeneration truncation decision."""
    valid_end = bounds.valid_end if reviewed_valid_end is None else int(reviewed_valid_end)
    reviewed = answer_bounds(bounds.answer_start, bounds.sequence_length, valid_end)
    if reviewed.valid_end > bounds.valid_end:
        raise ValueError("reviewed valid_end cannot extend the existing valid region")
    if degeneration_onset is not None:
        onset_end = bounds.answer_start + int(degeneration_onset)
        if not bounds.answer_start <= onset_end <= bounds.sequence_length:
            raise ValueError("degeneration onset is outside the answer continuation")
    pathological = degeneration_onset is not None
    if reviewed.valid_end < bounds.valid_end and review_decision == "unreviewed":
        raise ValueError("shortening valid_end requires an explicit review decision")
    return DegenerationMetadata(
        truncated=bool(hit_max_new_tokens),
        pathological=pathological,
        degeneration_onset=degeneration_onset,
        valid_end=reviewed.valid_end,
        review_decision=review_decision,
    )


def per_example_seed(
    base_seed: int, example_id: int, step: int = 0, *, seed_index: int = 0
) -> int:
    """Stable paired seed derived without process-global RNG state.

    Coordinates are validated and injectively packed over the supported
    protocol domain, so distinct supported tuples cannot collide.
    """
    if min(int(base_seed), int(example_id), int(step), int(seed_index)) < 0:
        raise ValueError("seed coordinates must be non-negative")
    coordinates = (int(base_seed), int(example_id), int(seed_index), int(step))
    limits = (SEED_BASE_LIMIT, SEED_EXAMPLE_LIMIT, SEED_INDEX_LIMIT, SEED_STEP_LIMIT)
    if any(value < 0 or value >= limit for value, limit in zip(coordinates, limits)):
        raise ValueError(
            "seed coordinates out of supported bounds: "
            f"base<{SEED_BASE_LIMIT}, example<{SEED_EXAMPLE_LIMIT}, "
            f"seed_index<{SEED_INDEX_LIMIT}, step<{SEED_STEP_LIMIT}"
        )
    base, example, index, refinement_step = coordinates
    # Mixed-radix packing is injective over the validated coordinate domain and
    # fits within 63 bits, which is accepted by Python/NumPy/Torch.
    return (
        (((base * SEED_EXAMPLE_LIMIT) + example) * SEED_INDEX_LIMIT + index)
        * SEED_STEP_LIMIT
        + refinement_step
    )


def torch_generator_for_example(
    base_seed: int,
    example_id: int,
    step: int = 0,
    *,
    seed_index: int = 0,
    device: Any = "cpu",
) -> Any:
    """Return a private deterministic generator without changing global RNG state."""
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(
        per_example_seed(base_seed, example_id, step, seed_index=seed_index)
    )
    return generator


def numpy_generator_for_example(
    base_seed: int, example_id: int, step: int = 0, *, seed_index: int = 0
) -> Any:
    """Return a private deterministic NumPy generator without global side effects."""
    import numpy as np

    return np.random.default_rng(
        per_example_seed(base_seed, example_id, step, seed_index=seed_index)
    )


def seed_everything(seed: int) -> int:
    """Seed global RNGs for new protocol code that explicitly requests it.

    Historical v1 artifacts retain their recorded provenance; these helpers do
    not rewrite metadata or existing artifacts.
    """
    import numpy as np
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def seed_for_example(
    base_seed: int, example_id: int, step: int = 0, *, seed_index: int = 0
) -> int:
    seed = per_example_seed(base_seed, example_id, step, seed_index=seed_index)
    return seed_everything(seed)


def token_kl_sum_and_count(
    student_logits: Any, teacher_logits: Any, mask: Any = None
) -> tuple[Any, Any]:
    """Return the KL numerator and valid-token count without local averaging."""
    import torch
    import torch.nn.functional as functional

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    per_token = functional.kl_div(
        torch.log_softmax(student_logits.float(), dim=-1),
        torch.softmax(teacher_logits.float(), dim=-1),
        reduction="none",
    ).sum(dim=-1)
    if mask is None:
        valid = torch.ones_like(per_token, dtype=torch.bool)
    else:
        valid = mask.to(device=per_token.device, dtype=torch.bool)
        try:
            valid = torch.broadcast_to(valid, per_token.shape)
        except RuntimeError as error:
            raise ValueError("KL mask is not broadcastable to token dimensions") from error
    count = valid.sum()
    if int(count.item()) == 0:
        raise ValueError("token-normalized KL requires at least one valid token")
    return per_token.masked_select(valid).sum(), count


def token_normalized_kl(student_logits: Any, teacher_logits: Any, mask: Any = None) -> Any:
    """Compute KL(p_teacher || p_student), averaged over all valid tokens."""
    numerator, count = token_kl_sum_and_count(student_logits, teacher_logits, mask)
    return numerator / count


@dataclass
class TokenKLAggregator:
    """Accumulate detached KL numerators/counts for a globally weighted mean."""

    numerator: float = 0.0
    count: int = 0

    def update(self, numerator: Any, count: Any) -> None:
        add_count = int(count.item() if hasattr(count, "item") else count)
        if add_count <= 0:
            raise ValueError("KL count must be positive")
        add_numerator = float(
            numerator.detach().item() if hasattr(numerator, "detach") else numerator
        )
        self.numerator += add_numerator
        self.count += add_count

    def update_logits(
        self, student_logits: Any, teacher_logits: Any, mask: Any = None
    ) -> None:
        self.update(*token_kl_sum_and_count(student_logits, teacher_logits, mask))

    @property
    def mean(self) -> float:
        if self.count == 0:
            raise ValueError("cannot report KL before any valid tokens were accumulated")
        return self.numerator / self.count


INVALID_FUNCTIONAL_V1_LABEL = (
    "INVALID_V1: off-by-one functional targets and/or incremental one-token "
    "Attention2 execution; preserved for audit only"
)
CORRECTED_V2_PROTOCOL = "experiment-004-corrected-v2"
CORRECTED_V2_PATH_MARKER = "corrected-v2"
SCORER_PROTOCOL = "exp001-exact-48-char-explicit-window-cap-safe-v1"
SEED_PROTOCOL = "injective-packed-coordinates-v3"
SEED_BASE_LIMIT = 1 << 24
SEED_EXAMPLE_LIMIT = 1 << 24
SEED_INDEX_LIMIT = 1 << 8
SEED_STEP_LIMIT = 1 << 7


def require_invalid_v1_opt_in(allowed: bool, script_name: str) -> None:
    """Make accidental execution of quarantined invalid-v1 pipelines fail closed."""
    if not allowed:
        raise RuntimeError(
            f"{script_name} is quarantined ({INVALID_FUNCTIONAL_V1_LABEL}). "
            "Pass --allow-invalid-v1 only for an explicitly labeled audit rerun."
        )


def require_legacy_artifact_opt_in(allowed: bool, script_name: str) -> None:
    """Prevent accidental mutation or partial resume of frozen legacy artifacts."""
    if not allowed:
        raise RuntimeError(
            f"{script_name} writes a frozen legacy artifact tree. "
            "Pass an explicit legacy-rebuild flag only for a fresh provenance-preserving repair; "
            "new work must use a separate corrected-v2 path."
        )


def corrected_v2_protocol_metadata() -> dict[str, Any]:
    """Immutable identifiers required on future corrected-v2 artifacts."""
    return {
        "protocol": CORRECTED_V2_PROTOCOL,
        "scorer_protocol": SCORER_PROTOCOL,
        "seed_protocol": SEED_PROTOCOL,
        "prompt_protocol": "native-chat-template-no-extra-special-tokens-v1",
        "generation_protocol": "greedy-exp001-stops-cap-safe-v1",
        "kl_protocol": "global-numerator-over-valid-token-count-v1",
    }


def ensure_new_artifact_root(path: Any) -> None:
    """Atomically create a fresh artifact root; partial legacy resume is forbidden."""
    from pathlib import Path

    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)


def ensure_audit_rerun_root(path: Any) -> None:
    """Create a separate, fresh root for an explicitly invalid audit rerun."""
    from pathlib import Path

    root = Path(path)
    if "audit-rerun" not in root.parts and "audit-rerun" not in root.name:
        raise ValueError("invalid-v1 audit reruns require a separate audit-rerun path")
    ensure_new_artifact_root(root)


def ensure_audit_rerun_output(path: Any) -> None:
    """Require an explicit audit-rerun path and refuse replacement."""
    from pathlib import Path

    output = Path(path)
    if "audit-rerun" not in output.parts and "audit-rerun" not in output.name:
        raise ValueError("invalid-v1 audit outputs require a separate audit-rerun path")
    ensure_new_output_path(output)


def write_json_exclusive(path: Any, value: Any) -> None:
    """Publish a JSON result with exclusive creation; never replace an artifact."""
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def prepare_corrected_v2_output(path: Any) -> None:
    """Validate a corrected-v2 location and persist its protocol manifest."""
    from pathlib import Path

    output = Path(path)
    raw_parts = [output, *output.parents]
    if any(part.is_symlink() for part in raw_parts if part.exists()):
        raise ValueError("corrected-v2 outputs may not traverse symlinks")
    output = output.resolve(strict=False)
    candidates = [output.parent, *output.parents]
    root = next((candidate for candidate in candidates if candidate.name == CORRECTED_V2_PATH_MARKER), None)
    if root is None:
        raise ValueError("corrected-v2 outputs require a corrected-v2 directory")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    if root.exists() and not (root / "protocol.json").exists() and any(root.iterdir()):
        raise ValueError("non-empty corrected-v2 root requires protocol metadata before resume")
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "protocol.json"
    expected = corrected_v2_protocol_metadata()
    if manifest.exists():
        actual = json.loads(manifest.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("corrected-v2 protocol manifest mismatch")
    else:
        try:
            write_json_exclusive(manifest, expected)
        except FileExistsError:
            actual = json.loads(manifest.read_text(encoding="utf-8"))
            if actual != expected:
                raise ValueError("corrected-v2 protocol manifest mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)


def guard_corrected_v2_artifact_root(path: Any, metadata: dict[str, Any] | None = None) -> None:
    """Reject corrected-v2 writes/resumes in an unmarked or mismatched root.

    The caller must use a visibly separate ``corrected-v2`` path. On resume it
    must also supply the existing protocol metadata for exact validation.
    """
    from pathlib import Path

    root = Path(path)
    if CORRECTED_V2_PATH_MARKER not in root.parts and CORRECTED_V2_PATH_MARKER not in root.name:
        raise ValueError("corrected-v2 artifacts require a separate corrected-v2 path")
    if root.exists() and not root.is_dir():
        raise ValueError("corrected-v2 artifact root must be a directory")
    if root.exists() and any(root.iterdir()) and metadata is None:
        raise ValueError("non-empty corrected-v2 root requires protocol metadata before resume")
    if metadata is not None:
        expected = corrected_v2_protocol_metadata()
        mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
        if mismatches:
            raise ValueError(f"corrected-v2 protocol mismatch: {mismatches}")


def sha256_file(path: Any) -> str:
    """Return a streaming SHA-256 checksum for artifact identity metadata."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_new_output_path(path: Any) -> None:
    """Fail closed rather than overwrite an existing scientific result."""
    from pathlib import Path

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)


def synchronize_cuda(device: Any = None) -> None:
    """Synchronize only when CUDA is available and the selected device is CUDA."""
    import torch

    selected = torch.device(device) if device is not None else None
    if torch.cuda.is_available() and (selected is None or selected.type == "cuda"):
        torch.cuda.synchronize(selected)


@dataclass
class Timing:
    seconds: float = 0.0


@contextmanager
def synchronized_cuda_timer(device: Any = None) -> Iterator[Timing]:
    """Wall-clock a block with CUDA synchronized immediately before and after."""
    result = Timing()
    synchronize_cuda(device)
    start = time.perf_counter()
    try:
        yield result
    finally:
        synchronize_cuda(device)
        result.seconds = time.perf_counter() - start


_T = TypeVar("_T")


def measure_synchronized(
    function: Callable[[], _T], *, device: Any = None, warmup: int = 0
) -> tuple[_T, float]:
    """Run optional unmeasured warmups, then one synchronized timed invocation."""
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    for _ in range(warmup):
        function()
    with synchronized_cuda_timer(device) as timing:
        result = function()
    return result, timing.seconds
