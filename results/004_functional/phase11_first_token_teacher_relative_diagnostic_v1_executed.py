import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/attentionsquared")
from scripts import run_004_functional_overfit_gate_v3 as runner
from src.training.functional_objective import freeze_module
from src.training.functional_protocol import Attention2Architecture, load_shared_initialization
from transformers import AutoModelForCausalLM

ATTEMPT = Path("/workspace/.functional_overfit_v3.attempt-a8055cc452af48e1b5741ff06e0e1674")
OUTPUT = Path("/workspace/functional_protocol/phase11_first_token_teacher_relative_diagnostic_v1.json")
if OUTPUT.exists():
    raise FileExistsError(OUTPUT)

device = torch.device("cuda")
examples = runner.load_examples(device)
huginn = AutoModelForCausalLM.from_pretrained(
    runner.MODEL_ID,
    revision=runner.MODEL_REVISION,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
).cuda()
freeze_module(huginn)
a2 = load_shared_initialization(
    runner.INITIALIZATION, rounds=4, architecture=Attention2Architecture()
).cuda().eval()
checkpoint = torch.load(ATTEMPT / "model.pt", map_location="cpu", weights_only=True)
a2.load_state_dict(checkpoint["state_dict"], strict=True)

rows = []
kl_sum = 0.0
argmax_equal = 0
teacher_target_top1 = 0
student_target_top1 = 0
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for example in examples:
        teacher_logits = runner.transient_teacher_logits(huginn, example)
        output, _, _ = a2(
            example["h0"], example["x"], token_mask=example["attention_mask"]
        )
        student_logits = runner.coda_logits(
            huginn, example, output[:, 15].to(torch.bfloat16)
        )
        position = example["answer_start"] - 1
        target = int(example["input_ids"][0, example["answer_start"]].item())
        teacher_first = teacher_logits[0, position].float()
        student_first = student_logits[0, position].float()
        teacher_probs = torch.softmax(teacher_first, -1)
        student_probs = torch.softmax(student_first, -1)
        first_kl = float(
            F.kl_div(torch.log_softmax(student_first, -1), teacher_probs, reduction="sum").item()
        )
        teacher_argmax = int(teacher_first.argmax().item())
        student_argmax = int(student_first.argmax().item())
        teacher_probability = float(teacher_probs[target].item())
        student_probability = float(student_probs[target].item())
        kl_sum += first_kl
        argmax_equal += int(teacher_argmax == student_argmax)
        teacher_target_top1 += int(teacher_argmax == target)
        student_target_top1 += int(student_argmax == target)
        rows.append({
            "example_id": example["example_id"],
            "target_token_id": target,
            "teacher_target_probability": teacher_probability,
            "student_target_probability": student_probability,
            "student_minus_teacher_target_probability": student_probability - teacher_probability,
            "teacher_argmax_token_id": teacher_argmax,
            "student_argmax_token_id": student_argmax,
            "teacher_student_argmax_equal": teacher_argmax == student_argmax,
            "teacher_first_distribution_to_student_kl": first_kl,
        })

mean_teacher = sum(row["teacher_target_probability"] for row in rows) / len(rows)
mean_student = sum(row["student_target_probability"] for row in rows) / len(rows)
result = {
    "protocol": "phase11-first-token-teacher-relative-diagnostic-v1",
    "status": "diagnostic_complete",
    "source_attempt": str(ATTEMPT),
    "model_sha256": "5fb7d0d467cb322a235c1293af0f9e9bd65249d32820fb0086e914b5e49307f0",
    "optimizer_steps": 0,
    "example_count": len(rows),
    "mean_teacher_target_probability": mean_teacher,
    "mean_student_target_probability": mean_student,
    "mean_student_minus_teacher_target_probability": mean_student - mean_teacher,
    "mean_teacher_first_distribution_to_student_kl": kl_sum / len(rows),
    "teacher_student_argmax_agreement_count": argmax_equal,
    "teacher_target_top1_count": teacher_target_top1,
    "student_target_top1_count": student_target_top1,
    "absolute_point_five_gate_was_met": mean_student >= 0.5,
    "per_example": rows,
}
OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
