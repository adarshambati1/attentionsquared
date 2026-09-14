import json
from pathlib import Path
import pytest
import torch
from torch import nn
from src.models.classical_iteration_huginn import ClassicalIterationHuginn

ROOT=Path(__file__).resolve().parents[1]

def test_anderson_single_iterate_is_ordinary_update():
 f=torch.randn(1,3,4);h=torch.randn_like(f)
 mixed,fallbacks,attempts,singular,nonfinite=ClassicalIterationHuginn._anderson_mix([f],[f-h],window=3,ridge=1e-4)
 assert torch.equal(mixed,f) and (fallbacks,attempts,singular,nonfinite)==(0,0,0,0)

def test_anderson_regularized_coefficients_are_finite():
 h0=torch.zeros(1,2,5);f0=torch.ones_like(h0);f1=torch.full_like(h0,2)
 mixed,fallbacks,attempts,singular,nonfinite=ClassicalIterationHuginn._anderson_mix([f0,f1],[f0-h0,f1-f0],window=3,ridge=1e-4)
 assert torch.isfinite(mixed).all()
 assert (fallbacks,attempts,singular,nonfinite)==(0,2,0,0)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA autocast')
def test_anderson_solve_stays_fp32_inside_bf16_autocast():
 h0=torch.randn(1,2,32,device='cuda');f0=torch.randn_like(h0);f1=torch.randn_like(h0)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  mixed,fallbacks,attempts,singular,nonfinite=ClassicalIterationHuginn._anderson_mix([f0,f1],[f0-h0,f1-f0],window=2,ridge=1e-4)
 assert mixed.dtype==torch.float32 and torch.isfinite(mixed).all()
 assert (fallbacks,attempts,singular,nonfinite)==(0,2,0,0)

def test_config_freezes_small_validation_sweep():
 c=json.loads((ROOT/'configs/008_classical_iteration_d8.json').read_text())
 assert c['depth']==8
 assert c['halpern_schedule']=='alpha_k=1/(k+2)'
 assert c['heavy_ball_momentum_candidates']==[0.05,0.1,0.2]
 assert c['anderson_window_candidates']==[2,3,4]
 assert c['anderson_ridge']==1e-4
 assert c['validation_ids']==[2250,2499] and c['test_ids']==[0,249]

def test_runner_has_no_training_and_reports_required_diagnostics():
 source=(ROOT/'scripts/run_008_classical_iteration_d8.py').read_text()
 assert 'optimizer' not in source.lower()
 assert 'loss.backward' not in source
 assert 'relative_update_norm' in source and 'cosine_to_final_state' in source
 assert 'anderson_fallbacks' in source and 'degeneration_rate' in source
 assert 'plain_logits_exact' in source and 'cached_full_argmax_exact' in source

def test_fp32_amendment_requires_successful_solves():
 model_source=(ROOT/'src/models/classical_iteration_huginn.py').read_text()
 runner_source=(ROOT/'scripts/run_008_anderson_fp32_amendment.py').read_text()
 assert 'enabled=False' in model_source and 'torch.linalg.solve_ex' in model_source
 assert "full.anderson_successful_solves!=expected" in runner_source
 assert "implementation_exceptions_are_fatal" in runner_source
