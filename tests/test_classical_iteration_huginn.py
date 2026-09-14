import json
from pathlib import Path
import torch
from torch import nn
from src.models.classical_iteration_huginn import ClassicalIterationHuginn

ROOT=Path(__file__).resolve().parents[1]

def test_anderson_single_iterate_is_ordinary_update():
 f=torch.randn(1,3,4);h=torch.randn_like(f)
 mixed,fallbacks=ClassicalIterationHuginn._anderson_mix([f],[f-h],window=3,ridge=1e-4)
 assert torch.equal(mixed,f) and fallbacks==0

def test_anderson_regularized_coefficients_are_finite():
 h0=torch.zeros(1,2,5);f0=torch.ones_like(h0);f1=torch.full_like(h0,2)
 mixed,fallbacks=ClassicalIterationHuginn._anderson_mix([f0,f1],[f0-h0,f1-f0],window=3,ridge=1e-4)
 assert torch.isfinite(mixed).all() and fallbacks==0

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
