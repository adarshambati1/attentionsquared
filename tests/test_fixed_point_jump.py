import json
from pathlib import Path
from types import SimpleNamespace
import torch
from src.models.fixed_point_jump_huginn import FixedPointJumpPredictor
from scripts.train_009_fixed_point_jump import fp_loss
ROOT=Path(__file__).resolve().parents[1]

def test_zero_initialized_predictor_is_exact_residual_identity():
 model=FixedPointJumpPredictor(12,4);h0=torch.randn(2,3,12);x=torch.randn_like(h0)
 assert torch.equal(model(h0,x),h0)

def test_fixed_point_loss_is_aggregate_relative_residual_on_answer_positions():
 h=torch.ones(1,5,2);f=h.clone();f[:,2:4]+=1
 out=SimpleNamespace(latent_states=h,fixed_point_image=f,core_steps_executed=1)
 # answer_start=3 selects latent positions 2 and 3 for two target tokens.
 assert torch.allclose(fp_loss(out,3,2,1e-12),torch.tensor(1.0))

def test_fixed_protocol_has_no_sweep_or_endpoint_imitation():
 c=json.loads((ROOT/'configs/009_fixed_point_jump.json').read_text())
 assert c['bottleneck_size']==256 and c['fixed_point_weight']==0.1
 assert c['predictor_initialization_seed']==90091 and c['training_shuffle_seed']==9009
 assert c['train_ids']==[0,2249] and c['validation_ids']==[2250,2499] and c['test_ids']==[0,249]
 assert c['selection_criterion'].startswith('lowest full validation answer-token CE')
 model=(ROOT/'src/models/fixed_point_jump_huginn.py').read_text().lower();train=(ROOT/'scripts/train_009_fixed_point_jump.py').read_text().lower()
 assert 'h16' not in model+train and 'h64' not in model+train
 assert 'num_steps' not in model

def test_training_uses_functional_task_and_fixed_point_losses():
 source=(ROOT/'scripts/train_009_fixed_point_jump.py').read_text()
 assert "task+c['fixed_point_weight']*fixed" in source
 assert 'answer_cross_entropy' in source
 assert "compute_fixed_point=use_fixed" in source
 assert "best=val['answer_token_cross_entropy']" in source

def test_evaluation_resumes_only_contiguous_bound_records():
 source=(ROOT/'scripts/evaluate_009_fixed_point_jump.py').read_text()
 assert "r.get('example_id')!=expected_id" in source
 assert "r.get('config_sha256')!=config_sha" in source
 assert "r.get('checkpoint_sha256')!=best_sha" in source
 assert "range(c['test_ids'][0]+len(records)" in source
