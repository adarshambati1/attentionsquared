import json
from pathlib import Path
import torch
from src.models.perturbed_huginn import token_relative_noise
ROOT=Path(__file__).resolve().parents[1]

def test_token_relative_noise_has_frozen_norm_and_shared_direction():
 state=torch.randn(1,4,32);positions=torch.arange(4).unsqueeze(0);kwargs=dict(example_id=7,perturb_depth=8,seed_index=2,base_seed=11011,token_positions=positions)
 small,relative=token_relative_noise(state,sigma=.01,**kwargs);large,_=token_relative_noise(state,sigma=.1,**kwargs);repeat,_=token_relative_noise(state,sigma=.01,**kwargs)
 assert torch.equal(small,repeat)
 assert torch.allclose(relative,torch.full_like(relative,.01),atol=1e-6,rtol=1e-5)
 cosine=torch.nn.functional.cosine_similarity(small-state,large-state,dim=-1)
 assert torch.all(cosine>.99999)

def test_perturbation_is_realized_in_fp32_even_from_bf16_input():
 state=torch.randn(1,3,64).bfloat16();positions=torch.arange(3).unsqueeze(0);out,relative=token_relative_noise(state,sigma=.01,example_id=1,perturb_depth=4,seed_index=0,base_seed=11011,token_positions=positions)
 assert out.dtype==torch.float32
 assert torch.allclose(relative,torch.full_like(relative,.01),atol=1e-6,rtol=1e-5)

def test_padding_positions_are_not_perturbed():
 state=torch.randn(1,3,16);mask=torch.tensor([[True,False,True]]);out,_=token_relative_noise(state,sigma=.1,example_id=0,perturb_depth=4,seed_index=0,base_seed=11011,token_positions=torch.arange(3).unsqueeze(0),token_mask=mask)
 assert torch.equal(out[:,1],state[:,1]) and not torch.equal(out[:,0],state[:,0])

def test_protocol_is_fixed_and_parameter_free():
 c=json.loads((ROOT/'configs/011_perturb_recover.json').read_text())
 assert c['perturb_depths']==[4,8] and c['noise_strengths']==[.01,.05,.1] and c['perturbation_seed_indices']==[0,1,2]
 assert c['clean_depth']==32 and c['functional_generation_perturb_depths']==[8]
 source=(ROOT/'src/models/perturbed_huginn.py').read_text()
 assert 'nn.Parameter' not in source and 'requires_grad_(False)' in source

def test_optional_h4_generation_followup_fills_complete_table():
 c=json.loads((ROOT/'configs/011_h4_generation_followup.json').read_text())
 assert c['perturb_depth']==4 and c['noise_strengths']==[.01,.05,.1] and c['perturbation_seed_indices']==[0,1,2]
 source=(ROOT/'scripts/run_011_h4_generation_followup.py').read_text()
 assert "base_root/'complete_table.json'" in source and "perturb_depth=follow['perturb_depth']" in source

def test_primary_metric_and_native_resume_are_explicit():
 source=(ROOT/'scripts/run_011_perturb_recover_latent.py').read_text()
 assert 'error_normalized_to_initial' in source and 'per_step_contraction' in source
 assert 'core_block_forward(state,x,frequencies,None,None,index,depth)' in source
 assert "range(max_depth)" in source
