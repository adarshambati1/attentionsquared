import json
from types import SimpleNamespace
import torch
from scripts.train_009_fixed_point_jump import losses
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_task_only_differs_from_fixed_point_run_only_by_objective_and_output():
 a=json.loads((ROOT/'configs/009_fixed_point_jump.json').read_text());b=json.loads((ROOT/'configs/010_task_only_jump.json').read_text())
 allowed={'protocol','fixed_point_weight','training_objective','output_root'}
 assert {k for k in a if a.get(k)!=b.get(k)}==allowed
 assert b['fixed_point_weight']==0.0 and b['optimizer_steps']==1000 and b['bottleneck_size']==256

def test_zero_weight_skips_consistency_core_during_training():
 source=(ROOT/'scripts/train_009_fixed_point_jump.py').read_text()
 assert "use_fixed=bool(c['fixed_point_weight']) or force_fixed_point_measurement" in source
 assert 'compute_fixed_point=use_fixed' in source

def test_task_only_loss_behaviorally_skips_core_and_equals_ce():
 class FakeWrapper:
  def __init__(self):self.compute=None
  def __call__(self,ids,h0,compute_fixed_point):
   self.compute=compute_fixed_point
   return SimpleNamespace(logits=torch.randn(1,ids.shape[1],64),fixed_point_image=None,latent_states=h0,core_steps_executed=0)
 wrapper=FakeWrapper();ids=torch.tensor([[1,2,3,4]]);h0=torch.randn(1,4,8);config={'fixed_point_weight':0.0,'fixed_point_epsilon':1e-12}
 output,task,fixed,total,count=losses(wrapper,ids,h0,2,config)
 assert wrapper.compute is False and output.core_steps_executed==0
 assert torch.equal(total,task) and fixed.item()==0 and count==2
