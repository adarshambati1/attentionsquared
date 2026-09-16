import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_depth_protocol_is_exact_and_stops_after_step2():
 c=json.loads((ROOT/'configs/012_depth_regression.json').read_text())
 assert c['depths']==[4,8,16,32,64,128,256,512]
 assert c['test_ids']==[0,249] and c['max_new_tokens']==1024
 assert c['do_not_start_step_3'] is True and c['step']=='2-of-11'

def test_runner_preserves_all_requested_fields_and_outputs():
 s=(ROOT/'scripts/run_012_depth_regression.py').read_text()
 for field in ('generated_text','extracted_answer','correct','generated_tokens','hit_max_new_tokens','degeneration','generation_latency_seconds','fixed_point_residual'):
  assert field in s
 for name in ('accuracy_vs_depth.png','residual_vs_depth.png','correctness_transitions.png','correctness_heatmap.png','raw_results.json','raw_results.csv'):
  assert name in s
 for transition in ('wrong_to_correct','correct_to_wrong','correct_to_correct','wrong_to_wrong'):
  assert transition in s

def test_plain_depth_model_has_no_trainable_additions():
 s=(ROOT/'src/models/plain_depth_huginn.py').read_text()
 assert 'nn.Parameter' not in s and 'requires_grad_(False)' in s
 assert 'compute_fixed_point_residual' in s and 'core_block_forward(state,x,frequencies,None,None,index,depth)' in s
