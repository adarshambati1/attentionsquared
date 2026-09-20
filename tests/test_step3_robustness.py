import json
from pathlib import Path
from src.evaluation.step3_robustness import clustered_paired_bootstrap,question_and_gold,score_dataset_answer,wilson_interval
from src.training.recurtrace_protocol import sample_loop_depths
ROOT=Path(__file__).resolve().parents[1]
def test_frozen_step3_huginn_scope():
 c=json.loads((ROOT/'configs/013_step3_huginn_robustness.json').read_text())
 assert c['h0_seed_indices']==[0,1,2]
 assert c['gsm8k']['ids']==[0,249]
 assert c['cross_dataset_conditions']==[['plain',8],['current',8],['shared',8],['plain',16],['plain',64]]
 required_3a={('plain',8),('current',8),('projected_uniform',8),('shared',8),('per_layer',8),('plain',16),('plain',64)}
 assert required_3a<=set(map(tuple,c['gsm8k_conditions']))
 expected_transfer={(m,d) for m in ('plain','current','shared') for d in (4,8,16,32,64)}
 assert set(map(tuple,c['depth_transfer_conditions']))==expected_transfer and expected_transfer<=set(map(tuple,c['gsm8k_conditions']))
 assert c['do_not_start_step_4'] is True
def test_frozen_step3d_design_and_training_interlock():
 c=json.loads((ROOT/'configs/013_step3_qwen_recurtrace.json').read_text())
 assert c['evaluation_variants']==['plain','shared_current','shared_history','per_layer_current','recurtrace']
 assert c['causal_pairs']==[['shared_current','shared_history'],['per_layer_current','recurtrace']]
 assert c['loop_layers_zero_indexed_inclusive']==[12,14]
 assert c['primary_evaluation_loop_count']==2 and c['secondary_diagnostic_loop_count']==4
 assert c['training_launch_allowed'] is False and c['halting_head'] is False and c['do_not_start_step_4'] is True

def test_recurtrace_depth_schedule_is_deterministic_bounded_and_heavy_tailed():
 a=sample_loop_depths(steps=10000,seed=7);b=sample_loop_depths(steps=10000,seed=7)
 assert a==b and min(a)>=1 and max(a)<=8
 assert 3.7<sum(a)/len(a)<4.1 and sum(x>=6 for x in a)/len(a)>.15

def test_dataset_fields_and_numeric_scoring():
 assert question_and_gold('svamp',{'question_concat':'q','Answer':'2'})==('q','2')
 assert question_and_gold('math500',{'problem':'p','answer':'x'})==('p','x')
 assert score_dataset_answer('svamp','Answer: 2','2',hit_max_new_tokens=False)['correct']
 assert score_dataset_answer('svamp','Answer: 2','2',hit_max_new_tokens=True)['correct']
 assert not score_dataset_answer('svamp','work 1 then 2','2',hit_max_new_tokens=True)['correct']
 assert score_dataset_answer('math500','Thus $\\boxed{\\frac{1}{2}}$.','\\frac{1}{2}',hit_max_new_tokens=False)['correct']
 assert not score_dataset_answer('math500','work 1 then 2','2',hit_max_new_tokens=True)['correct']
 assert not score_dataset_answer('math500','Earlier $\\boxed{3}$; final answer 7','7',hit_max_new_tokens=True)['correct']
 assert score_dataset_answer('math500','Thus $\\boxed{\\frac{1}{2}}$','\\frac{1}{2}',hit_max_new_tokens=True)['correct']
def test_wilson_and_clustered_paired_bootstrap():
 lo,hi=wilson_interval(5,10);assert lo<.5<hi
 a=[{'example_id':i,'seed_index':s,'correct':True} for i in range(3) for s in range(3)]
 b=[{'example_id':i,'seed_index':s,'correct':False} for i in range(3) for s in range(3)]
 r=clustered_paired_bootstrap({'a':a,'b':b},[('a','b')],samples=100,seed=1)
 assert r['paired_delta_cluster_bootstrap_95ci']['a_minus_b']==[1.0,1.0]
