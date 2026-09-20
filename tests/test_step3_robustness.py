import json
from pathlib import Path
from src.evaluation.step3_robustness import clustered_paired_bootstrap,question_and_gold,score_dataset_answer,wilson_interval
from src.evaluation.step3_concurrency import production_waves,validate_gate_contract
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
 assert c['protocol']=='step3-huginn-robustness-concurrent-serial-v3'
 assert c['output_root']=='/workspace/step3_huginn_robustness_concurrent_v3'
 assert c['concurrent_workers_by_depth']=={'4':3,'8':3,'16':3,'32':3,'64':2}
 assert c['concurrency_gate_examples']==[0]
 assert c['concurrency_gate_short_tokens']==64
 assert c['concurrency_gate_policy']=='user-approved-proportional-v1'
 assert c['do_not_start_step_4'] is True
def test_concurrent_serial_waves_exactly_reconstruct_production():
 c=json.loads((ROOT/'configs/013_step3_huginn_robustness.json').read_text())
 waves=production_waves(c)
 assert [(w['wave_id'],w['members']) for w in waves]==[
  ('gsm8k-d4-wave0',['plain_d4','current_d4','shared_d4']),
  ('gsm8k-d8-wave0',['plain_d8','current_d8','projected_uniform_d8']),
  ('gsm8k-d8-wave1',['shared_d8','per_layer_d8']),
  ('gsm8k-d16-wave0',['plain_d16','current_d16','shared_d16']),
  ('gsm8k-d32-wave0',['plain_d32','current_d32','shared_d32']),
  ('gsm8k-d64-wave0',['plain_d64','current_d64']),
  ('gsm8k-d64-wave1',['shared_d64']),
  ('cross_dataset-d8-wave0',['plain_d8','current_d8','shared_d8']),
  ('cross_dataset-d16-wave0',['plain_d16']),
  ('cross_dataset-d64-wave0',['plain_d64'])]

def test_concurrency_gate_contract_is_fail_closed():
 c=json.loads((ROOT/'configs/013_step3_huginn_robustness.json').read_text());waves=production_waves(c)
 def entry(w,variant,tokens):
  keys=[{'condition':condition,'variant':variant,'example_id':example_id,'seed_index':0} for condition in w['members'] for example_id in c['concurrency_gate_examples']]
  return {**w,'variants':[variant],'max_new_tokens':tokens,'records':len(keys),'record_keys':keys,'exact':True,'natural_stopping_observed':variant=='natural','forced_cap_completed':variant=='forced_cap'}
 selected=[max((w for w in waves if w['depth']==d),key=lambda w:w['workers']) for d in sorted({w['depth'] for w in waves})]
 stress=[max((w for w in waves if w['depth']==d),key=lambda w:w['workers']) for d in (32,64)]
 gate={'protocol':c['protocol']+'-concurrency-gate','status':'pass','gate_policy':c['concurrency_gate_policy'],'production_wave_manifest':waves,'all_exact':True,'correctness_gate_sha256':'gate','checkpoint_sha256_before':c['checkpoint_sha256'],'checkpoint_sha256_after':c['checkpoint_sha256'],'git_commit':'commit','config_sha256':'config','hardware':c['hardware'],'tiers':{'all_waves_short':[entry(w,'forced_cap',c['concurrency_gate_short_tokens']) for w in waves],'natural_stop_by_depth':[entry(w,'natural',c['max_new_tokens']) for w in selected],'worst_case_forced_cap':[entry(w,'forced_cap',c['max_new_tokens']) for w in stress]}}
 validate_gate_contract(gate,config=c,commit='commit',config_sha256='config',correctness_gate_sha256='gate',hardware=c['hardware'])
 gate['tiers']['all_waves_short'][0]['exact']=False
 try:validate_gate_contract(gate,config=c,commit='commit',config_sha256='config',correctness_gate_sha256='gate',hardware=c['hardware'])
 except RuntimeError:pass
 else:raise AssertionError('mutated gate was accepted')

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
