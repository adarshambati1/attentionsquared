import json
from pathlib import Path
from src.evaluation.step3_robustness import clustered_paired_bootstrap,question_and_gold,score_dataset_answer,wilson_interval
ROOT=Path(__file__).resolve().parents[1]
def test_frozen_step3_huginn_scope():
 c=json.loads((ROOT/'configs/013_step3_huginn_robustness.json').read_text())
 assert c['h0_seed_indices']==[0,1,2]
 assert c['gsm8k']['ids']==[0,249]
 assert c['cross_dataset_conditions']==[['plain',8],['current',8],['shared',8],['plain',16]]
 assert {d for m,d in c['gsm8k_conditions'] if m in {'plain','current','shared'}}=={4,8,16,32,64}
 assert c['do_not_start_step_4'] is True
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
