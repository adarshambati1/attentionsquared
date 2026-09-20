"""Shared Step-3 dataset/scoring and paired-statistics helpers."""
from __future__ import annotations
import math,random
from typing import Any
from src.evaluation.correctness import score_generation

def question_and_gold(dataset_name:str,item:dict)->tuple[str,str]:
 if dataset_name=='gsm8k':return item['question'],item['answer']
 if dataset_name=='svamp':return item['question_concat'],item['Answer']
 if dataset_name=='math500':return item['problem'],item['answer']
 raise ValueError(dataset_name)

def _last_boxed_expression(text:str)->str|None:
 starts=[];cursor=0
 while True:
  index=text.find('\\boxed{',cursor)
  if index<0:break
  starts.append(index);cursor=index+7
 if not starts:return None
 for index in reversed(starts):
  depth=0
  for end in range(index+6,len(text)):
   if text[end]=='{':depth+=1
   elif text[end]=='}':
    depth-=1
    if depth==0:return text[index:end+1]
 return None

def score_dataset_answer(dataset_name:str,text:str,gold:str,*,hit_max_new_tokens:bool)->dict[str,Any]:
 if dataset_name in {'gsm8k','svamp'}:
  reference=gold if dataset_name=='gsm8k' else f'#### {gold}'
  score=score_generation(text,reference,hit_max_new_tokens=hit_max_new_tokens)
  return {'correct':score.correct,'predicted_answer':score.predicted_answer,'gold_answer':score.gold_answer}
 from math_verify import parse,verify
 gold_parsed=parse(f'\\boxed{{{gold}}}')
 candidate=_last_boxed_expression(text) if hit_max_new_tokens else text
 prediction=parse(candidate) if candidate is not None else []
 correct=bool(gold_parsed and prediction and verify(gold_parsed,prediction))
 return {'correct':correct,'predicted_answer':str(prediction[0]) if prediction else None,'gold_answer':gold}

def wilson_interval(successes:int,total:int,z:float=1.959963984540054)->list[float]:
 if total<=0:raise ValueError('total must be positive')
 p=successes/total;den=1+z*z/total;center=(p+z*z/(2*total))/den;half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
 return [center-half,center+half]

def clustered_paired_bootstrap(records_by_model:dict[str,list[dict]],comparisons:list[tuple[str,str]],*,samples:int,seed:int)->dict:
 models=list(records_by_model);keys={(r['example_id'],r['seed_index']) for r in records_by_model[models[0]]};example_ids=sorted({k[0] for k in keys});seeds=sorted({k[1] for k in keys});lookup={m:{(r['example_id'],r['seed_index']):int(r['correct']) for r in rs} for m,rs in records_by_model.items()}
 if any(len(lookup[m])!=len(records_by_model[m]) for m in models):raise ValueError('duplicate paired record key')
 if any(set(v)!=keys for v in lookup.values()):raise ValueError('paired record keys differ')
 rng=random.Random(seed);draws={m:[] for m in models};delta_draws={f'{a}_minus_{b}':[] for a,b in comparisons}
 for _ in range(samples):
  chosen=[example_ids[rng.randrange(len(example_ids))] for _ in example_ids];den=len(chosen)*len(seeds);vals={m:sum(lookup[m][(i,s)] for i in chosen for s in seeds)/den for m in models}
  for m in models:draws[m].append(vals[m])
  for a,b in comparisons:delta_draws[f'{a}_minus_{b}'].append(vals[a]-vals[b])
 def interval(values):
  values=sorted(values);return [values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]]
 return {'model_accuracy_cluster_bootstrap_95ci':{m:interval(v) for m,v in draws.items()},'paired_delta_cluster_bootstrap_95ci':{k:interval(v) for k,v in delta_draws.items()},'samples':samples,'cluster':'example_id','seed':seed}
