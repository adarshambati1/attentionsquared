#!/usr/bin/env python3
"""Create fail-closed Step 3A-3C paired tables after all Huginn evaluations."""
from __future__ import annotations
import hashlib,json,math,os,statistics,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from datasets import load_dataset
from transformers import AutoTokenizer
from src.evaluation.correctness import find_repetition_onset,generation_status,per_example_seed,stop_token_ids
from src.evaluation.step3_robustness import clustered_paired_bootstrap,score_dataset_answer
from src.evaluation.step3_concurrency import validate_correctness_gate_contract,validate_gate_contract
CONFIG=ROOT/'configs/013_step3_huginn_robustness.json'
def publish(path,value):
 data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
 if path.exists():
  if path.read_bytes()!=data:raise RuntimeError(f'existing artifact differs: {path}')
  return
 tmp=path.with_name(f'.{path.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp')
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 try:
  try:os.link(tmp,path)
  except FileExistsError:
   if path.read_bytes()!=data:raise RuntimeError(f'concurrent artifact differs: {path}')
  fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 finally:tmp.unlink(missing_ok=True)
def gold_answer(dataset,item):
 if dataset=='gsm8k':return str(item['answer'])
 if dataset=='svamp':return str(item['Answer'])
 if dataset=='math500':return str(item['answer'])
 raise ValueError(dataset)
def metrics(rs,seeds):
 by_seed={str(s):sum(r['correct'] for r in rs if r['seed_index']==s)/sum(r['seed_index']==s for r in rs) for s in seeds};return {'by_seed_accuracy':by_seed,'correct':sum(r['correct'] for r in rs),'total':len(rs),'pooled_accuracy':sum(r['correct'] for r in rs)/len(rs),'mean_seed_accuracy':sum(by_seed.values())/len(by_seed),'cap_hit_rate':sum(r['hit_max_new_tokens'] for r in rs)/len(rs),'degeneration_rate':sum(r['degeneration'] for r in rs)/len(rs),'mean_output_length':sum(r['generated_tokens'] for r in rs)/len(rs),'mean_generation_latency_seconds':sum(r['generation_latency_seconds'] for r in rs)/len(rs),'median_generation_latency_seconds':statistics.median(r['generation_latency_seconds'] for r in rs)}
def main():
 c=json.loads(CONFIG.read_text());root=Path(c['output_root']);dirty=subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
 if dirty:raise RuntimeError(f'tracked/untracked worktree dirty: {dirty}')
 commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();config_sha=hashlib.sha256(CONFIG.read_bytes()).hexdigest();correctness_gate_path=root/'correctness_gate.json';gate_sha=hashlib.sha256(correctness_gate_path.read_bytes()).hexdigest();correctness_gate=json.loads(correctness_gate_path.read_text());concurrency_gate_path=root/'concurrency_gate.json';concurrency_gate_sha=hashlib.sha256(concurrency_gate_path.read_bytes()).hexdigest();concurrency_gate=json.loads(concurrency_gate_path.read_text());checkpoint_set=hashlib.sha256(json.dumps(c['checkpoint_sha256'],sort_keys=True).encode()).hexdigest();summaries={}
 validate_correctness_gate_contract(correctness_gate,config=c,commit=commit,config_sha256=config_sha,hardware=c['hardware'])
 validate_gate_contract(concurrency_gate,config=c,commit=commit,config_sha256=config_sha,correctness_gate_sha256=gate_sha,hardware=c['hardware'])
 expected_conditions={'gsm8k':[f'{m}_d{d}' for m,d in c['gsm8k_conditions']],'svamp':[f'{m}_d{d}' for m,d in c['cross_dataset_conditions']],'math500':[f'{m}_d{d}' for m,d in c['cross_dataset_conditions']]}
 tokenizer=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);datasets={name:load_dataset(c[name]['dataset_id'],c[name]['config'],split=c[name]['split'],revision=c[name]['revision']) for name in ('gsm8k','svamp','math500')}
 all_records={}
 for dataset in ('gsm8k','svamp','math500'):
  s=json.loads((root/f'{dataset}_summary.json').read_text());spec=c[dataset];expected_per=(spec['ids'][1]-spec['ids'][0]+1)*len(c['h0_seed_indices'])
  if s.get('status')!='complete' or s.get('git_commit')!=commit or s.get('config_sha256')!=config_sha or s.get('checkpoint_sha256_before')!=c['checkpoint_sha256'] or s.get('checkpoint_sha256_after')!=c['checkpoint_sha256'] or s.get('checkpoint_set_sha256')!=checkpoint_set or s.get('gate_sha256')!=gate_sha or s.get('concurrency_gate_sha256')!=concurrency_gate_sha or s.get('hardware')!=c['hardware'] or sorted(s.get('conditions',{}))!=sorted(expected_conditions[dataset]) or s.get('records')!=expected_per*len(expected_conditions[dataset]):raise RuntimeError(f'{dataset} summary provenance/coverage mismatch')
  summaries[dataset]=s
  for name in expected_conditions[dataset]:
   model,depth_text=name.rsplit('_d',1);depth=int(depth_text);paths=sorted((root/'records'/dataset/name).glob('seed_*/*.json'));rs=[json.loads(p.read_text()) for p in paths];keys=[(r['example_id'],r['seed_index']) for r in rs];expected={(i,j) for i in range(spec['ids'][0],spec['ids'][1]+1) for j in c['h0_seed_indices']}
   if len(keys)!=len(set(keys)) or set(keys)!=expected:raise RuntimeError(f'{dataset}/{name} Cartesian coverage mismatch')
   for path,r in zip(paths,rs):
    expected_path=root/'records'/dataset/name/f"seed_{r['seed_index']}"/f"{r['example_id']:04d}.json";item=datasets[dataset][r['example_id']];status=generation_status(r['generated_token_ids'],max_new_tokens=c['max_new_tokens'],stop_token_ids=stop_token_ids(tokenizer));rescored=score_dataset_answer(dataset,r['generated_text'],gold_answer(dataset,item),hit_max_new_tokens=status.hit_max_new_tokens);repetition=find_repetition_onset(r['generated_token_ids'],chunk_size=c['degeneration_chunk_size'],repetitions=c['degeneration_repetitions'])
    fixed={'protocol':c['protocol']+'-record','dataset':dataset,'model':model,'depth':depth,'git_commit':commit,'config_sha256':config_sha,'hardware':c['hardware'],'checkpoint_sha256_before':c['checkpoint_sha256'],'gate_sha256':gate_sha,'concurrency_gate_sha256':concurrency_gate_sha,'checkpoint_set_sha256':checkpoint_set,'h0_seed':per_example_seed(c['h0_base_seed'],r['example_id'],step=0,seed_index=r['seed_index'])}
    if path!=expected_path or any(r.get(k)!=v for k,v in fixed.items()) or r['generated_text']!=tokenizer.decode(r['generated_token_ids'],skip_special_tokens=False) or any(r.get(k)!=v for k,v in rescored.items()) or r['generated_tokens']!=len(r['generated_token_ids']) or r['hit_max_new_tokens']!=status.hit_max_new_tokens or r['ended_naturally']!=status.ended_naturally or r['repetition_onset']!=repetition or r['degeneration']!=(repetition is not None) or not isinstance(r['used_full_prefix_fallback'],bool) or (r['used_full_prefix_fallback']!=(r['full_prefix_fallback_onset'] is not None)) or (r['full_prefix_fallback_onset'] is not None and (not isinstance(r['full_prefix_fallback_onset'],int) or not 1<=r['full_prefix_fallback_onset']<=r['generated_tokens'])) or not isinstance(r['generation_latency_seconds'],(int,float)) or not math.isfinite(r['generation_latency_seconds']) or r['generation_latency_seconds']<0 or not isinstance(r['peak_memory_bytes'],int) or r['peak_memory_bytes']<0:raise RuntimeError(f'{dataset}/{name} strict record validation failed: {path}')
   all_records[(dataset,name)]=rs
 def paired_block(dataset,names,comparisons,seed_offset):
  rec={name:all_records[(dataset,name)] for name in names};boot=clustered_paired_bootstrap(rec,comparisons,samples=c['bootstrap_samples'],seed=c['bootstrap_seed']+seed_offset);lookup={name:{(r['example_id'],r['seed_index']):r for r in rs} for name,rs in rec.items()};paired={}
  for a,b in comparisons:
   wins=sum(lookup[a][k]['correct'] and not lookup[b][k]['correct'] for k in lookup[a]);losses=sum(not lookup[a][k]['correct'] and lookup[b][k]['correct'] for k in lookup[a]);paired[f'{a}_minus_{b}']={'wins':wins,'losses':losses,'ties':len(lookup[a])-wins-losses,'delta_accuracy':metrics(rec[a],c['h0_seed_indices'])['pooled_accuracy']-metrics(rec[b],c['h0_seed_indices'])['pooled_accuracy'],'cluster_bootstrap_95ci':boot['paired_delta_cluster_bootstrap_95ci'][f'{a}_minus_{b}']}
  model_metrics={name:{**metrics(rs,c['h0_seed_indices']),'example_cluster_bootstrap_95ci':boot['model_accuracy_cluster_bootstrap_95ci'][name]} for name,rs in rec.items()};return model_metrics,paired
 gsm_names=['plain_d8','current_d8','projected_uniform_d8','shared_d8','per_layer_d8','plain_d16','plain_d64'];gsm_comparisons=[('current_d8','plain_d8'),('projected_uniform_d8','current_d8'),('shared_d8','current_d8'),('per_layer_d8','current_d8'),('shared_d8','plain_d8'),('per_layer_d8','plain_d8'),('plain_d16','plain_d8'),('plain_d64','plain_d8')];table1,paired1=paired_block('gsm8k',gsm_names,gsm_comparisons,0)
 table2={};paired2={}
 for offset,dataset in enumerate(('gsm8k','svamp','math500'),10):
  names=['plain_d8','current_d8','shared_d8','plain_d16','plain_d64'];vals,pairs=paired_block(dataset,names,[('current_d8','plain_d8'),('shared_d8','current_d8'),('shared_d8','plain_d8'),('plain_d16','plain_d8'),('plain_d64','plain_d8')],offset);table2[dataset]=vals;paired2[dataset]=pairs
 table3={};paired3={}
 for offset,depth in enumerate((4,8,16,32,64),20):
  names=[f'plain_d{depth}',f'current_d{depth}',f'shared_d{depth}'];vals,pairs=paired_block('gsm8k',names,[(f'current_d{depth}',f'plain_d{depth}'),(f'shared_d{depth}',f'current_d{depth}'),(f'shared_d{depth}',f'plain_d{depth}')],offset);table3[f'D{depth}']=vals;paired3[f'D{depth}']=pairs
 latency=json.loads((root/'gsm8k_fixed_forward_latency.json').read_text());latency_provenance={'protocol':c['protocol']+'-fixed-forward-latency','git_commit':commit,'config_sha256':config_sha,'checkpoint_sha256':c['checkpoint_sha256'],'gate_sha256':gate_sha,'concurrency_gate_sha256':concurrency_gate_sha,'hardware':c['hardware'],'tokens':c['fixed_forward_tokens'],'repetitions':c['fixed_forward_repetitions'],'warmup':c['fixed_forward_warmup']};expected_latency=['plain_d8','current_d8','projected_uniform_d8','shared_d8','per_layer_d8','plain_d16','plain_d64']
 if latency.get('provenance')!=latency_provenance or sorted(latency.get('conditions',{}))!=sorted(expected_latency) or any(any(not isinstance(v.get(k),(int,float)) or not math.isfinite(v[k]) or v[k]<0 for k in ('mean_seconds','median_seconds','peak_memory_bytes')) or v.get('tokens')!=c['fixed_forward_tokens'] or v.get('repetitions')!=c['fixed_forward_repetitions'] or v.get('warmup')!=c['fixed_forward_warmup'] for v in latency['conditions'].values()):raise RuntimeError('fixed-forward latency strict validation failed')
 for name in expected_latency:
  entry=json.loads((root/'fixed_forward_latency'/f'{name}.json').read_text())
  if any(entry.get(k)!=v for k,v in {**latency_provenance,'condition':name}.items()) or entry.get('measurement')!=latency['conditions'][name]:raise RuntimeError(f'per-condition latency mismatch: {name}')
 result={'protocol':c['protocol']+'-summary-3abc','status':'complete','inference_estimand':'Accuracy averaged over the three frozen deterministic h0 seeds. Confidence intervals resample example IDs as clusters and average the three fixed seeds; they quantify held-out-example uncertainty conditional on these seeds and do not claim generalization to arbitrary initialization seeds.','generation_latency_semantics':'Per-record worker wall-clock under concurrent condition execution is operational only and not comparable to serial per-example latency; fixed_forward_latency is the comparison metric.','table1_gsm8k_paired_seeds':table1,'table1_paired_comparisons':paired1,'table2_cross_dataset':table2,'table2_paired_comparisons':paired2,'table3_depth_generalization':table3,'table3_paired_comparisons':paired3,'fixed_forward_latency':latency,'git_commit':commit,'config_sha256':config_sha,'checkpoint_sha256':c['checkpoint_sha256'],'gate_sha256':gate_sha,'concurrency_gate_sha256':concurrency_gate_sha,'hardware':c['hardware'],'do_not_start_step_4':True};publish(root/'summary_3abc.json',result);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
