#!/usr/bin/env python3
"""Functional h8 perturb-and-recover generation through ordinary frozen D16."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,math,os,statistics,subprocess,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.correctness import build_chat_prompt,find_repetition_onset,generation_status,score_generation,stop_token_ids,tokenize_prompt
from src.models.perturbed_huginn import PerturbedHuginn
from src.training.latent_history import materialize_h0_schedule
from scripts.run_011_perturb_recover_latent import write_exclusive_json

def generate(wrapper,tok,prompt,schedule,*,example_id,sigma,seed,c):
 stops={int(v) for v in stop_token_ids(tok) if v is not None and int(v)>=0};tokens=[];cache=None;current=prompt;position=None;torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for _ in range(c['max_new_tokens']):
   h0=schedule[:,:current.shape[1]] if position is None else schedule[:,position:position+1];cp=None if position is None else torch.tensor([position],device='cuda');out=wrapper(current,h0,depth=c['functional_generation_depth'],perturb_depth=8,sigma=sigma,example_id=example_id,perturbation_seed_index=seed,perturbation_base_seed=c['perturbation_base_seed'],past_key_values=cache,use_cache=True,cache_position=cp);cache=out.past_key_values;token=int(out.logits[0,-1].argmax());tokens.append(token)
   if token in stops:break
   position=prompt.shape[1]+len(tokens)-1;current=torch.tensor([[token]],device='cuda')
 torch.cuda.synchronize();status=generation_status(tokens,max_new_tokens=c['max_new_tokens'],stop_token_ids=stop_token_ids(tok));return {'token_ids':tokens,'text':tok.decode(tokens,skip_special_tokens=False),'latency_seconds':time.perf_counter()-start,'peak_memory_bytes':int(torch.cuda.max_memory_allocated()),'hit_cap':status.hit_max_new_tokens}
def summarize(rs):return {'correct':sum(r['correct'] for r in rs),'total':len(rs),'accuracy':sum(r['correct'] for r in rs)/len(rs),'paired_wins_vs_clean':sum(r['correct'] and not r['clean_d16_correct'] for r in rs),'paired_losses_vs_clean':sum(r['clean_d16_correct'] and not r['correct'] for r in rs),'cap_hit_rate':sum(r['hit_max_new_tokens'] for r in rs)/len(rs),'degeneration_rate':sum(r['repetition_onset'] is not None for r in rs)/len(rs),'mean_generated_tokens':sum(r['generated_tokens'] for r in rs)/len(rs),'mean_generation_latency_seconds':sum(r['generation_latency_seconds'] for r in rs)/len(rs),'median_generation_latency_seconds':statistics.median(r['generation_latency_seconds'] for r in rs),'peak_memory_bytes':max(r['peak_memory_bytes'] for r in rs)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/011_perturb_recover.json'));a=p.parse_args();from datasets import load_dataset;from transformers import AutoModelForCausalLM,AutoTokenizer
 cp=ROOT/a.config;c=json.loads(cp.read_text());config_sha=hashlib.sha256(cp.read_bytes()).hexdigest();commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();root=Path(c['output_root']);gate=json.loads((root/'correctness_gate.json').read_text())
 if gate.get('status')!='pass' or gate.get('config_sha256')!=config_sha or gate.get('git_commit')!=commit:raise RuntimeError('correctness gate provenance mismatch')
 latent=json.loads((root/'latent_summary.json').read_text())
 def all_finite(value):
  if isinstance(value,float):return math.isfinite(value)
  if isinstance(value,dict):return all(all_finite(v) for v in value.values())
  if isinstance(value,list):return all(all_finite(v) for v in value)
  return True
 if latent.get('status')!='complete' or latent.get('protocol')!=c['protocol']+'-latent' or latent.get('protocol_binding',{}).get('config_sha256')!=config_sha or latent.get('git_commit')!=commit or latent.get('examples')!=250 or len(latent.get('table',[]))!=6 or any(row.get('samples')!=750 for row in latent.get('table',[])) or not all_finite(latent):raise RuntimeError('primary latent audit provenance/completeness mismatch')
 lock=(root/'generation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);record_dir=root/'generation_records';record_dir.mkdir(exist_ok=True);log=root/'generation_records.jsonl';summary_path=root/'generation_summary.json'
 if summary_path.exists():raise FileExistsError(summary_path)
 ref_path=ROOT/'results/005_latent_history_attention/comparison.json';reference=json.loads(ref_path.read_text());ref_config=json.loads((ROOT/'configs/005_latent_history_attention_d8.json').read_text())
 for key in ('model_id','model_revision','dataset_id','dataset_revision','system_instruction','h0_base_seed','max_new_tokens','test_ids'):
  if ref_config[key]!=c[key]:raise RuntimeError(f'clean D16 reference protocol mismatch: {key}')
 clean_records={r['example_id']:r for r in reference['records'] if r['model']=='original_huginn_d16'}
 if len(clean_records)!=250:raise RuntimeError('clean D16 paired records incomplete')
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);test=load_dataset(c['dataset_id'],c['dataset_config'],split='test',revision=c['dataset_revision']);huginn=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();wrapper=PerturbedHuginn(huginn).cuda().eval();conditions=[(sigma,seed,i) for sigma in c['noise_strengths'] for seed in c['perturbation_seed_indices'] for i in range(c['test_ids'][0],c['test_ids'][1]+1)];records=[]
 for record_path in sorted(record_dir.glob('*.json')):
  if record_path.name!=f'{len(records):06d}.json':raise RuntimeError(f'noncontiguous generation record: {record_path}')
  r=json.loads(record_path.read_text())
  if (r['sigma'],r['perturbation_seed_index'],r['example_id'])!=conditions[len(records)] or r.get('config_sha256')!=config_sha:raise RuntimeError(f'generation resume mismatch {record_path}')
  records.append(r)
 prompt_lengths=[tokenize_prompt(tok,build_chat_prompt(tok,test[i]['question'],c['system_instruction']))['input_ids'].shape[1] for i in range(250)]
 if max(prompt_lengths)+c['max_new_tokens']>2048:raise RuntimeError('prompt plus cap exceeds h0 schedule')
 for sigma,seed,i in conditions[len(records):]:
  prompt=tokenize_prompt(tok,build_chat_prompt(tok,test[i]['question'],c['system_instruction']))['input_ids'].cuda();schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=i,base_seed=c['h0_base_seed']);g=generate(wrapper,tok,prompt,schedule,example_id=i,sigma=sigma,seed=seed,c=c);score=score_generation(g['text'],test[i]['answer'],hit_max_new_tokens=g['hit_cap']);r={'sigma':sigma,'perturbation_seed_index':seed,'example_id':i,'config_sha256':config_sha,'noise_seed_derivation':c['noise_seed_derivation'],'correct':score.correct,'predicted_answer':score.predicted_answer,'gold_answer':score.gold_answer,'generated_text':g['text'],'generated_token_ids':g['token_ids'],'generated_tokens':len(g['token_ids']),'generation_latency_seconds':g['latency_seconds'],'peak_memory_bytes':g['peak_memory_bytes'],'hit_max_new_tokens':g['hit_cap'],'repetition_onset':find_repetition_onset(g['token_ids'],chunk_size=c['degeneration_chunk_size'],repetitions=c['degeneration_repetitions']),'clean_d16_correct':clean_records[i]['correct']};records.append(r)
  write_exclusive_json(record_dir/f'{len(records)-1:06d}.json',r)
  print(json.dumps({k:v for k,v in r.items() if k not in ('generated_text','generated_token_ids')}),flush=True)
 by_seed={f'sigma{sigma}:seed{seed}':summarize([r for r in records if r['sigma']==sigma and r['perturbation_seed_index']==seed]) for sigma in c['noise_strengths'] for seed in c['perturbation_seed_indices']};pooled={f'sigma{sigma}':summarize([r for r in records if r['sigma']==sigma]) for sigma in c['noise_strengths']};result={'protocol':c['protocol']+'-generation','status':'complete','perturb_depth':8,'final_depth':16,'per_seed':by_seed,'pooled_three_seeds':pooled,'clean_d16_reference':{'accuracy':reference['models']['original_huginn_d16']['accuracy'],'paired_records':250,'artifact':'results/005_latent_history_attention/comparison.json','sha256':hashlib.sha256((ROOT/'results/005_latent_history_attention/comparison.json').read_bytes()).hexdigest()},'config_sha256':config_sha,'git_commit':commit,'generation_record_files':len(records),'full_vocabulary_logits_persisted':False};
 if log.exists():raise FileExistsError(log)
 write_exclusive_json(log,records);write_exclusive_json(summary_path,result);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
