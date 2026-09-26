#!/usr/bin/env python3
"""Step 3A-3C frozen-Huginn paired-seed robustness evaluation."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,math,os,statistics,subprocess,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.correctness import build_chat_prompt,find_repetition_onset,generation_status,per_example_seed,stop_token_ids,tokenize_prompt
from src.evaluation.step3_robustness import question_and_gold,score_dataset_answer,wilson_interval
from src.evaluation.step3_concurrency import validate_correctness_gate_contract,validate_gate_contract
from src.evaluation.step3_context_amendment import effective_math500_cap
from src.models.latent_history_huginn import LatentHistoryHuginn
from src.models.per_layer_history_huginn import PerLayerHistoryHuginn
from src.training.latent_history import generate_cached,materialize_h0_schedule
AMENDMENT=ROOT/'configs/013_step3_math500_context_amendment.json'

def publish(path,value,pretty=False):
 data=(json.dumps(value,indent=2 if pretty else None,sort_keys=True)+'\n').encode()
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
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def condition_name(model,depth):return f'{model}_d{depth}'
def build_models(huginn,c):
 observed={k:sha(Path(v)) for k,v in c['checkpoints'].items()}
 if observed!=c['checkpoint_sha256']:raise RuntimeError({'checkpoint_hash_mismatch':observed})
 checkpoints={k:torch.load(v,map_location='cpu',weights_only=False) for k,v in c['checkpoints'].items()}
 wrappers={'plain':LatentHistoryHuginn(huginn,c['projection_size'],c['attention_heads']).cuda().eval(),'current':LatentHistoryHuginn(huginn,c['projection_size'],c['attention_heads']).cuda().eval(),'projected_uniform':LatentHistoryHuginn(huginn,c['projection_size'],c['attention_heads']).cuda().eval(),'shared':LatentHistoryHuginn(huginn,c['projection_size'],c['attention_heads']).cuda().eval(),'per_layer':PerLayerHistoryHuginn(huginn,c['projection_size'],c['attention_heads']).cuda().eval()}
 for name in ('current','projected_uniform','shared','per_layer'):wrappers[name].history_attention.load_state_dict(checkpoints[name]['history_attention_state_dict'])
 return wrappers,observed
def model_mode(name):return {'plain':'disabled','current':'current_only','projected_uniform':'uniform','shared':'learned','per_layer':'learned'}[name]
def validate(r,*,dataset_name,model,depth,seed_index,example_id,c,commit,tok,item,prompt_tokens):
 if dataset_name!='math500':raise RuntimeError('amendment runner is MATH-500-only')
 required=('generated_text','generated_token_ids','correct','predicted_answer','gold_answer','generated_tokens','hit_max_new_tokens','ended_naturally','generation_latency_seconds','peak_memory_bytes','repetition_onset','degeneration','used_full_prefix_fallback','full_prefix_fallback_onset');base=c['_amendment']['base_git_commit'];legacy=r.get('protocol')==c['protocol']+'-record';effective=effective_math500_cap(dataset=dataset_name,prompt_tokens=prompt_tokens,config=c,amendment=c['_amendment'])
 common={'dataset':dataset_name,'model':model,'depth':depth,'seed_index':seed_index,'example_id':example_id,'config_sha256':c['_amendment']['base_config_sha256'],'hardware':c['hardware'],'checkpoint_sha256_before':c['_checkpoint_hashes_observed'],'gate_sha256':c['_gate_sha256'],'concurrency_gate_sha256':c['_concurrency_gate_sha256'],'checkpoint_set_sha256':hashlib.sha256(json.dumps(c['_checkpoint_hashes_observed'],sort_keys=True).encode()).hexdigest()}
 if legacy:
  amendment_only=('amendment_sha256','prompt_tokens','requested_max_new_tokens','effective_max_new_tokens','context_limited')
  if any(field in r for field in amendment_only):raise RuntimeError('legacy record contains amendment-only metadata')
  expected={**common,'protocol':c['protocol']+'-record','git_commit':base};max_tokens=c['max_new_tokens']
  if prompt_tokens+c['max_new_tokens']>c['maximum_schedule_tokens']:raise RuntimeError('legacy record exceeds frozen context policy')
 else:
  expected={**common,'protocol':c['_amendment']['protocol']+'-record','git_commit':commit,'amendment_sha256':c['_amendment_sha256'],'prompt_tokens':prompt_tokens,'requested_max_new_tokens':c['max_new_tokens'],'effective_max_new_tokens':effective,'context_limited':effective<c['max_new_tokens']};max_tokens=effective
  required+=('amendment_sha256','prompt_tokens','requested_max_new_tokens','effective_max_new_tokens','context_limited')
 if len(r.get('generated_token_ids',[]))>max_tokens or any(k not in r for k in required) or any(r.get(k)!=v for k,v in expected.items()):raise RuntimeError(f'record provenance/schema mismatch {dataset_name}/{model}/D{depth}/S{seed_index}/ID{example_id}')
 if r['h0_seed']!=per_example_seed(c['h0_base_seed'],example_id,step=0,seed_index=seed_index) or r['generated_tokens']!=len(r['generated_token_ids']) or r['generated_text']!=tok.decode(r['generated_token_ids'],skip_special_tokens=False):raise RuntimeError('record seed/token reconstruction mismatch')
 if not isinstance(r['used_full_prefix_fallback'],bool) or (r['used_full_prefix_fallback']!=(r['full_prefix_fallback_onset'] is not None)) or (r['full_prefix_fallback_onset'] is not None and (not isinstance(r['full_prefix_fallback_onset'],int) or not 1<=r['full_prefix_fallback_onset']<=r['generated_tokens'])):raise RuntimeError('invalid fallback metadata')
 if not math.isfinite(float(r['generation_latency_seconds'])) or r['generation_latency_seconds']<0 or not isinstance(r['peak_memory_bytes'],int) or r['peak_memory_bytes']<0:raise RuntimeError('invalid record numerical field')
 status=generation_status(r['generated_token_ids'],max_new_tokens=max_tokens,stop_token_ids=stop_token_ids(tok));_,gold=question_and_gold(dataset_name,item);sc=score_dataset_answer(dataset_name,r['generated_text'],gold,hit_max_new_tokens=status.hit_max_new_tokens);rep=find_repetition_onset(r['generated_token_ids'],chunk_size=c['degeneration_chunk_size'],repetitions=c['degeneration_repetitions'])
 if r['hit_max_new_tokens']!=status.hit_max_new_tokens or r['ended_naturally']!=status.ended_naturally or any(r[k]!=sc[k] for k in ('correct','predicted_answer','gold_answer')) or r['repetition_onset']!=rep or r['degeneration']!=(rep is not None):raise RuntimeError('record scoring/status reconstruction mismatch')
def fixed_benchmark(wrapper,tok,dataset,c,model,depth,huginn):
 tokens=[]
 for i in range(len(dataset)):
  q,_=question_and_gold('gsm8k',dataset[i]);tokens.extend(tokenize_prompt(tok,build_chat_prompt(tok,q,c['system_instruction']))['input_ids'][0].tolist())
  if len(tokens)>=c['fixed_forward_tokens']:break
 ids=torch.tensor([tokens[:c['fixed_forward_tokens']]],device='cuda');schedule=materialize_h0_schedule(huginn,device=ids.device,example_id=0,base_seed=c['h0_base_seed'],seed_index=0);mode=model_mode(model)
 for _ in range(c['fixed_forward_warmup']):
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):wrapper(ids,schedule[:,:ids.shape[1]],depth=depth,mode=mode)
 samples=[];torch.cuda.reset_peak_memory_stats()
 for _ in range(c['fixed_forward_repetitions']):
  torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=wrapper(ids,schedule[:,:ids.shape[1]],depth=depth,mode=mode)
  torch.cuda.synchronize();samples.append(time.perf_counter()-start)
  if not bool(torch.isfinite(out.logits).all()):raise FloatingPointError('fixed-forward nonfinite')
 return {'tokens':ids.shape[1],'warmup':c['fixed_forward_warmup'],'repetitions':len(samples),'mean_seconds':sum(samples)/len(samples),'median_seconds':statistics.median(samples),'peak_memory_bytes':int(torch.cuda.max_memory_allocated())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/013_step3_huginn_robustness.json'));p.add_argument('--dataset',choices=('gsm8k','svamp','math500'),required=True);p.add_argument('--condition-names',default=None);p.add_argument('--no-finalize',action='store_true');a=p.parse_args();from datasets import load_dataset;from transformers import AutoModelForCausalLM,AutoTokenizer
 cp=ROOT/a.config;c=json.loads(cp.read_text());amendment=json.loads(AMENDMENT.read_text());c['_amendment']=amendment;c['_amendment_sha256']=sha(AMENDMENT);commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();dirty=subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
 if dirty:raise RuntimeError(f'tracked worktree dirty: {dirty}')
 if a.dataset!='math500':raise RuntimeError('context-amended runner may only execute MATH-500')
 if amendment['base_protocol']!=c['protocol'] or amendment['base_config_sha256']!=sha(cp) or amendment['requested_max_new_tokens']!=c['max_new_tokens'] or amendment['maximum_total_tokens']!=c['maximum_schedule_tokens']:raise RuntimeError('context amendment/base mismatch')
 root=Path(c['output_root']);root.mkdir(exist_ok=True);gate_path=root/'correctness_gate.json';gate_sha=sha(gate_path);gate=json.loads(gate_path.read_text());concurrency_gate_path=root/'concurrency_gate.json';concurrency_gate=json.loads(concurrency_gate_path.read_text())
 if torch.cuda.get_device_name()!=c['hardware']:raise RuntimeError('hardware mismatch')
 validate_correctness_gate_contract(gate,config=c,commit=amendment['base_git_commit'],config_sha256=sha(cp),hardware=c['hardware'])
 validate_gate_contract(concurrency_gate,config=c,commit=amendment['base_git_commit'],config_sha256=sha(cp),correctness_gate_sha256=gate_sha,hardware=c['hardware'])
 spec=c[a.dataset];dataset=load_dataset(spec['dataset_id'],spec['config'],split=spec['split'],revision=spec['revision']);ids=list(range(spec['ids'][0],spec['ids'][1]+1));tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);huginn=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();wrappers,checkpoint_hashes_before=build_models(huginn,c);c['_checkpoint_hashes_observed']=checkpoint_hashes_before;c['_gate_sha256']=gate_sha;c['_concurrency_gate_sha256']=sha(concurrency_gate_path);checkpoint_set_observed=hashlib.sha256(json.dumps(checkpoint_hashes_before,sort_keys=True).encode()).hexdigest();all_conditions=c['gsm8k_conditions'] if a.dataset=='gsm8k' else c['cross_dataset_conditions'];requested=set(a.condition_names.split(',')) if a.condition_names else None;conditions=[condition for condition in all_conditions if requested is None or condition_name(*condition) in requested]
 if requested is not None and {condition_name(*x) for x in conditions}!=requested:raise RuntimeError(f'unknown condition names: {requested-{condition_name(*x) for x in conditions}}')
 records=[];locks=[]
 mapping=[{'dataset':a.dataset,'example_id':i,'seed_index':s,'h0_seed':per_example_seed(c['h0_base_seed'],i,step=0,seed_index=s)} for i in ids for s in c['h0_seed_indices']];publish(root/f'{a.dataset}_seed_mapping.json',mapping,True)
 for model,depth in conditions:
  condition_lock=(root/'condition_locks'/a.dataset/f'{condition_name(model,depth)}.lock');condition_lock.parent.mkdir(parents=True,exist_ok=True);lock_handle=condition_lock.open('a');fcntl.flock(lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock_handle);wrapper=wrappers[model];mode=model_mode(model);name=condition_name(model,depth)
  for seed_index in c['h0_seed_indices']:
   directory=root/'records'/a.dataset/name/f'seed_{seed_index}';directory.mkdir(parents=True,exist_ok=True);existing=[]
   for path in sorted(directory.glob('*.json')):
    expected=ids[len(existing)]
    if path.name!=f'{expected:04d}.json':raise RuntimeError(f'noncontiguous record {path}')
    item=dataset[expected];question,_=question_and_gold(a.dataset,item);prompt_tokens=tokenize_prompt(tok,build_chat_prompt(tok,question,c['system_instruction']))['input_ids'].shape[1];r=json.loads(path.read_text());validate(r,dataset_name=a.dataset,model=model,depth=depth,seed_index=seed_index,example_id=expected,c=c,commit=commit,tok=tok,item=item,prompt_tokens=prompt_tokens);existing.append(r)
   records.extend(existing)
   for example_id in ids[len(existing):]:
    if a.dataset!='math500':raise RuntimeError('context amendment authorizes new records only for math500')
    question,gold=question_and_gold(a.dataset,dataset[example_id]);prompt=tokenize_prompt(tok,build_chat_prompt(tok,question,c['system_instruction']))['input_ids'].cuda();prompt_tokens=prompt.shape[1];effective=effective_math500_cap(dataset=a.dataset,prompt_tokens=prompt_tokens,config=c,amendment=amendment)
    schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=example_id,base_seed=c['h0_base_seed'],seed_index=seed_index);g=generate_cached(wrapper,tok,prompt,schedule,depth=depth,mode=mode,max_new_tokens=effective,max_cache_allocated_bytes=c['generation_cache_allocated_limit_bytes']);sc=score_dataset_answer(a.dataset,g.text,gold,hit_max_new_tokens=g.hit_max_new_tokens);rep=find_repetition_onset(g.token_ids,chunk_size=c['degeneration_chunk_size'],repetitions=c['degeneration_repetitions']);r={'protocol':amendment['protocol']+'-record','dataset':a.dataset,'model':model,'depth':depth,'seed_index':seed_index,'example_id':example_id,'h0_seed':per_example_seed(c['h0_base_seed'],example_id,step=0,seed_index=seed_index),'generated_text':g.text,'generated_token_ids':g.token_ids,**sc,'generated_tokens':g.generated_tokens,'hit_max_new_tokens':g.hit_max_new_tokens,'ended_naturally':g.ended_naturally,'generation_latency_seconds':g.latency_seconds,'peak_memory_bytes':g.peak_memory_bytes,'repetition_onset':rep,'degeneration':rep is not None,'used_full_prefix_fallback':g.used_full_prefix_fallback,'full_prefix_fallback_onset':g.full_prefix_fallback_onset,'git_commit':commit,'config_sha256':sha(cp),'hardware':c['hardware'],'checkpoint_sha256_before':checkpoint_hashes_before,'gate_sha256':gate_sha,'concurrency_gate_sha256':sha(concurrency_gate_path),'checkpoint_set_sha256':checkpoint_set_observed,'amendment_sha256':c['_amendment_sha256'],'prompt_tokens':prompt_tokens,'requested_max_new_tokens':c['max_new_tokens'],'effective_max_new_tokens':effective,'context_limited':effective<c['max_new_tokens']};validate(r,dataset_name=a.dataset,model=model,depth=depth,seed_index=seed_index,example_id=example_id,c=c,commit=commit,tok=tok,item=dataset[example_id],prompt_tokens=prompt_tokens);publish(directory/f'{example_id:04d}.json',r);records.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ('generated_text','generated_token_ids')}),flush=True)
 checkpoint_hashes_after={k:sha(Path(v)) for k,v in c['checkpoints'].items()}
 if checkpoint_hashes_after!=checkpoint_hashes_before:raise RuntimeError({'checkpoint_mutation':{'before':checkpoint_hashes_before,'after':checkpoint_hashes_after}})
 if a.no_finalize:
  result={'protocol':amendment['protocol']+'-records-only','status':'complete','dataset':a.dataset,'conditions':[condition_name(*x) for x in conditions],'records_loaded_or_generated':len(records),'git_commit':commit,'config_sha256':sha(cp),'correctness_gate_sha256':gate_sha,'concurrency_gate_sha256':sha(concurrency_gate_path)};print(json.dumps(result,indent=2,sort_keys=True));return
 if conditions!=all_conditions:raise RuntimeError('filtered condition runs must use --no-finalize')
 summaries={}
 for model,depth in conditions:
  name=condition_name(model,depth);rs=[r for r in records if r['model']==model and r['depth']==depth];by_seed={}
  for s in c['h0_seed_indices']:
   x=[r for r in rs if r['seed_index']==s];correct=sum(r['correct'] for r in x);by_seed[str(s)]={'correct':correct,'total':len(x),'accuracy':correct/len(x),'wilson_95ci':wilson_interval(correct,len(x))}
  correct=sum(r['correct'] for r in rs);summaries[name]={'model':model,'depth':depth,'by_seed':by_seed,'correct':correct,'total':len(rs),'pooled_accuracy':correct/len(rs),'wilson_95ci_naive':wilson_interval(correct,len(rs)),'mean_seed_accuracy':sum(v['accuracy'] for v in by_seed.values())/len(by_seed),'cap_hit_rate':sum(r['hit_max_new_tokens'] for r in rs)/len(rs),'degeneration_rate':sum(r['degeneration'] for r in rs)/len(rs),'mean_output_length':sum(r['generated_tokens'] for r in rs)/len(rs),'mean_generation_latency_seconds':sum(r['generation_latency_seconds'] for r in rs)/len(rs),'median_generation_latency_seconds':statistics.median(r['generation_latency_seconds'] for r in rs),'full_prefix_fallbacks':sum(r['used_full_prefix_fallback'] for r in rs)}
 if a.dataset=='gsm8k':
  latency_dir=root/'fixed_forward_latency';latency_dir.mkdir(exist_ok=True);latency={};provenance={'protocol':c['protocol']+'-fixed-forward-latency','git_commit':commit,'config_sha256':sha(cp),'checkpoint_sha256':checkpoint_hashes_before,'gate_sha256':gate_sha,'concurrency_gate_sha256':sha(concurrency_gate_path),'hardware':c['hardware'],'tokens':c['fixed_forward_tokens'],'repetitions':c['fixed_forward_repetitions'],'warmup':c['fixed_forward_warmup']}
  for model,depth in (('plain',8),('current',8),('projected_uniform',8),('shared',8),('per_layer',8),('plain',16),('plain',64)):
   name=condition_name(model,depth);path=latency_dir/f'{name}.json'
   if path.exists():entry=json.loads(path.read_text())
   else:entry={**provenance,'condition':name,'measurement':fixed_benchmark(wrappers[model],tok,dataset,c,model,depth,huginn)};publish(path,entry,True)
   measurement=entry.get('measurement',{});numeric=('mean_seconds','median_seconds','peak_memory_bytes')
   if any(entry.get(k)!=v for k,v in {**provenance,'condition':name}.items()) or any(not isinstance(measurement.get(k),(int,float)) or not math.isfinite(measurement[k]) or measurement[k]<0 for k in numeric) or measurement.get('tokens')!=c['fixed_forward_tokens'] or measurement.get('repetitions')!=c['fixed_forward_repetitions'] or measurement.get('warmup')!=c['fixed_forward_warmup']:raise RuntimeError(f'fixed-forward latency provenance/value mismatch: {name}')
   latency[name]=measurement
  aggregate={'provenance':provenance,'conditions':latency};latency_path=root/'gsm8k_fixed_forward_latency.json'
  if latency_path.exists() and json.loads(latency_path.read_text())!=aggregate:raise RuntimeError('fixed-forward aggregate mismatch')
  if not latency_path.exists():publish(latency_path,aggregate,True)
 result={'protocol':amendment['protocol']+f'-{a.dataset}-evaluation','status':'complete','dataset':a.dataset,'conditions':summaries,'records':len(records),'git_commit':commit,'amendment_sha256':c['_amendment_sha256'],'legacy_records':sum(r['protocol']==c['protocol']+'-record' for r in records),'amended_records':sum(r['protocol']==amendment['protocol']+'-record' for r in records),'context_limited_records':sum(bool(r.get('context_limited')) for r in records),'config_sha256':sha(cp),'checkpoint_sha256_before':checkpoint_hashes_before,'checkpoint_sha256_after':checkpoint_hashes_after,'checkpoint_set_sha256':checkpoint_set_observed,'gate_sha256':gate_sha,'concurrency_gate_sha256':sha(concurrency_gate_path),'hardware':torch.cuda.get_device_name(),'generation_latency_semantics':'worker wall-clock under concurrent condition execution; operational only and not comparable to serial per-example latency','full_vocabulary_logits_persisted':False,'do_not_start_step_4':True};publish(root/f'{a.dataset}_summary.json',result,True);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
