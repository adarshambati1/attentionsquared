#!/usr/bin/env python3
"""Step 3A-3C frozen-Huginn paired-seed robustness evaluation."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,math,os,statistics,subprocess,sys,time
from types import SimpleNamespace
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.correctness import build_chat_prompt,find_repetition_onset,generation_status,per_example_seed,stop_token_ids,tokenize_prompt
from src.evaluation.step3_robustness import question_and_gold,score_dataset_answer,wilson_interval
from src.models.latent_history_huginn import LatentHistoryHuginn
from src.models.per_layer_history_huginn import PerLayerHistoryHuginn
from src.training.latent_history import generate_cached_batch,materialize_h0_schedule

def publish(path,value,pretty=False):
 data=(json.dumps(value,indent=2 if pretty else None,sort_keys=True)+'\n').encode()
 if path.exists():
  if path.read_bytes()!=data:raise RuntimeError(f'existing artifact differs: {path}')
  return
 tmp=path.with_name(f'.{path.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp')
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 try:os.link(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
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
def validate(r,*,dataset_name,model,depth,seed_index,example_id,c,commit,tok,item):
 required=('generated_text','generated_token_ids','correct','predicted_answer','gold_answer','generated_tokens','hit_max_new_tokens','ended_naturally','batch_wall_seconds_to_row_stop','batch_elapsed_seconds','batch_generated_tokens_total','peak_memory_bytes','repetition_onset','degeneration','used_full_prefix_fallback','full_prefix_fallback_onset','generation_execution','generation_batch_size','batch_id','batch_member_index','batch_manifest_sha256','batch_measurement_sha256')
 if any(k not in r for k in required) or any((r.get(k)!=v) for k,v in {'protocol':c['protocol']+'-record','dataset':dataset_name,'model':model,'depth':depth,'seed_index':seed_index,'example_id':example_id,'git_commit':commit,'config_sha256':sha(ROOT/'configs/013_step3_huginn_robustness.json'),'hardware':c['hardware'],'checkpoint_sha256_before':c['_checkpoint_hashes_observed'],'gate_sha256':c['_gate_sha256'],'checkpoint_set_sha256':hashlib.sha256(json.dumps(c['_checkpoint_hashes_observed'],sort_keys=True).encode()).hexdigest()}.items()):raise RuntimeError(f'record provenance/schema mismatch {dataset_name}/{model}/D{depth}/S{seed_index}/ID{example_id}')
 if r['h0_seed']!=per_example_seed(c['h0_base_seed'],example_id,step=0,seed_index=seed_index) or r['generated_tokens']!=len(r['generated_token_ids']) or r['generated_text']!=tok.decode(r['generated_token_ids'],skip_special_tokens=False):raise RuntimeError('record seed/token reconstruction mismatch')
 batch=c['_batch_lookup'].get((model,depth,seed_index,example_id))
 measurement_path=Path(c['output_root'])/'batch_measurements'/dataset_name/condition_name(model,depth)/f"{r.get('batch_id')}.json"
 if batch is None or not measurement_path.is_file() or sha(measurement_path)!=r.get('batch_measurement_sha256') or r['generation_execution']!='exact-length-batched-greedy' or r['generation_batch_size']!=len(batch['members']) or r['batch_id']!=batch['batch_id'] or r['batch_member_index']!=batch['member_index'] or r['batch_manifest_sha256']!=c['_batch_manifest_sha256'] or not isinstance(r['batch_measurement_sha256'],str) or len(r['batch_measurement_sha256'])!=64:raise RuntimeError('invalid generation batch provenance')
 if not isinstance(r['used_full_prefix_fallback'],bool) or (r['used_full_prefix_fallback']!=(r['full_prefix_fallback_onset'] is not None)) or (r['full_prefix_fallback_onset'] is not None and (not isinstance(r['full_prefix_fallback_onset'],int) or not 1<=r['full_prefix_fallback_onset']<=r['generated_tokens'])):raise RuntimeError('invalid fallback metadata')
 measurement=json.loads(measurement_path.read_text());member_results=measurement.get('member_results',[])
 if r['batch_member_index']>=len(member_results) or member_results[r['batch_member_index']].get('token_ids')!=r['generated_token_ids'] or member_results[r['batch_member_index']].get('latency_seconds')!=r['batch_wall_seconds_to_row_stop'] or member_results[r['batch_member_index']].get('used_full_prefix_fallback')!=r['used_full_prefix_fallback'] or member_results[r['batch_member_index']].get('full_prefix_fallback_onset')!=r['full_prefix_fallback_onset']:raise RuntimeError('batch member measurement binding mismatch')
 if r['batch_elapsed_seconds']!=measurement.get('batch_elapsed_seconds') or r['batch_generated_tokens_total']!=measurement.get('batch_generated_tokens_total') or r['peak_memory_bytes']!=measurement.get('peak_memory_bytes'):raise RuntimeError('batch measurement binding mismatch')
 if not math.isfinite(float(r['batch_wall_seconds_to_row_stop'])) or r['batch_wall_seconds_to_row_stop']<0 or not math.isfinite(float(r['batch_elapsed_seconds'])) or r['batch_elapsed_seconds']<r['batch_wall_seconds_to_row_stop'] or not isinstance(r['batch_generated_tokens_total'],int) or r['batch_generated_tokens_total']<r['generated_tokens'] or not isinstance(r['peak_memory_bytes'],int) or r['peak_memory_bytes']<0:raise RuntimeError('invalid record numerical field')
 status=generation_status(r['generated_token_ids'],max_new_tokens=c['max_new_tokens'],stop_token_ids=stop_token_ids(tok));_,gold=question_and_gold(dataset_name,item);sc=score_dataset_answer(dataset_name,r['generated_text'],gold,hit_max_new_tokens=status.hit_max_new_tokens);rep=find_repetition_onset(r['generated_token_ids'],chunk_size=c['degeneration_chunk_size'],repetitions=c['degeneration_repetitions'])
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
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/013_step3_huginn_robustness.json'));p.add_argument('--dataset',choices=('gsm8k','svamp','math500'),required=True);a=p.parse_args();from datasets import load_dataset;from transformers import AutoModelForCausalLM,AutoTokenizer
 cp=ROOT/a.config;c=json.loads(cp.read_text());commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();dirty=subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
 if dirty:raise RuntimeError(f'tracked worktree dirty: {dirty}')
 root=Path(c['output_root']);root.mkdir(exist_ok=True);lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gate_path=root/'correctness_gate.json';gate_sha=sha(gate_path);gate=json.loads(gate_path.read_text())
 if gate.get('status')!='pass' or gate.get('git_commit')!=commit or gate.get('config_sha256')!=sha(cp) or gate.get('hardware')!=c['hardware'] or torch.cuda.get_device_name()!=c['hardware']:raise RuntimeError('correctness gate/hardware mismatch')
 spec=c[a.dataset];dataset=load_dataset(spec['dataset_id'],spec['config'],split=spec['split'],revision=spec['revision']);ids=list(range(spec['ids'][0],spec['ids'][1]+1));tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);huginn=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();wrappers,checkpoint_hashes_before=build_models(huginn,c);c['_checkpoint_hashes_observed']=checkpoint_hashes_before;c['_gate_sha256']=gate_sha;checkpoint_set_observed=hashlib.sha256(json.dumps(checkpoint_hashes_before,sort_keys=True).encode()).hexdigest();conditions=c['gsm8k_conditions'] if a.dataset=='gsm8k' else c['cross_dataset_conditions'];records=[]
 mapping=[{'dataset':a.dataset,'example_id':i,'seed_index':s,'h0_seed':per_example_seed(c['h0_base_seed'],i,step=0,seed_index=s)} for i in ids for s in c['h0_seed_indices']];publish(root/f'{a.dataset}_seed_mapping.json',mapping,True)
 prompts={}
 for example_id in ids:
  question,_=question_and_gold(a.dataset,dataset[example_id]);prompt=tokenize_prompt(tok,build_chat_prompt(tok,question,c['system_instruction']))['input_ids'].cpu()
  if prompt.shape[1]+c['max_new_tokens']>c['maximum_schedule_tokens']:raise RuntimeError(f'schedule overflow {a.dataset} ID={example_id}')
  prompts[example_id]=prompt
 batch_conditions={}
 for model,depth in conditions:
  name=condition_name(model,depth);groups={}
  for example_id in ids:
   for seed_index in c['h0_seed_indices']:groups.setdefault(prompts[example_id].shape[1],[]).append((example_id,seed_index))
  batches=[];limit=int(c['generation_batch_size_by_depth'][str(depth)]);batch_number=0
  for prompt_length in sorted(groups):
   tasks=sorted(groups[prompt_length])
   for offset in range(0,len(tasks),limit):
    members=[{'example_id':example_id,'seed_index':seed_index} for example_id,seed_index in tasks[offset:offset+limit]];batches.append({'batch_id':f'{name}-b{batch_number:05d}','prompt_tokens':prompt_length,'configured_max_batch_size':limit,'members':members});batch_number+=1
  batch_conditions[name]=batches
 batch_manifest={'protocol':c['protocol']+f'-{a.dataset}-batch-manifest','dataset':a.dataset,'git_commit':commit,'config_sha256':sha(cp),'conditions':batch_conditions};batch_manifest_path=root/f'{a.dataset}_batch_manifest.json';publish(batch_manifest_path,batch_manifest,True);batch_manifest_sha=sha(batch_manifest_path);c['_batch_manifest_sha256']=batch_manifest_sha;c['_batch_lookup']={}
 for model,depth in conditions:
  name=condition_name(model,depth)
  for batch in batch_conditions[name]:
   for member_index,member in enumerate(batch['members']):c['_batch_lookup'][(model,depth,member['seed_index'],member['example_id'])]={**batch,'member_index':member_index}
 for model,depth in conditions:
  wrapper=wrappers[model];mode=model_mode(model);name=condition_name(model,depth);existing={}
  for seed_index in c['h0_seed_indices']:
   directory=root/'records'/a.dataset/name/f'seed_{seed_index}';directory.mkdir(parents=True,exist_ok=True);seen=set()
   for path in sorted(directory.glob('*.json')):
    r=json.loads(path.read_text());example_id=r.get('example_id')
    if path.name!=f'{example_id:04d}.json' or example_id not in ids or example_id in seen:raise RuntimeError(f'invalid/duplicate resumable record {path}')
    validate(r,dataset_name=a.dataset,model=model,depth=depth,seed_index=seed_index,example_id=example_id,c=c,commit=commit,tok=tok,item=dataset[example_id]);seen.add(example_id);existing[(example_id,seed_index)]=r;records.append(r)
  for batch in batch_conditions[name]:
   batch_tasks=[(m['example_id'],m['seed_index']) for m in batch['members']]
   if all(task in existing for task in batch_tasks):continue
   measurement_path=root/'batch_measurements'/a.dataset/name/f"{batch['batch_id']}.json";measurement_path.parent.mkdir(parents=True,exist_ok=True)
   if measurement_path.exists():
    measurement=json.loads(measurement_path.read_text());member_results=measurement.get('member_results',[]);sequences_sha=hashlib.sha256(json.dumps([x['token_ids'] for x in member_results],separators=(',',':')).encode()).hexdigest()
    if measurement.get('token_sequences_sha256')!=sequences_sha or measurement.get('ordered_members')!=batch['members'] or measurement.get('batch_manifest_sha256')!=batch_manifest_sha or len(member_results)!=len(batch_tasks):raise RuntimeError(f'batch measurement replay mismatch: {batch["batch_id"]}')
    outputs=[SimpleNamespace(**x) for x in member_results]
   else:
    prompt_batch=torch.cat([prompts[example_id] for example_id,_ in batch_tasks],0).cuda();schedule_batch=torch.cat([materialize_h0_schedule(huginn,device=prompt_batch.device,example_id=example_id,base_seed=c['h0_base_seed'],seed_index=seed_index) for example_id,seed_index in batch_tasks],0);generated_outputs=generate_cached_batch(wrapper,tok,prompt_batch,schedule_batch,depth=depth,mode=mode,max_new_tokens=c['max_new_tokens'],max_cache_allocated_bytes=c['generation_cache_allocated_limit_bytes']);member_results=[{'token_ids':g.token_ids,'text':g.text,'latency_seconds':g.latency_seconds,'generated_tokens':g.generated_tokens,'hit_max_new_tokens':g.hit_max_new_tokens,'ended_naturally':g.ended_naturally,'peak_memory_bytes':g.peak_memory_bytes,'used_full_prefix_fallback':g.used_full_prefix_fallback,'full_prefix_fallback_onset':g.full_prefix_fallback_onset} for g in generated_outputs];sequences_sha=hashlib.sha256(json.dumps([x['token_ids'] for x in member_results],separators=(',',':')).encode()).hexdigest();measurement={'protocol':c['protocol']+'-batch-measurement','batch_id':batch['batch_id'],'batch_manifest_sha256':batch_manifest_sha,'ordered_members':batch['members'],'token_sequences_sha256':sequences_sha,'member_results':member_results,'batch_elapsed_seconds':max(x['latency_seconds'] for x in member_results),'batch_generated_tokens_total':sum(x['generated_tokens'] for x in member_results),'peak_memory_bytes':max(x['peak_memory_bytes'] for x in member_results),'git_commit':commit,'config_sha256':sha(cp)};publish(measurement_path,measurement,True);outputs=[SimpleNamespace(**x) for x in member_results]
   measurement_sha=sha(measurement_path);batch_elapsed=measurement['batch_elapsed_seconds'];batch_tokens=measurement['batch_generated_tokens_total'];batch_peak=measurement['peak_memory_bytes']
   for member_index,((example_id,seed_index),g) in enumerate(zip(batch_tasks,outputs)):
    if (example_id,seed_index) in existing:
     old=existing[(example_id,seed_index)]
     if old['generated_token_ids']!=g.token_ids or old['hit_max_new_tokens']!=g.hit_max_new_tokens or old['ended_naturally']!=g.ended_naturally:raise RuntimeError(f'partial-batch replay mismatch: {batch["batch_id"]}')
     continue
    _,gold=question_and_gold(a.dataset,dataset[example_id]);sc=score_dataset_answer(a.dataset,g.text,gold,hit_max_new_tokens=g.hit_max_new_tokens);rep=find_repetition_onset(g.token_ids,chunk_size=c['degeneration_chunk_size'],repetitions=c['degeneration_repetitions']);r={'protocol':c['protocol']+'-record','dataset':a.dataset,'model':model,'depth':depth,'seed_index':seed_index,'example_id':example_id,'h0_seed':per_example_seed(c['h0_base_seed'],example_id,step=0,seed_index=seed_index),'generated_text':g.text,'generated_token_ids':g.token_ids,**sc,'generated_tokens':g.generated_tokens,'hit_max_new_tokens':g.hit_max_new_tokens,'ended_naturally':g.ended_naturally,'batch_wall_seconds_to_row_stop':g.latency_seconds,'batch_elapsed_seconds':batch_elapsed,'batch_generated_tokens_total':batch_tokens,'peak_memory_bytes':batch_peak,'repetition_onset':rep,'degeneration':rep is not None,'used_full_prefix_fallback':g.used_full_prefix_fallback,'full_prefix_fallback_onset':g.full_prefix_fallback_onset,'generation_execution':'exact-length-batched-greedy','generation_batch_size':len(batch_tasks),'batch_id':batch['batch_id'],'batch_member_index':member_index,'batch_manifest_sha256':batch_manifest_sha,'batch_measurement_sha256':measurement_sha,'git_commit':commit,'config_sha256':sha(cp),'hardware':c['hardware'],'checkpoint_sha256_before':checkpoint_hashes_before,'gate_sha256':gate_sha,'checkpoint_set_sha256':checkpoint_set_observed};validate(r,dataset_name=a.dataset,model=model,depth=depth,seed_index=seed_index,example_id=example_id,c=c,commit=commit,tok=tok,item=dataset[example_id]);directory=root/'records'/a.dataset/name/f'seed_{seed_index}';publish(directory/f'{example_id:04d}.json',r);existing[(example_id,seed_index)]=r;records.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ('generated_text','generated_token_ids')}),flush=True)
 summaries={}
 for model,depth in conditions:
  name=condition_name(model,depth);rs=[r for r in records if r['model']==model and r['depth']==depth];by_seed={}
  for s in c['h0_seed_indices']:
   x=[r for r in rs if r['seed_index']==s];correct=sum(r['correct'] for r in x);by_seed[str(s)]={'correct':correct,'total':len(x),'accuracy':correct/len(x),'wilson_95ci':wilson_interval(correct,len(x))}
  correct=sum(r['correct'] for r in rs);batch_metrics={r['batch_id']:(r['batch_elapsed_seconds'],r['batch_generated_tokens_total']) for r in rs};elapsed=sum(v[0] for v in batch_metrics.values());batch_tokens=sum(v[1] for v in batch_metrics.values());summaries[name]={'model':model,'depth':depth,'by_seed':by_seed,'correct':correct,'total':len(rs),'pooled_accuracy':correct/len(rs),'wilson_95ci_naive':wilson_interval(correct,len(rs)),'mean_seed_accuracy':sum(v['accuracy'] for v in by_seed.values())/len(by_seed),'cap_hit_rate':sum(r['hit_max_new_tokens'] for r in rs)/len(rs),'degeneration_rate':sum(r['degeneration'] for r in rs)/len(rs),'mean_output_length':sum(r['generated_tokens'] for r in rs)/len(rs),'batched_generated_tokens_per_second':batch_tokens/elapsed,'total_batch_wall_seconds':elapsed,'batch_count':len(batch_metrics),'latency_estimand':'fixed-forward latency is comparable across mechanisms; generation reports amortized exact-length-batch throughput, not serial per-example latency','full_prefix_fallbacks':sum(r['used_full_prefix_fallback'] for r in rs)}
 if a.dataset=='gsm8k':
  latency_dir=root/'fixed_forward_latency';latency_dir.mkdir(exist_ok=True);latency={};provenance={'protocol':c['protocol']+'-fixed-forward-latency','git_commit':commit,'config_sha256':sha(cp),'checkpoint_sha256':checkpoint_hashes_before,'gate_sha256':gate_sha,'hardware':c['hardware'],'tokens':c['fixed_forward_tokens'],'repetitions':c['fixed_forward_repetitions'],'warmup':c['fixed_forward_warmup']}
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
 checkpoint_hashes_after={k:sha(Path(v)) for k,v in c['checkpoints'].items()}
 if checkpoint_hashes_after!=checkpoint_hashes_before:raise RuntimeError({'checkpoint_mutation':{'before':checkpoint_hashes_before,'after':checkpoint_hashes_after}})
 result={'protocol':c['protocol']+f'-{a.dataset}-evaluation','status':'complete','dataset':a.dataset,'conditions':summaries,'records':len(records),'git_commit':commit,'config_sha256':sha(cp),'checkpoint_sha256_before':checkpoint_hashes_before,'checkpoint_sha256_after':checkpoint_hashes_after,'checkpoint_set_sha256':checkpoint_set_observed,'gate_sha256':gate_sha,'batch_manifest_sha256':batch_manifest_sha,'hardware':torch.cuda.get_device_name(),'full_vocabulary_logits_persisted':False,'do_not_start_step_4':True};publish(root/f'{a.dataset}_summary.json',result,True);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
