#!/usr/bin/env python3
"""Pinned-A100 correctness gate for Step 3A-3C."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.correctness import build_chat_prompt,score_generation,tokenize_prompt
from src.evaluation.step3_robustness import score_dataset_answer
from src.models.latent_history_huginn import LatentHistoryHuginn
from src.models.per_layer_history_huginn import PerLayerHistoryHuginn
from src.training.latent_history import generate_cached,generate_cached_batch,materialize_h0_schedule
CONFIG=ROOT/'configs/013_step3_huginn_robustness.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def publish(p,v):
 data=(json.dumps(v,indent=2,sort_keys=True)+'\n').encode();tmp=p.with_name(f'.{p.name}.{os.getpid()}.tmp')
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 try:os.link(tmp,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 finally:tmp.unlink(missing_ok=True)
def main():
 from datasets import load_dataset;from transformers import AutoModelForCausalLM,AutoTokenizer
 c=json.loads(CONFIG.read_text());
 expected_transfer={(m,d) for m in ('plain','current','shared') for d in (4,8,16,32,64)}
 expected_cross={('plain',8),('current',8),('shared',8),('plain',16),('plain',64)}
 if set(map(tuple,c['depth_transfer_conditions']))!=expected_transfer or set(map(tuple,c['cross_dataset_conditions']))!=expected_cross or c['h0_seed_indices']!=[0,1,2] or c['do_not_start_step_4'] is not True:raise RuntimeError('frozen Step-3A-C scope mismatch')
 if torch.cuda.get_device_name()!=c['hardware']:raise RuntimeError(f"expected {c['hardware']}, got {torch.cuda.get_device_name()}")
 commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();dirty=subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
 if dirty:raise RuntimeError(f'tracked worktree dirty: {dirty}')
 root=Path(c['output_root']);root.mkdir(exist_ok=True);out_path=root/'correctness_gate.json'
 if out_path.exists():raise FileExistsError(out_path)
 hashes={k:sha(v) for k,v in c['checkpoints'].items()}
 if hashes!=c['checkpoint_sha256']:raise RuntimeError({'checkpoint_hash_mismatch':hashes})
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);huginn=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();checkpoints={k:torch.load(v,map_location='cpu',weights_only=False) for k,v in c['checkpoints'].items()};wrappers={'plain':LatentHistoryHuginn(huginn,c['projection_size'],c['attention_heads']).cuda().eval(),'current':LatentHistoryHuginn(huginn,c['projection_size'],c['attention_heads']).cuda().eval(),'projected_uniform':LatentHistoryHuginn(huginn,c['projection_size'],c['attention_heads']).cuda().eval(),'shared':LatentHistoryHuginn(huginn,c['projection_size'],c['attention_heads']).cuda().eval(),'per_layer':PerLayerHistoryHuginn(huginn,c['projection_size'],c['attention_heads']).cuda().eval()}
 for name in ('current','projected_uniform','shared','per_layer'):wrappers[name].history_attention.load_state_dict(checkpoints[name]['history_attention_state_dict'])
 modes={'plain':'disabled','current':'current_only','projected_uniform':'uniform','shared':'learned','per_layer':'learned'};gsm=load_dataset(c['gsm8k']['dataset_id'],c['gsm8k']['config'],split='test',revision=c['gsm8k']['revision']);ids=tokenize_prompt(tok,build_chat_prompt(tok,gsm[0]['question'],c['system_instruction']))['input_ids'].cuda();schedules={s:materialize_h0_schedule(huginn,device=ids.device,example_id=0,base_seed=c['h0_base_seed'],seed_index=s) for s in c['h0_seed_indices']};repeat=materialize_h0_schedule(huginn,device=ids.device,example_id=0,base_seed=c['h0_base_seed'],seed_index=0)
 seed_checks={'seed0_repeat_bitwise':torch.equal(schedules[0],repeat),'three_pairwise_distinct':all(not torch.equal(schedules[a],schedules[b]) for a,b in ((0,1),(0,2),(1,2))),'prefix_stable':torch.equal(schedules[0][:,:ids.shape[1]],repeat[:,:ids.shape[1]])}
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  native=huginn(input_ids=ids,input_states=schedules[0][:,:ids.shape[1]],attention_mask=torch.ones_like(ids,dtype=torch.bool),num_steps=8,use_cache=False,return_dict=True,output_details={'return_logits':True,'return_latents':True,'return_head':False,'return_stats':False})
  plain=wrappers['plain'](ids,schedules[0][:,:ids.shape[1]],depth=8,mode='disabled')
 parity={'plain_d8_logits_bitwise':torch.equal(native.logits.float(),plain.logits),'plain_d8_latents_bitwise':torch.equal(native.latent_states,plain.latent_states)}
 transfer=[]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for model,depth in c['gsm8k_conditions']:
   if model not in ('plain','current','shared') and depth!=8:continue
   output=wrappers[model](ids,schedules[0][:,:ids.shape[1]],depth=depth,mode=modes[model]);transfer.append({'model':model,'depth':depth,'finite_logits':bool(torch.isfinite(output.logits).all()),'finite_latents':bool(torch.isfinite(output.latent_states).all())})
 cache_checks=[]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for model,depth in [('plain',8),('current',8),('projected_uniform',8),('shared',8),('per_layer',8),('plain',16),('current',64),('shared',64)]:
   wrapper=wrappers[model];cache=None;current=ids;prefix=ids;position=None;steps=[]
   for step in range(3):
    states=schedules[0][:,:current.shape[1]] if position is None else schedules[0][:,position:position+1];cp=None if position is None else torch.tensor([position],device='cuda');cached=wrapper(current,states,depth=depth,mode=modes[model],past_key_values=cache,use_cache=True,cache_position=cp);cache=cached.past_key_values;full=wrapper(prefix,schedules[0][:,:prefix.shape[1]],depth=depth,mode=modes[model]);ct=int(cached.logits[0,-1].argmax());ft=int(full.logits[0,-1].argmax());finite=bool(torch.isfinite(cached.logits).all()) and bool(torch.isfinite(cached.latent_states).all()) and bool(torch.isfinite(full.logits).all());steps.append({'token_exact':ct==ft,'finite':finite})
    if ct!=ft or not finite:raise RuntimeError({'cache_full_mismatch':(model,depth,steps)})
    token=torch.tensor([[ct]],device='cuda');prefix=torch.cat((prefix,token),1);position=prefix.shape[1]-1;current=token
   cache_checks.append({'model':model,'depth':depth,'steps':steps})
 fallback_checks=[]
 for model,depth in (('plain',8),('current',64),('shared',64),('projected_uniform',8),('per_layer',8)):
  normal=generate_cached(wrappers[model],tok,ids,schedules[0],depth=depth,mode=modes[model],max_new_tokens=4)
  forced=generate_cached(wrappers[model],tok,ids,schedules[0],depth=depth,mode=modes[model],max_new_tokens=4,max_cache_allocated_bytes=0)
  fallback_checks.append({'model':model,'depth':depth,'complete_token_sequence_exact':normal.token_ids==forced.token_ids,'forced_fallback_used':forced.used_full_prefix_fallback,'fallback_onset':forced.full_prefix_fallback_onset})
 prompt_by_id={example_id:tokenize_prompt(tok,build_chat_prompt(tok,gsm[example_id]['question'],c['system_instruction']))['input_ids'].cuda() for example_id in range(250)};by_length={}
 for example_id,prompt in prompt_by_id.items():by_length.setdefault(prompt.shape[1],[]).append(example_id)
 batch_parity=[];production_cases=list(map(tuple,c['gsm8k_conditions']))
 for model,depth in production_cases:
  limit=int(c['generation_batch_size_by_depth'][str(depth)]);eligible=next((values for _,values in sorted(by_length.items()) if len(values)*len(c['h0_seed_indices'])>=limit),None)
  if eligible is None:raise RuntimeError(f'no exact-length production batch available at D{depth} size {limit}')
  tasks=[(example_id,seed) for example_id in eligible for seed in c['h0_seed_indices']][:limit];prompt_batch=torch.cat([prompt_by_id[example_id] for example_id,_ in tasks],0);schedule_batch=torch.cat([materialize_h0_schedule(huginn,device=prompt_batch.device,example_id=example_id,base_seed=c['h0_base_seed'],seed_index=seed) for example_id,seed in tasks],0);batched=generate_cached_batch(wrappers[model],tok,prompt_batch,schedule_batch,depth=depth,mode=modes[model],max_new_tokens=c['max_new_tokens'],max_cache_allocated_bytes=c['generation_cache_allocated_limit_bytes'],honor_stop_tokens=False);serial=[]
  for example_id,seed in tasks:serial.append(generate_cached(wrappers[model],tok,prompt_by_id[example_id],materialize_h0_schedule(huginn,device=prompt_batch.device,example_id=example_id,base_seed=c['h0_base_seed'],seed_index=seed),depth=depth,mode=modes[model],max_new_tokens=c['max_new_tokens'],max_cache_allocated_bytes=c['generation_cache_allocated_limit_bytes'],honor_stop_tokens=False))
  exact=all(a.token_ids==b.token_ids and a.hit_max_new_tokens==b.hit_max_new_tokens and a.ended_naturally==b.ended_naturally for a,b in zip(batched,serial));batch_parity.append({'model':model,'depth':depth,'batch_size':len(tasks),'configured_batch_size':limit,'full_cap_tokens':c['max_new_tokens'],'token_and_stopping_exact':exact,'peak_memory_bytes':max(x.peak_memory_bytes for x in batched),'within_device_memory':max(x.peak_memory_bytes for x in batched)<torch.cuda.get_device_properties(0).total_memory})
 natural_tasks=[(example_id,seed) for example_id in next(values for _,values in sorted(by_length.items()) if len(values)*3>=12) for seed in c['h0_seed_indices']][:12];natural_prompts=torch.cat([prompt_by_id[e] for e,_ in natural_tasks],0);natural_schedules=torch.cat([materialize_h0_schedule(huginn,device=natural_prompts.device,example_id=e,base_seed=c['h0_base_seed'],seed_index=s) for e,s in natural_tasks],0);natural_batch=generate_cached_batch(wrappers['plain'],tok,natural_prompts,natural_schedules,depth=8,mode='disabled',max_new_tokens=c['max_new_tokens']);natural_serial=[generate_cached(wrappers['plain'],tok,prompt_by_id[e],materialize_h0_schedule(huginn,device=natural_prompts.device,example_id=e,base_seed=c['h0_base_seed'],seed_index=s),depth=8,mode='disabled',max_new_tokens=c['max_new_tokens']) for e,s in natural_tasks];natural_parity={'token_stopping_exact':all(a.token_ids==b.token_ids and a.hit_max_new_tokens==b.hit_max_new_tokens and a.ended_naturally==b.ended_naturally for a,b in zip(natural_batch,natural_serial)),'staggered_stopping_observed':len({len(x.token_ids) for x in natural_batch})>1}
 forced_batch=generate_cached_batch(wrappers['plain'],tok,natural_prompts,natural_schedules,depth=8,mode='disabled',max_new_tokens=4,max_cache_allocated_bytes=0);forced_serial=[generate_cached(wrappers['plain'],tok,prompt_by_id[e],materialize_h0_schedule(huginn,device=natural_prompts.device,example_id=e,base_seed=c['h0_base_seed'],seed_index=s),depth=8,mode='disabled',max_new_tokens=4,max_cache_allocated_bytes=0) for e,s in natural_tasks];batch_fallback={'token_and_metadata_exact':all(a.token_ids==b.token_ids and a.used_full_prefix_fallback==b.used_full_prefix_fallback and a.full_prefix_fallback_onset==b.full_prefix_fallback_onset for a,b in zip(forced_batch,forced_serial))}
 scoring={'svamp_numeric':score_dataset_answer('svamp','The answer is 27.','27',hit_max_new_tokens=False)['correct'],'math_latex':score_dataset_answer('math500','Therefore $\\boxed{(3,\\frac{\\pi}{2})}$.','\\left( 3, \\frac{\\pi}{2} \\right)',hit_max_new_tokens=False)['correct'],'cap_disallows_unmarked_fallback':not score_dataset_answer('math500','work 3 then 7','7',hit_max_new_tokens=True)['correct']}
 unchanged={k:sha(v)==hashes[k] for k,v in c['checkpoints'].items()};result={'protocol':c['protocol']+'-correctness-gate','status':'pass','checkpoint_sha256':hashes,'seed_checks':seed_checks,'native_parity':parity,'depth_transfer_finiteness':transfer,'cache_vs_full_prefix':cache_checks,'forced_fallback_checks':fallback_checks,'batched_serial_parity':batch_parity,'batched_natural_stopping_parity':natural_parity,'batched_forced_fallback_parity':batch_fallback,'scoring_checks':scoring,'checkpoints_unchanged':unchanged,'git_commit':commit,'config_sha256':sha(CONFIG),'hardware':torch.cuda.get_device_name(),'full_vocabulary_logits_persisted':False}
 if not all(seed_checks.values()) or not all(parity.values()) or not all(x['finite_logits'] and x['finite_latents'] for x in transfer) or not all(x['complete_token_sequence_exact'] and x['forced_fallback_used'] and x['fallback_onset']==1 for x in fallback_checks) or not all(x['token_and_stopping_exact'] and x['batch_size']==x['configured_batch_size'] and x['within_device_memory'] for x in batch_parity) or not all(natural_parity.values()) or not batch_fallback['token_and_metadata_exact'] or not all(scoring.values()) or not all(unchanged.values()):raise RuntimeError(result)
 publish(out_path,result);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
