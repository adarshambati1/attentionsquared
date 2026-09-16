#!/usr/bin/env python3
"""Correctness and D512 viability gate for Step-2 depth regression."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.correctness import build_chat_prompt,stop_token_ids,tokenize_prompt
from src.models.plain_depth_huginn import PlainDepthHuginn
from src.training.latent_history import materialize_h0_schedule
CONFIG=ROOT/'configs/012_depth_regression.json'
def sha(module):
 d=hashlib.sha256()
 for n,v in module.state_dict().items():d.update(n.encode());d.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return d.hexdigest()
def tensor_bytes(value,seen=None):
 if seen is None:seen=set()
 if id(value) in seen:return 0
 seen.add(id(value))
 if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
 if isinstance(value,dict):return sum(tensor_bytes(k,seen)+tensor_bytes(v,seen) for k,v in value.items())
 if isinstance(value,(list,tuple,set)):return sum(tensor_bytes(v,seen) for v in value)
 if hasattr(value,'__dict__'):return tensor_bytes(vars(value),seen)
 return 0
def publish(path,value):
 data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode();tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 try:
  os.link(tmp,path);directory_fd=os.open(path.parent,os.O_RDONLY);os.fsync(directory_fd);os.close(directory_fd)
 finally:tmp.unlink(missing_ok=True)
def main():
 from datasets import load_dataset;from transformers import AutoModelForCausalLM,AutoTokenizer
 c=json.loads(CONFIG.read_text());dirty=subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
 if dirty:raise RuntimeError(f'tracked worktree is dirty: {dirty}')
 root=Path(c['output_root']);root.mkdir(exist_ok=True);output=root/'correctness_gate.json'
 if output.exists():raise FileExistsError(output)
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);test=load_dataset(c['dataset_id'],c['dataset_config'],split='test',revision=c['dataset_revision']);huginn=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();model=PlainDepthHuginn(huginn).cuda().eval();prompt=tokenize_prompt(tok,build_chat_prompt(tok,test[0]['question'],c['system_instruction']))['input_ids'].cuda();schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=0,base_seed=c['h0_base_seed']);h0=schedule[:,:prompt.shape[1]];before=sha(huginn);native_checks=[]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for depth in c['depths']:
   native=huginn(input_ids=prompt,input_states=h0,attention_mask=torch.ones_like(prompt,dtype=torch.bool),num_steps=depth,use_cache=False,return_dict=True,output_details={'return_logits':True,'return_latents':True,'return_head':False,'return_stats':False});wrapped=model(prompt,h0,depth=depth,compute_fixed_point_residual=True);native_checks.append({'depth':depth,'logits_bitwise_exact':torch.equal(native.logits.float(),wrapped.logits),'latents_bitwise_exact':torch.equal(native.latent_states,wrapped.latent_states),'fixed_point_residual_finite':wrapped.fixed_point_residual is not None and torch.isfinite(torch.tensor(wrapped.fixed_point_residual)).item()})
 cache_checks=[];stops={int(v) for v in stop_token_ids(tok) if v is not None and int(v)>=0}
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for depth in c['depths']:
   native_cache=model.new_dynamic_cache();wrapped_cache=model.new_dynamic_cache();current=prompt;position=None;steps=[]
   for step in range(8):
    input_h0=schedule[:,:current.shape[1]] if position is None else schedule[:,position:position+1];cp=None if position is None else torch.tensor([position],device='cuda');native=huginn(input_ids=current,input_states=input_h0,num_steps=depth,past_key_values=native_cache,use_cache=True,cache_position=cp,return_dict=True,output_details={'return_logits':True,'return_latents':True,'return_head':False,'return_stats':False});wrapped=model(current,input_h0,depth=depth,past_key_values=wrapped_cache,use_cache=True,cache_position=cp);native_cache=native.past_key_values;wrapped_cache=wrapped.past_key_values;finite=bool(torch.isfinite(native.logits).all()) and bool(torch.isfinite(native.latent_states).all()) and bool(torch.isfinite(wrapped.logits).all()) and bool(torch.isfinite(wrapped.latent_states).all());delta=float((native.logits[:,-1].float()-wrapped.logits[:,-1]).abs().max());native_token=int(native.logits[0,-1].argmax());wrapped_token=int(wrapped.logits[0,-1].argmax());steps.append({'step':step,'generated_token_exact':native_token==wrapped_token,'native_stop':native_token in stops,'wrapped_stop':wrapped_token in stops,'max_logit_abs_error':delta,'all_outputs_finite':finite})
    if not finite or not torch.isfinite(torch.tensor(delta)) or native_token!=wrapped_token or (native_token in stops)!=(wrapped_token in stops) or delta>.25:raise RuntimeError({'depth':depth,'steps':steps})
    if native_token in stops:break
    position=prompt.shape[1]+step;current=torch.tensor([[native_token]],device='cuda')
   cache_checks.append({'depth':depth,'steps':steps})
 prompt_lengths=[]
 for i in range(c['test_ids'][0],c['test_ids'][1]+1):prompt_lengths.append((tokenize_prompt(tok,build_chat_prompt(tok,test[i]['question'],c['system_instruction']))['input_ids'].shape[1],i))
 longest_length,longest_id=max(prompt_lengths);longest=tokenize_prompt(tok,build_chat_prompt(tok,test[longest_id]['question'],c['system_instruction']))['input_ids'].cuda();long_schedule=materialize_h0_schedule(huginn,device=longest.device,example_id=longest_id,base_seed=c['h0_base_seed']);torch.cuda.reset_peak_memory_stats();cache=None;current=longest;position=None;d512_tokens=[];torch.cuda.synchronize();started=torch.cuda.Event(enable_timing=True);ended=torch.cuda.Event(enable_timing=True);started.record()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for step in range(32):
   input_h0=long_schedule[:,:current.shape[1]] if position is None else long_schedule[:,position:position+1];cp=None if position is None else torch.tensor([position],device='cuda');out=model(current,input_h0,depth=512,past_key_values=cache,use_cache=True,cache_position=cp);cache=out.past_key_values
   if not bool(torch.isfinite(out.logits).all()) or not bool(torch.isfinite(out.latent_states).all()):raise FloatingPointError('D512 preflight non-finite')
   token=out.logits[:,-1].argmax(-1,keepdim=True);d512_tokens.append(int(token));position=longest.shape[1]+step;current=token
 ended.record();torch.cuda.synchronize();elapsed_seconds=started.elapsed_time(ended)/1000;peak=int(torch.cuda.max_memory_allocated());cache_bytes=tensor_bytes(cache);observed_tokens=longest_length+31;projected_cache_bytes=cache_bytes*(longest_length+c['max_new_tokens'])/observed_tokens;projected_peak=peak-cache_bytes+projected_cache_bytes
 worst_case_hours=(elapsed_seconds/32)*250*c['max_new_tokens']*(sum(c['depths'])/512)/3600
 if projected_peak>c['d512_max_projected_peak_bytes']:raise RuntimeError(f'D512 projected memory exceeds limit: {projected_peak}')
 if worst_case_hours>c['d512_max_projected_worst_case_hours']:raise RuntimeError(f'projected runtime exceeds limit: {worst_case_hours} hours')
 unchanged=before==sha(huginn);result={'protocol':c['protocol']+'-correctness-gate','status':'pass','native_parity':native_checks,'cache_vs_full_prefix':cache_checks,'d512_preflight':{'longest_prompt_example_id':longest_id,'longest_prompt_tokens':longest_length,'generated_tokens':d512_tokens,'finite':bool(torch.isfinite(out.logits).all()),'elapsed_seconds_32_steps':elapsed_seconds,'seconds_per_step':elapsed_seconds/32,'peak_memory_bytes':peak,'observed_cache_bytes':cache_bytes,'projected_peak_bytes_at_prompt_plus_cap':projected_peak,'projected_worst_case_all_depths_hours':worst_case_hours,'memory_headroom_threshold_bytes':c['d512_max_projected_peak_bytes'],'runtime_viability_threshold_hours':c['d512_max_projected_worst_case_hours']},'huginn_weights_unchanged':unchanged,'new_parameters':0,'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),'git_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip(),'full_vocabulary_logits_persisted':False}
 if not all(all(r[k] for k in ('logits_bitwise_exact','latents_bitwise_exact','fixed_point_residual_finite')) for r in native_checks) or not result['d512_preflight']['finite'] or not unchanged:raise RuntimeError(result)
 publish(output,result);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
