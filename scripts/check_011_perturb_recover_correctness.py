#!/usr/bin/env python3
"""Correctness gate for the parameter-free perturb-and-recover experiment."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.correctness import build_chat_prompt,tokenize_prompt
from src.models.perturbed_huginn import PerturbedHuginn,derived_noise_seed,token_relative_noise
from src.training.latent_history import materialize_h0_schedule
from scripts.run_011_perturb_recover_latent import trajectory
DEFAULT_CONFIG=Path('configs/011_perturb_recover.json')
def sha(module):
 d=hashlib.sha256()
 for n,v in module.state_dict().items():d.update(n.encode());d.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return d.hexdigest()
def main():
 from datasets import load_dataset;from transformers import AutoModelForCausalLM,AutoTokenizer
 parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG);args=parser.parse_args();config_path=ROOT/args.config;c=json.loads(config_path.read_text());output=Path(c['output_root'])/'correctness_gate.json';output.parent.mkdir(exist_ok=True)
 if output.exists():raise FileExistsError(output)
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);test=load_dataset(c['dataset_id'],c['dataset_config'],split='test',revision=c['dataset_revision']);huginn=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();wrapper=PerturbedHuginn(huginn).cuda().eval();prompt=tokenize_prompt(tok,build_chat_prompt(tok,test[0]['question'],c['system_instruction']))['input_ids'].cuda();schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=0,base_seed=c['h0_base_seed']);h0=schedule[:,:prompt.shape[1]];before=sha(huginn)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  baseline=huginn(input_ids=prompt,input_states=h0,attention_mask=torch.ones_like(prompt,dtype=torch.bool),num_steps=16,use_cache=False,return_dict=True,output_details={'return_logits':True,'return_latents':True,'return_head':False,'return_stats':False});zero=wrapper(prompt,h0,depth=16,perturb_depth=8,sigma=0.0,example_id=0,perturbation_seed_index=0,perturbation_base_seed=c['perturbation_base_seed']);states,_,_,_=trajectory(huginn,prompt,h0,33)
 zero_exact={'logits':torch.equal(baseline.logits.float(),zero.logits),'latents':torch.equal(baseline.latent_states,zero.latent_states)}
 native={}
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for depth in (8,16,32):
   out=huginn(input_ids=prompt,input_states=h0,attention_mask=torch.ones_like(prompt,dtype=torch.bool),num_steps=depth,use_cache=False,return_dict=True,output_details={'return_logits':False,'return_latents':True,'return_head':False,'return_stats':False});native[str(depth)]=torch.equal(huginn.transformer.ln_f(states[depth]),out.latent_states)
 positions=torch.arange(prompt.shape[1],device='cuda').unsqueeze(0);perturbations={};direction_cosines={};seed_direction_cosines={}
 if states[4].dtype!=torch.float32 or states[8].dtype!=torch.float32:raise RuntimeError('native h4/h8 must be FP32 for the frozen perturbation protocol')
 for depth in c['perturb_depths']:
  seed_deltas={}
  for seed in c['perturbation_seed_indices']:
   deltas={}
   for sigma in c['noise_strengths']:
    a,relative=token_relative_noise(states[depth],sigma=sigma,example_id=0,perturb_depth=depth,seed_index=seed,base_seed=c['perturbation_base_seed'],token_positions=positions);b,_=token_relative_noise(states[depth],sigma=sigma,example_id=0,perturb_depth=depth,seed_index=seed,base_seed=c['perturbation_base_seed'],token_positions=positions);key=f'h{depth}:seed{seed}:sigma{sigma}';perturbations[key]={'realized_dtype':str(a.dtype),'deterministic_bitwise':torch.equal(a,b),'mean_relative_norm':float(relative.mean()),'max_relative_norm_error':float((relative-sigma).abs().max())};deltas[sigma]=a-states[depth].float()
   direction_cosines[f'h{depth}:seed{seed}']=min(float(torch.nn.functional.cosine_similarity(deltas[a],deltas[b],dim=-1).mean()) for a,b in ((.01,.05),(.01,.1),(.05,.1)));seed_deltas[seed]=deltas[.05]
  seed_direction_cosines[f'h{depth}']=[float(torch.nn.functional.cosine_similarity(seed_deltas[a],seed_deltas[b],dim=-1).mean()) for a,b in ((0,1),(0,2),(1,2))]
 cache_checks=[]
 for sigma in c['noise_strengths']:
  for seed in c['perturbation_seed_indices']:
   cache=None;current=prompt;prefix=prompt;position=None;condition={'sigma':sigma,'seed':seed,'steps':[]}
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for step in range(3):
     input_h0=schedule[:,:current.shape[1]] if position is None else schedule[:,position:position+1];cp=None if position is None else torch.tensor([position],device='cuda');kwargs={'depth':16,'perturb_depth':8,'sigma':sigma,'example_id':0,'perturbation_seed_index':seed,'perturbation_base_seed':c['perturbation_base_seed']};cached=wrapper(current,input_h0,past_key_values=cache,use_cache=True,cache_position=cp,**kwargs);cache=cached.past_key_values;full=wrapper(prefix,schedule[:,:prefix.shape[1]],**kwargs);delta=float((cached.logits[:,-1]-full.logits[:,-1]).abs().max());same=torch.equal(cached.logits[:,-1].argmax(-1),full.logits[:,-1].argmax(-1));condition['steps'].append({'step':step,'argmax_exact':same,'max_logit_abs_error':delta})
     if not same or delta>.25:raise RuntimeError(condition)
     token=cached.logits[:,-1].argmax(-1,keepdim=True);prefix=torch.cat((prefix,token),1);position=prefix.shape[1]-1;current=token
   cache_checks.append(condition)
 result={'protocol':c['protocol']+'-correctness-gate','status':'pass','sigma_zero_native_exact':zero_exact,'manual_trajectory_native_exact':native,'perturbations':perturbations,'same_direction_across_sigma_min_cosines':direction_cosines,'distinct_seed_direction_cosines':seed_direction_cosines,'seed_derivation':'sha256(base_seed,example_id,perturb_depth,seed_index,absolute_token_position)->63-bit seed','cache_vs_full_prefix':cache_checks,'huginn_weights_unchanged':before==sha(huginn),'new_parameters':0,'git_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip(),'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'full_vocabulary_logits_persisted':False}
 if not all(zero_exact.values()) or not all(native.values()) or not result['huginn_weights_unchanged'] or not all(v['deterministic_bitwise'] and v['max_relative_norm_error']<1e-5 for v in perturbations.values()) or not all(v>.99999 for v in direction_cosines.values()) or not all(abs(v)<.05 for values in seed_direction_cosines.values() for v in values):raise RuntimeError(result)
 data=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode();temporary=output.with_name(f'.{output.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp')
 with temporary.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
 try:os.link(temporary,output)
 finally:temporary.unlink(missing_ok=True)
 print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
