#!/usr/bin/env python3
"""QUARANTINED 4C evaluator: incremental one-token A² execution is invalid."""
from __future__ import annotations
import argparse,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from src.evaluation.correctness import INVALID_FUNCTIONAL_V1_LABEL,build_chat_prompt,ensure_audit_rerun_output,generation_config_kwargs,generation_status,require_invalid_v1_opt_in,score_generation,stop_token_ids,synchronized_cuda_timer,tokenize_prompt,write_json_exclusive
from src.models.attention2 import Attention2Lite
def load_a2(path,device):
 ck=torch.load(path,map_location='cpu'); c=ck['config']; m=Attention2Lite(5280,16,16,c['K'],1,16.0,False).to(device); m.load_state_dict(ck['state_dict']); return m.eval()
def patch(model,a2):
 original=model.iterate_forward
 def jumped(this,input_embeds,input_states,freqs_cis,block_idx,mask,past_key_values=None,num_steps=None,init_scale=1.0):
  h0=this.initialize_state(input_embeds,scale=init_scale) if input_states is None else input_states.clone()
  with torch.autocast(device_type='cuda',dtype=torch.bfloat16): z,_,_=a2(h0.float(),input_embeds.float())
  return this.transformer.ln_f(z[:,15].to(h0.dtype)),0,0,h0.detach(),block_idx
 model.iterate_forward=__import__('types').MethodType(jumped,model); return original
def generate(model,tok,text):
 enc={k:v.cuda() for k,v in tokenize_prompt(tok,text).items()}
 with synchronized_cuda_timer('cuda') as timing:
  with torch.inference_mode(): out=model.generate(**enc,generation_config=GenerationConfig(**generation_config_kwargs(tok,1024)),num_steps=16,tokenizer=tok)
 generated=out.sequences[0][enc['input_ids'].shape[-1]:]
 status=generation_status(generated.tolist(),max_new_tokens=1024,stop_token_ids=stop_token_ids(tok))
 return tok.decode(generated,skip_special_tokens=False),timing.seconds,status.hit_max_new_tokens
def main(config,cache,models,output,allow_invalid_v1=False):
 require_invalid_v1_opt_in(allow_invalid_v1, Path(__file__).name)
 ensure_audit_rerun_output(output)
 c=json.loads(config.read_text()); device=torch.device('cuda'); tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision']); ds=load_dataset(c['dataset_id'],c['dataset_config'],split=c['dataset_split'],revision=c['dataset_revision']); paths=sorted((cache/'test').glob('*.npz')); results=[]
 for k in [1,2,4]:
  a2=load_a2(models/f'K{k}'/'model.pt',device); hug=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True).eval().cuda(); cos=[]; rel=[]; kl=[]
  for path in paths:
   with np.load(path) as a: h0=torch.from_numpy(a['h0'].astype('float32')).cuda(); x=torch.from_numpy(a['x'].astype('float32')).cuda(); target=torch.from_numpy(a['trajectory'][-1].astype('float32')).cuda(); ids=torch.from_numpy(a['input_ids']).cuda().unsqueeze(0)
   with torch.inference_mode(): z,_,_=a2(h0[None],x[None]); pred=z[:,15]; cos.append(torch.nn.functional.cosine_similarity(pred,target,dim=-1).mean().item()); rel.append(((pred-target).norm(dim=-1)/(target.norm(dim=-1)+1e-8)).mean().item()); q=hug(input_ids=ids,input_states=target.to(torch.bfloat16).unsqueeze(0),num_steps=0,use_cache=False).logits.float(); r=hug(input_ids=ids,input_states=pred.to(torch.bfloat16),num_steps=0,use_cache=False).logits.float(); kl.append(torch.nn.functional.kl_div(torch.log_softmax(r,-1),torch.softmax(q,-1),reduction='batchmean').item())
  original=patch(hug,a2); correct=0; lat=[]
  for n,i in enumerate(range(c['splits']['test'][0],c['splits']['test'][1]+1),1):
   text=build_chat_prompt(tok,ds[i]['question'],c['system_instruction']); generated,seconds,hit_cap=generate(hug,tok,text); lat.append(seconds); correct += score_generation(generated,ds[i]['answer'],hit_max_new_tokens=hit_cap).correct
   if n%25==0: print(f'K={k} generation {n}/250',flush=True)
  hug.iterate_forward=original; result={'K':k,'scientific_validity':INVALID_FUNCTIONAL_V1_LABEL,'execution_protocol':'INVALID_incremental_one_token_A2_without_KV_cache','timing_protocol':'cuda-synchronized_but_execution_invalid','kl_protocol':'historical_batchmean_not_token_normalized','endpoint_cosine':float(np.mean(cos)),'endpoint_relative_l2':float(np.mean(rel)),'teacher_logit_kl':float(np.mean(kl)),'gsm8k_accuracy':correct/250,'mean_generation_latency_seconds':float(np.mean(lat)),'examples':250}; results.append(result); print(result,flush=True); del hug,a2; torch.cuda.empty_cache()
 write_json_exclusive(output,results)
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004b_fixed_k.json')); p.add_argument('--cache',type=Path,default=Path('results/004b_fixed_k/cache')); p.add_argument('--models',type=Path,default=Path('results/004b_fixed_k/models')); p.add_argument('--output',type=Path,default=Path('results/004b_fixed_k/evaluation_4c.json')); p.add_argument('--allow-invalid-v1',action='store_true',help='audit rerun only; outputs remain scientifically invalid'); a=p.parse_args(); main(a.config,a.cache,a.models,a.output,a.allow_invalid_v1)
