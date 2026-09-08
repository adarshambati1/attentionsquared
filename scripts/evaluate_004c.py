#!/usr/bin/env python3
"""4C frozen-coda and GSM8K evaluation for stable fixed-K A² models."""
from __future__ import annotations
import argparse,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from src.evaluation.gsm8k import extract_answer
from src.models.attention2 import Attention2Lite

def make_prompt(tok,q,s): return tok.apply_chat_template([{'role':'system','content':s},{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
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
 enc=tok(text,return_tensors='pt',add_special_tokens=False); enc.pop('token_type_ids',None); enc={k:v.cuda() for k,v in enc.items()}; t=time.perf_counter()
 with torch.inference_mode(): out=model.generate(**enc,generation_config=GenerationConfig(max_new_tokens=1024,stop_strings=['<|end_text|>','<|end_turn|>'],do_sample=False,use_cache=True,return_dict_in_generate=True,return_legacy_cache=False,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id or tok.eos_token_id),num_steps=16,tokenizer=tok)
 return tok.decode(out.sequences[0][enc['input_ids'].shape[-1]:],skip_special_tokens=False),time.perf_counter()-t
def main(config,cache,models,output):
 c=json.loads(config.read_text()); device=torch.device('cuda'); tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision']); ds=load_dataset(c['dataset_id'],c['dataset_config'],split=c['dataset_split'],revision=c['dataset_revision']); paths=sorted((cache/'test').glob('*.npz')); results=[]
 for k in [1,2,4]:
  a2=load_a2(models/f'K{k}'/'model.pt',device); hug=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True).eval().cuda(); cos=[]; rel=[]; kl=[]
  for path in paths:
   with np.load(path) as a: h0=torch.from_numpy(a['h0'].astype('float32')).cuda(); x=torch.from_numpy(a['x'].astype('float32')).cuda(); target=torch.from_numpy(a['trajectory'][-1].astype('float32')).cuda(); ids=torch.from_numpy(a['input_ids']).cuda().unsqueeze(0)
   with torch.inference_mode(): z,_,_=a2(h0[None],x[None]); pred=z[:,15]; cos.append(torch.nn.functional.cosine_similarity(pred,target,dim=-1).mean().item()); rel.append(((pred-target).norm(dim=-1)/(target.norm(dim=-1)+1e-8)).mean().item()); q=hug(input_ids=ids,input_states=target.to(torch.bfloat16).unsqueeze(0),num_steps=0,use_cache=False).logits.float(); r=hug(input_ids=ids,input_states=pred.to(torch.bfloat16),num_steps=0,use_cache=False).logits.float(); kl.append(torch.nn.functional.kl_div(torch.log_softmax(r,-1),torch.softmax(q,-1),reduction='batchmean').item())
  original=patch(hug,a2); correct=0; lat=[]
  for n,i in enumerate(range(c['splits']['test'][0],c['splits']['test'][1]+1),1):
   text=make_prompt(tok,ds[i]['question'],c['system_instruction']); generated,seconds=generate(hug,tok,text); lat.append(seconds); correct += extract_answer(generated,allow_fallback=True)==extract_answer(ds[i]['answer'],allow_fallback=True)
   if n%25==0: print(f'K={k} generation {n}/250',flush=True)
  hug.iterate_forward=original; result={'K':k,'endpoint_cosine':float(np.mean(cos)),'endpoint_relative_l2':float(np.mean(rel)),'teacher_logit_kl':float(np.mean(kl)),'gsm8k_accuracy':correct/250,'mean_generation_latency_seconds':float(np.mean(lat)),'examples':250}; results.append(result); print(result,flush=True); del hug,a2; torch.cuda.empty_cache()
 output.write_text(json.dumps(results,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004b_fixed_k.json')); p.add_argument('--cache',type=Path,default=Path('results/004b_fixed_k/cache')); p.add_argument('--models',type=Path,default=Path('results/004b_fixed_k/models')); p.add_argument('--output',type=Path,default=Path('results/004b_fixed_k/evaluation_4c.json')); a=p.parse_args(); main(a.config,a.cache,a.models,a.output)
