#!/usr/bin/env python3
"""Cache full frozen Huginn trajectories for 4B."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from types import MethodType
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer

def main(config,output):
 c=json.loads(config.read_text()); ds=load_dataset(c['dataset_id'],c['dataset_config'],split=c['dataset_split'],revision=c['dataset_revision']); tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision']); model=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True).eval().cuda(); output.mkdir(parents=True,exist_ok=True)
 for split,(start,end) in c['splits'].items():
  d=output/split; d.mkdir(exist_ok=True)
  for i in range(start,end+1):
   path=d/f'{i:05d}.npz'
   if path.exists(): continue
   seed=c['seed']+i; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); msgs=[{'role':'system','content':c['system_instruction']},{'role':'user','content':ds[i]['question']}]; text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True); enc=tok(text,return_tensors='pt',add_special_tokens=False); enc.pop('token_type_ids',None); enc={k:v.cuda() for k,v in enc.items()};
   if enc['input_ids'].shape[-1]>c['max_prompt_tokens']: raise ValueError(i)
   states=[]; captured={}; original=model.core_block_forward
   def wrapped(this,x,input_embeds,*args,**kwargs):
    if not states: captured['h0']=x.detach().float().cpu(); captured['x']=input_embeds.detach().float().cpu()
    r=original(x,input_embeds,*args,**kwargs); states.append(r[0].detach().float().cpu()); return r
   model.core_block_forward=MethodType(wrapped,model)
   try:
    with torch.inference_mode(): model(**enc,num_steps=16,use_cache=False,output_details={'return_logits':False,'return_latents':False,'return_head':False,'return_stats':False})
   finally: model.core_block_forward=original
   traj=torch.cat([captured['h0'],*states],0)[0].numpy().astype(np.float16)
   np.savez(path,h0=captured['h0'][0].numpy().astype(np.float16),x=captured['x'][0].numpy().astype(np.float16),trajectory=traj[1:],input_ids=enc['input_ids'].cpu().numpy()[0])
   print(f'{split} {i}: T={traj.shape[1]}',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004b_fixed_k.json')); p.add_argument('--output',type=Path,default=Path('results/004b_fixed_k/cache')); a=p.parse_args(); main(a.config,a.output)
