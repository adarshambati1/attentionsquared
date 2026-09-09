#!/usr/bin/env python3
"""Cache full frozen Huginn trajectories for 4B."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from types import MethodType
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
from src.evaluation.correctness import SEED_PROTOCOL,build_chat_prompt,ensure_new_artifact_root,seed_for_example,tokenize_prompt,write_json_exclusive

def main(config,output):
 ensure_new_artifact_root(output)
 c=json.loads(config.read_text()); derived_seeds=[]
 ds=load_dataset(c['dataset_id'],c['dataset_config'],split=c['dataset_split'],revision=c['dataset_revision']); tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision']); model=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True).eval().cuda()
 for split,(start,end) in c['splits'].items():
  d=output/split; d.mkdir(exist_ok=True)
  for i in range(start,end+1):
   path=d/f'{i:05d}.npz'
   derived_seed=seed_for_example(c['seed'],i); derived_seeds.append({'example_id':i,'seed':derived_seed}); text=build_chat_prompt(tok,ds[i]['question'],c['system_instruction']); enc={k:v.cuda() for k,v in tokenize_prompt(tok,text).items()};
   if enc['input_ids'].shape[-1]>c['max_prompt_tokens']: raise ValueError(i)
   states=[]; captured={}; original=model.core_block_forward
   def wrapped(this,x,input_embeds,*args,**kwargs):
    if not states:
     hx=x.detach().float().cpu(); ex=input_embeds.detach().float().cpu(); captured['h0']=hx[0] if hx.ndim==3 else hx; captured['x']=ex[0] if ex.ndim==3 else ex
    r=original(x,input_embeds,*args,**kwargs); state=r[0].detach().float().cpu(); states.append(state[0] if state.ndim==3 else state); return r
   model.core_block_forward=MethodType(wrapped,model)
   try:
    with torch.inference_mode(): model(**enc,num_steps=16,use_cache=False,output_details={'return_logits':False,'return_latents':False,'return_head':False,'return_stats':False})
   finally: model.core_block_forward=original
   traj=np.stack(states).astype(np.float16)
   np.savez(path,h0=captured['h0'].numpy().astype(np.float16),x=captured['x'].numpy().astype(np.float16),trajectory=traj,input_ids=enc['input_ids'].cpu().numpy()[0],base_seed=np.array(c['seed'],dtype=np.int64),derived_seed=np.array(derived_seed,dtype=np.int64),seed_protocol=np.array(SEED_PROTOCOL),example_id=np.array(i,dtype=np.int64),split=np.array(split),model_revision=np.array(c['model_revision']),dataset_revision=np.array(c['dataset_revision']),depth=np.array(16,dtype=np.int64))
   print(f'{split} {i}: shape={traj.shape}',flush=True)
 write_json_exclusive(output/'manifest.json',{'config':c,'seed_protocol':SEED_PROTOCOL,'derived_seeds':derived_seeds,'model_revision':c['model_revision'],'dataset_revision':c['dataset_revision']})
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004b_fixed_k.json')); p.add_argument('--output',type=Path,default=Path('results/004b_fixed_k/cache')); a=p.parse_args(); main(a.config,a.output)
