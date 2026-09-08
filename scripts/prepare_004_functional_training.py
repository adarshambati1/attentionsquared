#!/usr/bin/env python3
"""Prepare full-sequence h0/x and detached teacher logits for functional KL."""
from __future__ import annotations
import argparse,json,gc
from pathlib import Path
from types import MethodType
import numpy as np,torch
from transformers import AutoModelForCausalLM

def main(config,seq_dir,out):
 c=json.loads(config.read_text()); model=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True).eval().cuda(); out.mkdir(parents=True,exist_ok=True)
 for split in ('train','val','test'):
  od=out/split; od.mkdir(exist_ok=True)
  for sp in sorted((seq_dir/split).glob('*.npz')):
   dp=od/sp.name
   if dp.exists(): continue
   with np.load(sp) as a: ids=torch.from_numpy(a['input_ids']).cuda().unsqueeze(0); start=int(a['answer_start']); end=int(a['valid_end']) if 'valid_end' in a.files else len(a['input_ids'])
   captured={}; original=model.core_block_forward
   def wrapped(this,x,input_embeds,*args,**kwargs):
    if 'h0' not in captured:
     captured['h0']=x.detach().float().cpu()[0]; captured['x']=input_embeds.detach().float().cpu()[0]
    return original(x,input_embeds,*args,**kwargs)
   model.core_block_forward=MethodType(wrapped,model)
   try:
    torch.manual_seed(c['seed']+int(sp.stem)); torch.cuda.manual_seed_all(c['seed']+int(sp.stem))
    with torch.inference_mode(): result=model(input_ids=ids,num_steps=16,use_cache=False)
   finally: model.core_block_forward=original
   logits=result.logits[0,start:end].float().cpu().numpy().astype(np.float16)
   np.savez(dp,h0=captured['h0'].numpy().astype(np.float16),x=captured['x'].numpy().astype(np.float16),teacher_logits=logits,input_ids=ids[0].cpu().numpy().astype(np.int32),answer_start=np.array(start,dtype=np.int32),valid_end=np.array(end,dtype=np.int32))
   seq_len=len(captured['h0'])
   del result, logits, captured, states, enc, ids
   torch.cuda.empty_cache(); gc.collect()
   print(f'{split} {sp.stem}: T={seq_len} answer_tokens={end-start}',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004b_fixed_k.json')); p.add_argument('--sequences',type=Path,default=Path('results/004_functional/teacher_sequences')); p.add_argument('--output',type=Path,default=Path('results/004_functional/training_cache')); a=p.parse_args(); main(a.config,a.sequences,a.output)
