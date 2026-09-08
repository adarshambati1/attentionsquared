#!/usr/bin/env python3
"""Generate and cache frozen Huginn D16 teacher continuations for functional KL."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig

def main(config,output):
 c=json.loads(config.read_text()); ds=load_dataset(c['dataset_id'],c['dataset_config'],split=c['dataset_split'],revision=c['dataset_revision']); tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision']); model=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True).eval().cuda(); output.mkdir(parents=True,exist_ok=True)
 for split,(start,end) in c['splits'].items():
  d=output/split; d.mkdir(exist_ok=True)
  for i in range(start,end+1):
   path=d/f'{i:05d}.npz'
   if path.exists(): continue
   seed=c['seed']+i; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); msgs=[{'role':'system','content':c['system_instruction']},{'role':'user','content':ds[i]['question']}]; text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True); enc=tok(text,return_tensors='pt',add_special_tokens=False); enc.pop('token_type_ids',None); enc={k:v.cuda() for k,v in enc.items()}; prompt_len=int(enc['input_ids'].shape[-1])
   with torch.inference_mode(): out=model.generate(**enc,generation_config=GenerationConfig(max_new_tokens=1024,stop_strings=['<|end_text|>','<|end_turn|>'],do_sample=False,use_cache=True,return_dict_in_generate=True,return_legacy_cache=False,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id or tok.eos_token_id),num_steps=16,tokenizer=tok)
   seq=out.sequences[0].cpu().numpy().astype(np.int32); np.savez(path,input_ids=seq,answer_start=np.array(prompt_len,dtype=np.int32),generated_tokens=np.array(len(seq)-prompt_len,dtype=np.int32)); print(f'{split} {i}: prompt={prompt_len} generated={len(seq)-prompt_len}',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004b_fixed_k.json')); p.add_argument('--output',type=Path,default=Path('results/004_functional/teacher_sequences')); a=p.parse_args(); main(a.config,a.output)
