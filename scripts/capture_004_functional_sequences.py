#!/usr/bin/env python3
"""Generate and cache frozen Huginn D16 teacher continuations for functional KL."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import numpy as np,torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from src.evaluation.correctness import SEED_PROTOCOL,build_chat_prompt,ensure_new_artifact_root,generation_config_kwargs,generation_status,require_legacy_artifact_opt_in,seed_for_example,stop_token_ids,tokenize_prompt

def main(config,output,allow_legacy_rebuild=False):
 require_legacy_artifact_opt_in(allow_legacy_rebuild, Path(__file__).name)
 ensure_new_artifact_root(output)
 c=json.loads(config.read_text()); max_new_tokens=int(c.get('max_new_tokens',1024)); ds=load_dataset(c['dataset_id'],c['dataset_config'],split=c['dataset_split'],revision=c['dataset_revision']); tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision']); model=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True).eval().cuda()
 for split,(start,end) in c['splits'].items():
  d=output/split; d.mkdir(exist_ok=True)
  for i in range(start,end+1):
   path=d/f'{i:05d}.npz'
   derived_seed=seed_for_example(c['seed'],i); text=build_chat_prompt(tok,ds[i]['question'],c['system_instruction']); enc={k:v.cuda() for k,v in tokenize_prompt(tok,text).items()}; prompt_len=int(enc['input_ids'].shape[-1])
   with torch.inference_mode(): out=model.generate(**enc,generation_config=GenerationConfig(**generation_config_kwargs(tok,max_new_tokens)),num_steps=16,tokenizer=tok)
   generated=out.sequences[0][prompt_len:]; status=generation_status(generated.tolist(),max_new_tokens=max_new_tokens,stop_token_ids=stop_token_ids(tok)); seq=out.sequences[0].cpu().numpy().astype(np.int32); valid_end=len(seq); np.savez(path,input_ids=seq,answer_start=np.array(prompt_len,dtype=np.int32),valid_end=np.array(valid_end,dtype=np.int32),generated_tokens=np.array(len(seq)-prompt_len,dtype=np.int32),ended_naturally=np.array(status.ended_naturally,dtype=np.bool_),truncated=np.array(status.hit_max_new_tokens,dtype=np.bool_),seed=np.array(derived_seed,dtype=np.int64),seed_protocol=np.array(SEED_PROTOCOL)); print(f'{split} {i}: prompt={prompt_len} generated={len(seq)-prompt_len}',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004b_fixed_k.json')); p.add_argument('--output',type=Path,default=Path('results/004_functional/teacher_sequences')); p.add_argument('--allow-legacy-rebuild',action='store_true',help='explicit fresh rebuild to a new directory only'); a=p.parse_args(); main(a.config,a.output,a.allow_legacy_rebuild)
