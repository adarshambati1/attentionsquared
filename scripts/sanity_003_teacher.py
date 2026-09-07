#!/usr/bin/env python3
import argparse, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from src.evaluation.gsm8k import extract_answer

def main(cfg_path, output):
 c=json.loads(cfg_path.read_text()); tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision']); ds=load_dataset(c['dataset_id'],c['dataset_config'],split=c['dataset_split'],revision=c['dataset_revision']); model=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True).eval().cuda(); correct=0; lat=[]
 for n,i in enumerate(range(c['splits']['test'][0],c['splits']['test'][1]+1),1):
  messages=[{'role':'system','content':c['system_instruction']},{'role':'user','content':ds[i]['question']}]; text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True); enc=tok(text,return_tensors='pt',add_special_tokens=False); enc.pop('token_type_ids',None); enc={k:v.cuda() for k,v in enc.items()}; t=time.perf_counter()
  with torch.inference_mode(): out=model.generate(**enc,generation_config=GenerationConfig(max_new_tokens=1024,stop_strings=['<|end_text|>','<|end_turn|>'],do_sample=False,use_cache=True,return_dict_in_generate=True,return_legacy_cache=False,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id or tok.eos_token_id),num_steps=16,tokenizer=tok)
  text_out=tok.decode(out.sequences[0][enc['input_ids'].shape[-1]:],skip_special_tokens=False); lat.append(time.perf_counter()-t); correct += extract_answer(text_out,allow_fallback=True)==extract_answer(ds[i]['answer'],allow_fallback=True)
  if n%10==0: print(f'teacher generation {n}/250',flush=True)
 result={'model':'huginn_d16_teacher','examples':250,'gsm8k_accuracy':correct/250,'mean_generation_latency_seconds':sum(lat)/len(lat)}; output.write_text(json.dumps(result,indent=2)); print(result,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/003_direct_jump.json')); p.add_argument('--output',type=Path,default=Path('results/003_direct_jump/teacher_control.json')); a=p.parse_args(); main(a.config,a.output)
