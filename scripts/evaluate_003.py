#!/usr/bin/env python3
"""Evaluate Exp3 latent fidelity, frozen-coda KL, and greedy GSM8K generation."""
from __future__ import annotations
import argparse, json, re, time, sys
from pathlib import Path
from types import MethodType
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset
from train_003_predictors import JumpMLP

NUM_RE=re.compile(r"####\s*([-+]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")
def answer(text):
 m=NUM_RE.findall(text.replace("\u202f","")); return m[-1].replace(",","").replace("$","") if m else None

def prompt(tok,q,system):
 return tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":q}],tokenize=False,add_generation_prompt=True)

def load_model(cfg):
 return AutoModelForCausalLM.from_pretrained(cfg["model_id"],revision=cfg["model_revision"],torch_dtype=torch.bfloat16,trust_remote_code=True).eval().cuda()

def patch_jump(model,predictor):
 original=model.iterate_forward
 def jumped(this,input_embeds,input_states,freqs_cis,block_idx,mask,past_key_values=None,num_steps=None,init_scale=1.0):
  h0=this.initialize_state(input_embeds,scale=init_scale) if input_states is None else input_states.clone()
  pred=predictor(h0.float(),input_embeds.float()).to(h0.dtype)
  return this.transformer.ln_f(pred),0,0,h0.detach(),block_idx
 model.iterate_forward=MethodType(jumped,model)
 return original

def generate(model,tok,text,depth):
 enc=tok(text,return_tensors="pt",add_special_tokens=False); enc.pop("token_type_ids",None); enc={k:v.cuda() for k,v in enc.items()}
 with torch.inference_mode():
  out=model.generate(**enc,generation_config=GenerationConfig(max_new_tokens=1024,do_sample=False,use_cache=True,return_dict_in_generate=True,return_legacy_cache=False,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id or tok.eos_token_id),num_steps=depth,tokenizer=tok)
 seq=out.sequences[0]; return tok.decode(seq[enc["input_ids"].shape[-1]:],skip_special_tokens=False)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,default=Path("configs/003_direct_jump.json")); ap.add_argument("--cache",type=Path,default=Path("results/003_direct_jump/cache")); ap.add_argument("--models",type=Path,default=Path("results/003_direct_jump/models")); ap.add_argument("--output",type=Path,default=Path("results/003_direct_jump/evaluation.json")); args=ap.parse_args(); cfg=json.loads(args.config.read_text())
 tok=AutoTokenizer.from_pretrained(cfg["model_id"],revision=cfg["model_revision"]); ds=load_dataset(cfg["dataset_id"],cfg["dataset_config"],split=cfg["dataset_split"],revision=cfg["dataset_revision"])
 device=torch.device("cuda"); results=[]
 for name in ("direct","residual"):
  ck=torch.load(args.models/f"{name}.pt",map_location="cpu"); p=JumpMLP(ck["hidden"],ck["residual"]); p.load_state_dict(ck["state_dict"]); p.to(device).eval(); model=load_model(cfg)
  cos=[]; rel=[]; kl=[]
  for path in sorted((args.cache/"test").glob("*.npz")):
   with np.load(path) as a: h0=torch.from_numpy(a["h0"].astype(np.float32)).cuda(); x=torch.from_numpy(a["x"].astype(np.float32)).cuda(); target=torch.from_numpy(a["h16"].astype(np.float32)).cuda(); ids=torch.from_numpy(a["input_ids"]).cuda().unsqueeze(0)
   with torch.inference_mode(): pred=p(h0,x); cos.append(torch.nn.functional.cosine_similarity(pred,target,dim=-1).mean().item()); rel.append(((pred-target).norm(dim=-1)/(target.norm(dim=-1)+1e-8)).mean().item()); direct_out=model(input_ids=ids,input_states=pred.to(torch.bfloat16).unsqueeze(0),num_steps=0,use_cache=False); teacher_out=model(input_ids=ids,input_states=target.to(torch.bfloat16).unsqueeze(0),num_steps=0,use_cache=False); q=torch.softmax(teacher_out.logits.float(),-1); kl.append(torch.nn.functional.kl_div(torch.log_softmax(direct_out.logits.float(),-1),q,reduction="batchmean").item())
  original=patch_jump(model,p)
  correct=0; total=0; lat=[]
  for i in range(cfg["splits"]["test"][0],cfg["splits"]["test"][1]+1):
   text=prompt(tok,ds[i]["question"],cfg["system_instruction"]); t=time.perf_counter(); generated=generate(model,tok,text,16); lat.append(time.perf_counter()-t); correct += answer(generated)==answer(ds[i]["answer"]); total += 1
  model.iterate_forward=original
  results.append({"model":name,"latent_cosine":float(np.mean(cos)),"latent_relative_l2":float(np.mean(rel)),"teacher_logit_kl":float(np.mean(kl)),"gsm8k_accuracy":correct/total,"mean_generation_latency_seconds":float(np.mean(lat)),"examples":total})
  del model,p; torch.cuda.empty_cache(); print(results[-1],flush=True)
 args.output.write_text(json.dumps(results,indent=2))
if __name__=="__main__": main()
