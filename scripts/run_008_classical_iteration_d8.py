#!/usr/bin/env python3
"""Validation-selected classical D8 iteration rules and held-out comparison."""
from __future__ import annotations
import argparse, hashlib, json, math, os, statistics, subprocess, sys, time
from pathlib import Path
from typing import Any
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.evaluation.correctness import build_chat_prompt, find_repetition_onset, generation_status, score_generation, stop_token_ids, tokenize_prompt
from src.models.classical_iteration_huginn import ClassicalIterationHuginn
from src.training.latent_history import materialize_h0_schedule

CONFIG=Path("configs/008_classical_iteration_d8.json")

def param_sha(model):
 d=hashlib.sha256()
 for n,p in model.state_dict().items(): d.update(n.encode());d.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return d.hexdigest()

def write_exclusive(path,payload):
 data=(json.dumps(payload,indent=2,sort_keys=True)+"\n").encode();fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
 with os.fdopen(fd,"wb") as f:f.write(data);f.flush();os.fsync(f.fileno())

def append_jsonl(path,payload):
 with path.open("a") as f:f.write(json.dumps(payload)+"\n");f.flush();os.fsync(f.fileno())

def method_kwargs(mode,parameter,config):
 if mode=="heavy_ball": return {"momentum":float(parameter)}
 if mode=="anderson": return {"anderson_window":int(parameter),"anderson_ridge":config["anderson_ridge"]}
 return {}

def generate(model,tokenizer,prompt,schedule,mode,parameter,config):
 cache=None;current=prompt;position=None;tokens=[];fallbacks=0;solve_attempts=0;successful_solves=0;singular_solves=0;nonfinite_solves=0;stops={int(x) for x in stop_token_ids(tokenizer) if x is not None and int(x)>=0};kwargs=method_kwargs(mode,parameter,config)
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
  for _ in range(config["max_new_tokens"]):
   if position is None: h0=schedule[:,:current.shape[1]];cache_pos=None
   else: h0=schedule[:,position:position+1];cache_pos=torch.tensor([position],device="cuda")
   out=model(current,h0,depth=8,mode=mode,past_key_values=cache,use_cache=True,cache_position=cache_pos,**kwargs);cache=out.past_key_values;fallbacks+=out.anderson_fallbacks;solve_attempts+=out.anderson_solve_attempts;successful_solves+=out.anderson_successful_solves;singular_solves+=out.anderson_singular_solves;nonfinite_solves+=out.anderson_nonfinite_solves
   token=int(out.logits[0,-1].argmax());tokens.append(token)
   if token in stops:break
   position=prompt.shape[1]+len(tokens)-1;current=torch.tensor([[token]],device="cuda",dtype=prompt.dtype)
 torch.cuda.synchronize();latency=time.perf_counter()-start;status=generation_status(tokens,max_new_tokens=config["max_new_tokens"],stop_token_ids=stop_token_ids(tokenizer))
 return {"token_ids":tokens,"text":tokenizer.decode(tokens,skip_special_tokens=False),"latency_seconds":latency,"generated_tokens":len(tokens),"hit_cap":status.hit_max_new_tokens,"ended_naturally":status.ended_naturally,"peak_memory_bytes":int(torch.cuda.max_memory_allocated()),"anderson_fallbacks":fallbacks,"anderson_solve_attempts":solve_attempts,"anderson_successful_solves":successful_solves,"anderson_singular_solves":singular_solves,"anderson_nonfinite_solves":nonfinite_solves}

def diagnostics(model,full_ids,schedule,mode,parameter,config):
 kwargs=method_kwargs(mode,parameter,config)
 with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):out=model(full_ids,schedule[:,:full_ids.shape[1]],depth=8,mode=mode,return_loop_states=True,**kwargs)
 states=out.loop_states;final=states[-1].float().reshape(-1);records=[]
 for k in range(8):
  a=states[k].float().reshape(-1);b=states[k+1].float().reshape(-1)
  ratio=float(torch.linalg.vector_norm(b-a)/torch.linalg.vector_norm(a).clamp_min(1e-12));cos=float(torch.nn.functional.cosine_similarity(a[None],final[None]).item())
  records.append({"loop":k,"relative_update_norm":ratio,"cosine_to_final_state":cos})
 return records

def run_examples(model,tokenizer,dataset,ids,mode,parameter,config,output_jsonl=None,with_diagnostics=False):
 records=[]
 for example_id in ids:
  prompt_text=build_chat_prompt(tokenizer,dataset[example_id]["question"],config["system_instruction"]);prompt=tokenize_prompt(tokenizer,prompt_text)["input_ids"].cuda();schedule=materialize_h0_schedule(model.huginn,device=prompt.device,example_id=example_id,base_seed=config["h0_base_seed"])
  result=generate(model,tokenizer,prompt,schedule,mode,parameter,config);score=score_generation(result["text"],dataset[example_id]["answer"],hit_max_new_tokens=result["hit_cap"])
  record={"example_id":example_id,"mode":mode,"parameter":parameter,"correct":score.correct,"predicted_answer":score.predicted_answer,"gold_answer":score.gold_answer,"generated_text":result["text"],"generated_token_ids":result["token_ids"],"generation_latency_seconds":result["latency_seconds"],"generated_tokens":result["generated_tokens"],"hit_max_new_tokens":result["hit_cap"],"ended_naturally":result["ended_naturally"],"peak_memory_bytes":result["peak_memory_bytes"],"anderson_fallbacks":result["anderson_fallbacks"],"anderson_solve_attempts":result["anderson_solve_attempts"],"anderson_successful_solves":result["anderson_successful_solves"],"anderson_singular_solves":result["anderson_singular_solves"],"anderson_nonfinite_solves":result["anderson_nonfinite_solves"],"repetition_onset":find_repetition_onset(result["token_ids"],chunk_size=config["degeneration_chunk_size"],repetitions=config["degeneration_repetitions"])}
  if with_diagnostics:
   full=torch.cat((prompt,torch.tensor([result["token_ids"]],device="cuda",dtype=prompt.dtype)),dim=1);record["loop_diagnostics"]=diagnostics(model,full,schedule,mode,parameter,config)
  records.append(record)
  if output_jsonl:append_jsonl(output_jsonl,record)
  print(json.dumps({k:v for k,v in record.items() if k not in ("generated_text","generated_token_ids","loop_diagnostics")}),flush=True)
 return records

def summarize(records):
 return {"correct":sum(r["correct"] for r in records),"total":len(records),"accuracy":sum(r["correct"] for r in records)/len(records),"cap_hit_rate":sum(r["hit_max_new_tokens"] for r in records)/len(records),"degeneration_rate":sum(r["repetition_onset"] is not None for r in records)/len(records),"mean_generated_tokens":sum(r["generated_tokens"] for r in records)/len(records),"mean_generation_latency_seconds":sum(r["generation_latency_seconds"] for r in records)/len(records),"median_generation_latency_seconds":statistics.median(r["generation_latency_seconds"] for r in records),"peak_memory_bytes":max(r["peak_memory_bytes"] for r in records),"anderson_fallbacks":sum(r["anderson_fallbacks"] for r in records),"anderson_solve_attempts":sum(r["anderson_solve_attempts"] for r in records),"anderson_successful_solves":sum(r["anderson_successful_solves"] for r in records),"anderson_singular_solves":sum(r["anderson_singular_solves"] for r in records),"anderson_nonfinite_solves":sum(r["anderson_nonfinite_solves"] for r in records)}

def fixed_benchmark(model,ids,schedule,mode,parameter,config):
 kwargs=method_kwargs(mode,parameter,config)
 for _ in range(config["fixed_forward_warmup"]):
  with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):model(ids,schedule[:,:ids.shape[1]],depth=8,mode=mode,**kwargs)
 times=[];torch.cuda.reset_peak_memory_stats()
 for _ in range(config["fixed_forward_repetitions"]):
  torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):model(ids,schedule[:,:ids.shape[1]],depth=8,mode=mode,**kwargs)
  torch.cuda.synchronize();times.append(time.perf_counter()-start)
 return {"tokens":ids.shape[1],"mean_seconds":sum(times)/len(times),"median_seconds":statistics.median(times),"peak_memory_bytes":int(torch.cuda.max_memory_allocated())}

def correctness(model,huginn,tokenizer,dataset,config):
 before=param_sha(huginn);records=[]
 for example_id in config["correctness_prompt_ids"]:
  prompt=tokenize_prompt(tokenizer,build_chat_prompt(tokenizer,dataset[example_id]["question"],config["system_instruction"]))["input_ids"].cuda();schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=example_id,base_seed=config["h0_base_seed"]);h0=schedule[:,:prompt.shape[1]]
  with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
   baseline=huginn(input_ids=prompt,input_states=h0,attention_mask=torch.ones_like(prompt,dtype=torch.bool),num_steps=8,use_cache=False,return_dict=True);plain=model(prompt,h0,mode="plain",return_loop_states=True);modified=[model(prompt,h0,mode="halpern",return_loop_states=True),model(prompt,h0,mode="heavy_ball",momentum=.1,return_loop_states=True),model(prompt,h0,mode="anderson",anderson_window=3,anderson_ridge=1e-4,return_loop_states=True)]
  exact=torch.equal(baseline.logits.float(),plain.logits);finite=all(all(bool(torch.isfinite(state).all()) for state in x.loop_states) and bool(torch.isfinite(x.logits).all()) for x in modified)
  cache_checks=[]
  for mode,param in (("plain",None),("halpern",None),("heavy_ball",.1),("anderson",3)):
   kwargs=method_kwargs(mode,param,config)
   with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
    prefill=model(prompt,h0,mode=mode,use_cache=True,**kwargs);token=prefill.logits[:,-1].argmax(-1,keepdim=True);prefix=torch.cat((prompt,token),dim=1);incremental=model(token,schedule[:,prompt.shape[1]:prompt.shape[1]+1],mode=mode,use_cache=True,past_key_values=prefill.past_key_values,cache_position=torch.tensor([prompt.shape[1]],device="cuda"),**kwargs);full=model(prefix,schedule[:,:prefix.shape[1]],mode=mode,**kwargs)
   delta=float((incremental.logits[:,-1]-full.logits[:,-1]).abs().max());argmax=torch.equal(incremental.logits[:,-1].argmax(-1),full.logits[:,-1].argmax(-1));cache_checks.append({"mode":mode,"cached_full_argmax_exact":argmax,"max_logit_abs_error":delta})
   if not argmax or delta>0.25:raise RuntimeError(f"cache/full mismatch ID={example_id} mode={mode}")
  if not exact or not finite:raise RuntimeError(f"correctness failed ID={example_id}")
  records.append({"example_id":example_id,"plain_logits_exact":exact,"all_modified_loop_states_finite":finite,"cache_checks":cache_checks})
 after=param_sha(huginn)
 if before!=after:raise RuntimeError("frozen Huginn weights changed")
 return {"status":"pass","records":records,"huginn_weights_unchanged":True,"architecture_parameters_added":0,"cache_path_uses_loop_indexed_HuginnDynamicCache":True}

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=CONFIG);a=p.parse_args();from datasets import load_dataset;from transformers import AutoModelForCausalLM,AutoTokenizer
 config_path=ROOT/a.config;config=json.loads(config_path.read_text());root=Path(config["output_root"]);root.mkdir(exist_ok=False)
 tokenizer=AutoTokenizer.from_pretrained(config["model_id"],revision=config["model_revision"],local_files_only=True);huginn=AutoModelForCausalLM.from_pretrained(config["model_id"],revision=config["model_revision"],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();model=ClassicalIterationHuginn(huginn).cuda().eval();train=load_dataset(config["dataset_id"],config["dataset_config"],split="train",revision=config["dataset_revision"]);test=load_dataset(config["dataset_id"],config["dataset_config"],split="test",revision=config["dataset_revision"])
 gate=correctness(model,huginn,tokenizer,train,config);write_exclusive(root/"correctness_gate.json",gate)
 validation_log=root/"validation_sweep.jsonl";validation={}
 candidates=[("heavy_ball",x) for x in config["heavy_ball_momentum_candidates"]]+[("anderson",x) for x in config["anderson_window_candidates"]]
 for mode,param in candidates:
  records=run_examples(model,tokenizer,train,range(config["validation_ids"][0],config["validation_ids"][1]+1),mode,param,config,validation_log);validation[f"{mode}:{param}"]=summarize(records)
 def select(mode):
  options=[(p,validation[f"{mode}:{p}"]) for m,p in candidates if m==mode]
  return max(options,key=lambda item:(item[1]["accuracy"],-item[1]["cap_hit_rate"],-item[1]["mean_generated_tokens"],-float(item[0])))[0]
 selected={"heavy_ball":select("heavy_ball"),"anderson":select("anderson")};write_exclusive(root/"validation_selection.json",{"selection_rule":config["validation_selection"],"results":validation,"selected":selected,"test_used":False})
 test_log=root/"test_records.jsonl";methods=[("plain",None),("halpern",None),("heavy_ball",selected["heavy_ball"]),("anderson",selected["anderson"])];summaries={}
 fixed=[]
 for i in range(config["test_ids"][0],config["test_ids"][1]+1):fixed.extend(tokenize_prompt(tokenizer,build_chat_prompt(tokenizer,test[i]["question"],config["system_instruction"]))["input_ids"][0].tolist());
 fixed_ids=torch.tensor([fixed[:config["fixed_forward_tokens"]]],device="cuda");fixed_schedule=materialize_h0_schedule(huginn,device=fixed_ids.device,example_id=0,base_seed=config["h0_base_seed"])
 for mode,param in methods:
  for warm in range(config["generation_warmup_examples"]):
   prompt=tokenize_prompt(tokenizer,build_chat_prompt(tokenizer,test[warm]["question"],config["system_instruction"]))["input_ids"].cuda();schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=warm,base_seed=config["h0_base_seed"]);generate(model,tokenizer,prompt,schedule,mode,param,{**config,"max_new_tokens":32})
  records=run_examples(model,tokenizer,test,range(config["test_ids"][0],config["test_ids"][1]+1),mode,param,config,test_log,with_diagnostics=True);summary=summarize(records);summary["fixed_256_token_forward"]=fixed_benchmark(model,fixed_ids,fixed_schedule,mode,param,config)
  summary["mean_loop_diagnostics"]=[{"loop":k,"relative_update_norm":sum(r["loop_diagnostics"][k]["relative_update_norm"] for r in records)/len(records),"cosine_to_final_state":sum(r["loop_diagnostics"][k]["cosine_to_final_state"] for r in records)/len(records)} for k in range(8)];summaries[mode]=summary
 result={"protocol":config["protocol"],"status":"complete","selected":selected,"validation":validation,"test":summaries,"git_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip(),"config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),"training_performed":False,"new_parameters":0};write_exclusive(root/"comparison.json",result);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":main()
