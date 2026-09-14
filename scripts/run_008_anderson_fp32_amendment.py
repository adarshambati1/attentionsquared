#!/usr/bin/env python3
"""Corrected FP32 Anderson-only amendment; preserves valid Experiment-008 arms."""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, subprocess, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.correctness import build_chat_prompt, tokenize_prompt
from src.models.classical_iteration_huginn import ClassicalIterationHuginn
from src.training.latent_history import materialize_h0_schedule
from scripts.run_008_classical_iteration_d8 import append_jsonl, fixed_benchmark, generate, param_sha, run_examples, summarize, write_exclusive

def gate(model,huginn,tokenizer,dataset,config):
 before=param_sha(huginn);records=[]
 for example_id in config['correctness_prompt_ids']:
  prompt=tokenize_prompt(tokenizer,build_chat_prompt(tokenizer,dataset[example_id]['question'],config['system_instruction']))['input_ids'].cuda();schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=example_id,base_seed=config['h0_base_seed']);h0=schedule[:,:prompt.shape[1]];checks=[]
  for window in config['anderson_window_candidates']:
   kwargs={'mode':'anderson','anderson_window':window,'anderson_ridge':config['anderson_ridge']}
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    full=model(prompt,h0,return_loop_states=True,**kwargs);prefill=model(prompt,h0,use_cache=True,**kwargs);token=prefill.logits[:,-1].argmax(-1,keepdim=True);incremental=model(token,schedule[:,prompt.shape[1]:prompt.shape[1]+1],use_cache=True,past_key_values=prefill.past_key_values,cache_position=torch.tensor([prompt.shape[1]],device='cuda'),**kwargs);prefix=torch.cat((prompt,token),dim=1);uncached=model(prefix,schedule[:,:prefix.shape[1]],**kwargs)
   expected=7*prompt.shape[1];finite=all(bool(torch.isfinite(s).all()) for s in full.loop_states);argmax=torch.equal(incremental.logits[:,-1].argmax(-1),uncached.logits[:,-1].argmax(-1));delta=float((incremental.logits[:,-1]-uncached.logits[:,-1]).abs().max())
   check={'window':window,'expected_solve_attempts':expected,'solve_attempts':full.anderson_solve_attempts,'successful_solves':full.anderson_successful_solves,'fallbacks':full.anderson_fallbacks,'singular_solves':full.anderson_singular_solves,'nonfinite_solves':full.anderson_nonfinite_solves,'all_loop_states_finite':finite,'cached_full_argmax_exact':argmax,'max_logit_abs_error':delta}
   if full.anderson_solve_attempts!=expected or full.anderson_successful_solves!=expected or full.anderson_fallbacks or full.anderson_singular_solves or full.anderson_nonfinite_solves or not finite or not argmax or delta>0.25:raise RuntimeError(f'corrected Anderson gate failed: {check}')
   checks.append(check)
  records.append({'example_id':example_id,'checks':checks})
 after=param_sha(huginn)
 if before!=after:raise RuntimeError('frozen Huginn weights changed')
 return {'status':'pass','linear_solve_dtype':'FP32 outside autocast','implementation_exceptions_are_fatal':True,'huginn_weights_unchanged':True,'records':records}

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/008_anderson_fp32_amendment.json'));a=p.parse_args();from datasets import load_dataset;from transformers import AutoModelForCausalLM,AutoTokenizer
 config_path=ROOT/a.config;config=json.loads(config_path.read_text());outroot=Path(config['output_root']);outroot.mkdir(exist_ok=False)
 tokenizer=AutoTokenizer.from_pretrained(config['model_id'],revision=config['model_revision'],local_files_only=True);huginn=AutoModelForCausalLM.from_pretrained(config['model_id'],revision=config['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();model=ClassicalIterationHuginn(huginn).cuda().eval();train=load_dataset(config['dataset_id'],config['dataset_config'],split='train',revision=config['dataset_revision']);test=load_dataset(config['dataset_id'],config['dataset_config'],split='test',revision=config['dataset_revision'])
 correctness=gate(model,huginn,tokenizer,train,config);write_exclusive(outroot/'correctness_gate.json',correctness)
 validation={};log=outroot/'validation_sweep.jsonl'
 for window in config['anderson_window_candidates']:
  records=run_examples(model,tokenizer,train,range(config['validation_ids'][0],config['validation_ids'][1]+1),'anderson',window,config,log);validation[str(window)]=summarize(records)
 selected=max(config['anderson_window_candidates'],key=lambda w:(validation[str(w)]['accuracy'],-validation[str(w)]['cap_hit_rate'],-validation[str(w)]['mean_generated_tokens'],-w));write_exclusive(outroot/'validation_selection.json',{'selection_rule':config['validation_selection'],'results':validation,'selected_window':selected,'test_used':False})
 # Warm up only on validation prompts, never by pre-running held-out test examples.
 for example_id in config['correctness_prompt_ids']:
  prompt=tokenize_prompt(tokenizer,build_chat_prompt(tokenizer,train[example_id]['question'],config['system_instruction']))['input_ids'].cuda();schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=example_id,base_seed=config['h0_base_seed']);generate(model,tokenizer,prompt,schedule,'anderson',selected,{**config,'max_new_tokens':32})
 test_records=run_examples(model,tokenizer,test,range(config['test_ids'][0],config['test_ids'][1]+1),'anderson',selected,config,outroot/'test_records.jsonl',with_diagnostics=True);test_summary=summarize(test_records)
 fixed=[]
 for i in range(config['test_ids'][0],config['test_ids'][1]+1):fixed.extend(tokenize_prompt(tokenizer,build_chat_prompt(tokenizer,test[i]['question'],config['system_instruction']))['input_ids'][0].tolist())
 fixed_ids=torch.tensor([fixed[:config['fixed_forward_tokens']]],device='cuda');fixed_schedule=materialize_h0_schedule(huginn,device=fixed_ids.device,example_id=0,base_seed=config['h0_base_seed']);test_summary['fixed_256_token_forward']=fixed_benchmark(model,fixed_ids,fixed_schedule,'anderson',selected,config);test_summary['mean_loop_diagnostics']=[{'loop':k,'relative_update_norm':sum(r['loop_diagnostics'][k]['relative_update_norm'] for r in test_records)/len(test_records),'cosine_to_final_state':sum(r['loop_diagnostics'][k]['cosine_to_final_state'] for r in test_records)/len(test_records)} for k in range(8)]
 original_path=ROOT/'results/008_classical_iteration/comparison.json';original=json.loads(original_path.read_text());comparison={'protocol':config['protocol'],'status':'complete','corrected_arm':{'anderson':test_summary},'preserved_valid_arms':{k:original['test'][k] for k in ('plain','halpern','heavy_ball')},'selected_anderson_window':selected,'validation':validation,'original_comparison_sha256':hashlib.sha256(original_path.read_bytes()).hexdigest(),'invalid_anderson_result_preserved':True,'training_performed':False,'new_parameters':0,'git_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip(),'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest()};write_exclusive(outroot/'comparison_amended.json',comparison);print(json.dumps(comparison,indent=2,sort_keys=True))
if __name__=='__main__':main()
