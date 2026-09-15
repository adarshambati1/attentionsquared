#!/usr/bin/env python3
"""Held-out evaluation of validation-selected one-shot fixed-point jump."""
from __future__ import annotations
import argparse,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.correctness import build_chat_prompt,find_repetition_onset,score_generation,tokenize_prompt
from src.models.fixed_point_jump_huginn import FixedPointJumpHuginn
from src.training.latent_history import encode_supervised_example,generate_cached,materialize_h0_schedule
from scripts.train_009_fixed_point_jump import atomic_json,losses

def fixed_benchmark(wrapper,ids,schedule,c):
 for _ in range(c['fixed_forward_warmup']):
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):wrapper(ids,schedule[:,:ids.shape[1]])
 samples=[];torch.cuda.reset_peak_memory_stats()
 for _ in range(c['fixed_forward_repetitions']):
  torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=wrapper(ids,schedule[:,:ids.shape[1]])
  torch.cuda.synchronize();samples.append(time.perf_counter()-start)
  if out.core_steps_executed!=0:raise RuntimeError('inference executed recurrence')
 return {'tokens':ids.shape[1],'repetitions':len(samples),'mean_seconds':sum(samples)/len(samples),'median_seconds':statistics.median(samples),'peak_memory_bytes':int(torch.cuda.max_memory_allocated())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/009_fixed_point_jump.json'));a=p.parse_args();from datasets import load_dataset;from transformers import AutoModelForCausalLM,AutoTokenizer
 cp=ROOT/a.config;c=json.loads(cp.read_text());root=Path(c['output_root']);comparison=root/'comparison.json';test_log=root/'test_records.jsonl'
 if comparison.exists():raise FileExistsError('completed held-out evaluation already exists')
 config_sha=hashlib.sha256(cp.read_bytes()).hexdigest();commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();summary=json.loads((root/'training_summary.json').read_text());selection=json.loads((root/'validation_selection.json').read_text());best_path=root/'best.pt';best_sha=hashlib.sha256(best_path.read_bytes()).hexdigest()
 if summary.get('status')!='complete' or summary.get('config_sha256')!=config_sha or summary.get('git_commit')!=commit or summary.get('best_checkpoint_sha256')!=best_sha or selection.get('best_checkpoint_sha256')!=best_sha or selection.get('test_used') is not False:raise RuntimeError('validation-selected checkpoint provenance mismatch')
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);test=load_dataset(c['dataset_id'],c['dataset_config'],split='test',revision=c['dataset_revision']);train=load_dataset(c['dataset_id'],c['dataset_config'],split='train',revision=c['dataset_revision']);huginn=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();wrapper=FixedPointJumpHuginn(huginn,c['bottleneck_size']).cuda().eval();checkpoint=torch.load(best_path,map_location='cpu',weights_only=True)
 if checkpoint.get('protocol')!=c['protocol']+'-checkpoint' or checkpoint.get('config_sha256')!=config_sha or checkpoint.get('git_commit')!=commit or checkpoint.get('step')!=selection.get('best_step'):raise RuntimeError('checkpoint payload provenance mismatch')
 wrapper.predictor.load_state_dict(checkpoint['predictor_state_dict']);records=[];log=test_log
 if test_log.exists():
  for line_number,line in enumerate(test_log.read_text().splitlines(),1):
   try:r=json.loads(line)
   except json.JSONDecodeError as error:raise RuntimeError(f'corrupt preserved test record line {line_number}') from error
   expected_id=c['test_ids'][0]+len(records)
   if r.get('example_id')!=expected_id or r.get('config_sha256')!=config_sha or r.get('checkpoint_sha256')!=best_sha:raise RuntimeError(f'test resume provenance/contiguity failure line {line_number}')
   records.append(r)
 # Every prompt plus the frozen generation cap must fit the 2,048-position h0 schedule.
 prompt_lengths=[tokenize_prompt(tok,build_chat_prompt(tok,test[i]['question'],c['system_instruction']))['input_ids'].shape[1] for i in range(c['test_ids'][0],c['test_ids'][1]+1)]
 if max(prompt_lengths)+c['max_new_tokens']>2048:raise RuntimeError('prompt plus generation cap exceeds h0 schedule')
 if not records:
  for warm in range(c['generation_warmup_examples']):
   example_id=c['validation_ids'][0]+warm;prompt=tokenize_prompt(tok,build_chat_prompt(tok,train[example_id]['question'],c['system_instruction']))['input_ids'].cuda();schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=example_id,base_seed=c['h0_base_seed']);generate_cached(wrapper,tok,prompt,schedule,depth=0,mode='jump',max_new_tokens=32)
 for i in range(c['test_ids'][0]+len(records),c['test_ids'][1]+1):
  prompt=tokenize_prompt(tok,build_chat_prompt(tok,test[i]['question'],c['system_instruction']))['input_ids'].cuda();schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=i,base_seed=c['h0_base_seed']);g=generate_cached(wrapper,tok,prompt,schedule,depth=0,mode='jump',max_new_tokens=c['max_new_tokens']);score=score_generation(g.text,test[i]['answer'],hit_max_new_tokens=g.hit_max_new_tokens);r={'example_id':i,'correct':score.correct,'predicted_answer':score.predicted_answer,'gold_answer':score.gold_answer,'generated_text':g.text,'generated_token_ids':g.token_ids,'generated_tokens':g.generated_tokens,'generation_latency_seconds':g.latency_seconds,'hit_max_new_tokens':g.hit_max_new_tokens,'peak_memory_bytes':g.peak_memory_bytes,'repetition_onset':find_repetition_onset(g.token_ids,chunk_size=8,repetitions=3),'inference_recurrent_steps':0,'config_sha256':config_sha,'checkpoint_sha256':best_sha};records.append(r)
  with log.open('a') as f:f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
  print(json.dumps({k:v for k,v in r.items() if k not in ('generated_text','generated_token_ids')}),flush=True)
 # Teacher-forced held-out objective diagnostics; never used for selection.
 task_sum=fp_sum=tokens=0
 with torch.inference_mode():
  for i in range(c['test_ids'][0],c['test_ids'][1]+1):
   item=encode_supervised_example(tok,test[i]['question'],test[i]['answer'],c['system_instruction']);ids=item.input_ids[None].cuda();schedule=materialize_h0_schedule(huginn,device=ids.device,example_id=i,base_seed=c['h0_base_seed'])
   with torch.autocast('cuda',dtype=torch.bfloat16):out,task,fixed,total,count=losses(wrapper,ids,schedule[:,:ids.shape[1]],item.answer_start,c,force_fixed_point_measurement=True)
   task_sum+=float(task)*count;fp_sum+=float(fixed)*count;tokens+=count
 fixed_tokens=[]
 for i in range(250):fixed_tokens.extend(tokenize_prompt(tok,build_chat_prompt(tok,test[i]['question'],c['system_instruction']))['input_ids'][0].tolist())
 fixed_ids=torch.tensor([fixed_tokens[:c['fixed_forward_tokens']]],device='cuda');fixed_schedule=materialize_h0_schedule(huginn,device=fixed_ids.device,example_id=0,base_seed=c['h0_base_seed']);summary={'correct':sum(r['correct'] for r in records),'total':len(records),'accuracy':sum(r['correct'] for r in records)/len(records),'cap_hit_rate':sum(r['hit_max_new_tokens'] for r in records)/len(records),'degeneration_rate':sum(r['repetition_onset'] is not None for r in records)/len(records),'mean_generated_tokens':sum(r['generated_tokens'] for r in records)/len(records),'mean_generation_latency_seconds':sum(r['generation_latency_seconds'] for r in records)/len(records),'median_generation_latency_seconds':statistics.median(r['generation_latency_seconds'] for r in records),'peak_memory_bytes':max(r['peak_memory_bytes'] for r in records),'fixed_256_token_forward':fixed_benchmark(wrapper,fixed_ids,fixed_schedule,c),'teacher_forced_answer_token_ce':task_sum/tokens,'teacher_forced_fixed_point_relative_residual':fp_sum/tokens,'inference_recurrent_steps':0}
 ref005=ROOT/'results/005_latent_history_attention/comparison.json';ref007=ROOT/'results/007_raw_history_probe/comparison.json';five=json.loads(ref005.read_text());seven=json.loads(ref007.read_text());refs={'plain_huginn_d8':{'accuracy':five['models']['original_huginn_d8']['accuracy'],'artifact':str(ref005.relative_to(ROOT)),'sha256':hashlib.sha256(ref005.read_bytes()).hexdigest()},'current_state_only_d8':{'accuracy':seven['models']['current_state_only_huginn_d8']['accuracy'],'artifact':str(ref007.relative_to(ROOT)),'sha256':hashlib.sha256(ref007.read_bytes()).hexdigest()},'shared_history_d8':{'accuracy':five['models']['history_attention_huginn_d8']['accuracy'],'artifact':str(ref005.relative_to(ROOT)),'sha256':hashlib.sha256(ref005.read_bytes()).hexdigest()},'plain_huginn_d16':{'accuracy':five['models']['original_huginn_d16']['accuracy'],'artifact':str(ref005.relative_to(ROOT)),'sha256':hashlib.sha256(ref005.read_bytes()).hexdigest()}};result={'protocol':c['protocol']+'-evaluation','status':'complete','one_shot_jump':summary,'frozen_references':refs,'checkpoint_step':checkpoint['step'],'checkpoint_sha256':best_sha,'checkpoint_validation':checkpoint['validation'],'git_commit':commit,'config_sha256':config_sha,'test_used_for_selection':False,'h16_or_h64_imitation':False,'full_vocabulary_logits_persisted':False};atomic_json(comparison,result);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
