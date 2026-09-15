#!/usr/bin/env python3
"""Train only the functional one-shot fixed-point jump predictor."""
from __future__ import annotations
import argparse,hashlib,json,math,os,random,subprocess,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.models.fixed_point_jump_huginn import FixedPointJumpHuginn
from src.training.latent_history import answer_cross_entropy,encode_supervised_example,generate_cached,materialize_h0_schedule
CONFIG=Path('configs/009_fixed_point_jump.json')

def atomic_save(path,value):
 tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
 with tmp.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def git_commit():return subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
def tensor_state_sha(module):
 d=hashlib.sha256()
 for n,v in module.state_dict().items():d.update(n.encode());d.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return d.hexdigest()
def atomic_json(path,value):
 data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode();tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def encode(ds,tok,i,c):
 item=encode_supervised_example(tok,ds[i]['question'],ds[i]['answer'],c['system_instruction'])
 if len(item.input_ids)>c['maximum_sequence_tokens']:raise ValueError(f'sequence exceeds 2048 tokens ID={i}')
 return item
def fp_loss(output,answer_start,token_count,epsilon):
 if output.fixed_point_image is None or output.core_steps_executed!=1:raise RuntimeError('training must execute exactly one consistency core step')
 selected=slice(answer_start-1,answer_start-1+token_count);h=output.latent_states[:,selected].float();f=output.fixed_point_image[:,selected].float()
 return (f-h).square().sum()/(h.square().sum()+epsilon)
def losses(wrapper,ids,h0,answer_start,c,*,force_fixed_point_measurement=False):
 use_fixed=bool(c['fixed_point_weight']) or force_fixed_point_measurement;out=wrapper(ids,h0,compute_fixed_point=use_fixed);task,count=answer_cross_entropy(out.logits,ids,answer_start)
 fixed=fp_loss(out,answer_start,count,c['fixed_point_epsilon']) if use_fixed else task.new_zeros(())
 return out,task,fixed,task+c['fixed_point_weight']*fixed,count
def validate(wrapper,huginn,tok,ds,ids,c):
 wrapper.eval();task_sum=fp_sum=0.;tokens=0
 with torch.inference_mode():
  for i in ids:
   item=encode(ds,tok,i,c);x=item.input_ids[None].cuda();schedule=materialize_h0_schedule(huginn,device=x.device,example_id=i,base_seed=c['h0_base_seed'])
   with torch.autocast('cuda',dtype=torch.bfloat16):out,task,fixed,total,count=losses(wrapper,x,schedule[:,:x.shape[1]],item.answer_start,c)
   task_sum+=float(task)*count;fp_sum+=float(fixed)*count;tokens+=count
 return {'answer_token_cross_entropy':task_sum/tokens,'fixed_point_relative_residual':fp_sum/tokens if c['fixed_point_weight'] else None,'fixed_point_term_computed':bool(c['fixed_point_weight']),'total_objective':task_sum/tokens+c['fixed_point_weight']*fp_sum/tokens,'answer_tokens':tokens}
def payload(wrapper,opt,step,val,config_path,initialization_sha):
 config=json.loads(config_path.read_text());return {'protocol':config['protocol']+'-checkpoint','step':step,'validation':val,'predictor_state_dict':{k:v.detach().cpu() for k,v in wrapper.predictor.state_dict().items()},'optimizer_state_dict':opt.state_dict(),'git_commit':git_commit(),'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'predictor_initialization_sha256':initialization_sha}
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=CONFIG);a=p.parse_args();from datasets import load_dataset;from transformers import AutoModelForCausalLM,AutoTokenizer
 cp=ROOT/a.config;c=json.loads(cp.read_text());out=Path(c['output_root'])
 if out.exists():
  allowed={'correctness_gate.json','training.log','training.pid'};seen={p.name for p in out.iterdir()}
  if out.is_symlink() or 'correctness_gate.json' not in seen or not seen<=allowed:raise FileExistsError(f'unsafe existing output: {out}')
 else:out.mkdir()
 gate=json.loads((out/'correctness_gate.json').read_text());config_sha=hashlib.sha256(cp.read_bytes()).hexdigest();commit=git_commit()
 if gate.get('status')!='pass' or gate.get('protocol')!='functional-fixed-point-jump-correctness-gate-v1' or gate.get('config_sha256')!=config_sha or gate.get('git_commit')!=commit:raise RuntimeError('correctness gate is not bound to this committed config/code')
 (out/'checkpoints').mkdir(exist_ok=False);curve=out/'training_curve.jsonl'
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);ds=load_dataset(c['dataset_id'],c['dataset_config'],split='train',revision=c['dataset_revision']);huginn=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();torch.manual_seed(c['predictor_initialization_seed']);wrapper=FixedPointJumpHuginn(huginn,c['bottleneck_size']).cuda().train();initialization_sha=tensor_state_sha(wrapper.predictor)
 if initialization_sha!=gate.get('predictor_initialization_sha256'):raise RuntimeError('predictor initialization does not match correctness gate')
 opt=torch.optim.AdamW(wrapper.trainable_parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay']);train_ids=list(range(c['train_ids'][0],c['train_ids'][1]+1));val_ids=list(range(c['validation_ids'][0],c['validation_ids'][1]+1));rng=random.Random(c['training_shuffle_seed']);order=[];cursor=0;best=math.inf;best_step=None;started=time.time();opt.zero_grad(set_to_none=True)
 for step in range(1,c['optimizer_steps']+1):
  wrapper.train();batch=[]
  for _ in range(c['gradient_accumulation_examples']):
   if cursor>=len(order):order=train_ids.copy();rng.shuffle(order);cursor=0
   i=order[cursor];cursor+=1;batch.append((i,encode(ds,tok,i,c)))
  total_tokens=sum(len(item.input_ids)-item.answer_start for _,item in batch);task_sum=fp_sum=0.
  for i,item in batch:
   ids=item.input_ids[None].cuda();schedule=materialize_h0_schedule(huginn,device=ids.device,example_id=i,base_seed=c['h0_base_seed'])
   with torch.autocast('cuda',dtype=torch.bfloat16):output,task,fixed,total,count=losses(wrapper,ids,schedule[:,:ids.shape[1]],item.answer_start,c)
   (total*(count/total_tokens)).backward();task_sum+=float(task.detach())*count;fp_sum+=float(fixed.detach())*count
  grad=float(torch.nn.utils.clip_grad_norm_(list(wrapper.trainable_parameters()),c['gradient_clip_norm']))
  if not all(math.isfinite(v) for v in (grad,task_sum,fp_sum)):raise RuntimeError(f'nonfinite training at step {step}')
  opt.step();opt.zero_grad(set_to_none=True);record={'type':'train','step':step,'answer_token_cross_entropy':task_sum/total_tokens,'fixed_point_relative_residual':fp_sum/total_tokens if c['fixed_point_weight'] else None,'fixed_point_term_computed':bool(c['fixed_point_weight']),'total_objective':task_sum/total_tokens+c['fixed_point_weight']*fp_sum/total_tokens,'answer_tokens':total_tokens,'gradient_norm':grad,'elapsed_seconds':time.time()-started}
  with curve.open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
  print(json.dumps(record),flush=True)
  if step==c['startup_gate_step']:
   val=validate(wrapper,huginn,tok,ds,val_ids[:c['startup_validation_examples']],c);item=encode(ds,tok,val_ids[0],c);prompt=item.input_ids[:item.answer_start][None].cuda();schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=val_ids[0],base_seed=c['h0_base_seed']);gen=generate_cached(wrapper,tok,prompt,schedule,depth=0,mode='jump',max_new_tokens=32);atomic_save(out/'checkpoints'/'startup_step_00010.pt',payload(wrapper,opt,step,val,cp,initialization_sha));startup={'type':'startup_gate','step':step,'validation':val,'generation_tokens':gen.generated_tokens,'finite':all(math.isfinite(v) for v in val.values() if isinstance(v,(int,float))),'inference_recurrent_steps':0};print(json.dumps(startup),flush=True)
   if not startup['finite'] or gen.generated_tokens<1:raise RuntimeError('startup gate failed')
  if step%c['validation_every_steps']==0:
   val=validate(wrapper,huginn,tok,ds,val_ids,c);rec={'type':'validation','step':step,**val,'elapsed_seconds':time.time()-started}
   with curve.open('a') as f:f.write(json.dumps(rec)+'\n');f.flush();os.fsync(f.fileno())
   print(json.dumps(rec),flush=True);ck=payload(wrapper,opt,step,val,cp,initialization_sha);atomic_save(out/'checkpoints'/f'step_{step:05d}.pt',ck)
   if val['answer_token_cross_entropy']<best:best=val['answer_token_cross_entropy'];best_step=step;atomic_save(out/'best.pt',ck)
 best_path=out/'best.pt';best_sha=hashlib.sha256(best_path.read_bytes()).hexdigest();summary={'protocol':c['protocol']+'-training','status':'complete','optimizer_steps':c['optimizer_steps'],'best_step':best_step,'best_validation_answer_token_cross_entropy':best,'selection_criterion':c['selection_criterion'],'training_seconds':time.time()-started,'git_commit':git_commit(),'config_sha256':config_sha,'predictor_initialization_sha256':initialization_sha,'best_checkpoint_sha256':best_sha,'test_set_used_for_selection':False,'huginn_parameters_trained':False,'inference_recurrent_steps':0,'predictor_parameters':sum(p.numel() for p in wrapper.predictor.parameters()),'fixed_point_weight':c['fixed_point_weight']};atomic_json(out/'training_summary.json',summary);atomic_json(out/'validation_selection.json',{'criterion':c['selection_criterion'],'best_step':best_step,'best_validation_answer_token_cross_entropy':best,'best_checkpoint_sha256':best_sha,'config_sha256':config_sha,'git_commit':commit,'test_used':False});print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
