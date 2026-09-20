#!/usr/bin/env python3
"""Crash-resumable, frozen-backbone training for Step-3D Qwen controls."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,math,os,random,shutil,subprocess,sys,time,uuid
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.models.qwen_loop_memory import FixedLoopQwen3,match_corresponding_initialization
CONFIG=ROOT/'configs/013_step3_qwen_recurtrace.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def git():return subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
def atomic_json(path,value,no_overwrite=True):
 data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode();tmp=path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 try:
  if no_overwrite:os.link(tmp,path)
  else:os.replace(tmp,path)
  fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 finally:tmp.unlink(missing_ok=True)
def torch_publish(path,value,no_overwrite=True):
 tmp=path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
 with tmp.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
 try:
  if no_overwrite:os.link(tmp,path)
  else:os.replace(tmp,path)
  fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 finally:tmp.unlink(missing_ok=True)
def append(path,value):
 with path.open('a') as f:f.write(json.dumps(value,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def prompt(tokenizer,item,c):return tokenizer.apply_chat_template([{'role':'system','content':c['system_instruction']},{'role':'user','content':item['Problem']+'\nOptions: '+item['options']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
def encode(tokenizer,item,c):
 p=prompt(tokenizer,item,c);answer=tokenizer(item['correct']+tokenizer.eos_token,add_special_tokens=False).input_ids;ids=p+answer
 if len(ids)>c['maximum_sequence_tokens']:raise ValueError('sequence exceeds maximum')
 return torch.tensor(ids),len(p)
def loss_for(model,input_ids,answer_start):
 out=model(input_ids);selected=out.logits[:,answer_start-1:-1];target=input_ids[:,answer_start:];return torch.nn.functional.cross_entropy(selected.reshape(-1,selected.shape[-1]),target.reshape(-1),reduction='mean'),target.numel()
def initialize(model,variant,c):
 torch.manual_seed(c['module_initialization_seed']);wrapper=FixedLoopQwen3(model,variant,c['memory_width'],c['memory_heads'],*c['loop_layers_zero_indexed_inclusive']).cuda()
 match_corresponding_initialization(wrapper,c['module_initialization_seed']);return wrapper
def validate(wrapper,tok,data,c):
 wrapper.eval();wrapper.qwen.eval();total=0.;tokens=0
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for i in range(c['validation_ids'][0],c['validation_ids'][1]+1):
   ids,start=encode(tok,data[i],c);loss,n=loss_for(wrapper,ids[None].cuda(),start);total+=float(loss)*n;tokens+=n
 return total/tokens,tokens
def validate_resume_curve(lines,state,c):
 records=[]
 for number,line in enumerate(lines[:state['curve_lines']],1):
  try:records.append(json.loads(line))
  except Exception as exc:raise RuntimeError(f'invalid curve JSON line {number}') from exc
 replay_rng=random.Random(c['training_shuffle_seed']);order=list(range(c['train_ids'][0],c['train_ids'][1]+1));replay_rng.shuffle(order);cursor=0;position=0
 for step in range(1,state['step']+1):
  batch=[]
  for _ in range(c['gradient_accumulation_examples']):
   if cursor==len(order):replay_rng.shuffle(order);cursor=0
   batch.append(order[cursor]);cursor+=1
  if position>=len(records) or records[position].get('type')!='train' or records[position].get('step')!=step or records[position].get('example_ids')!=batch or not math.isfinite(float(records[position].get('mean_loss',math.nan))):raise RuntimeError(f'curve train record mismatch at step {step}')
  position+=1
  if step%c['validation_every_steps']==0 or step==c['optimizer_steps']:
   if position>=len(records) or records[position].get('type')!='validation' or records[position].get('step')!=step or not math.isfinite(float(records[position].get('answer_token_ce',math.nan))) or not isinstance(records[position].get('tokens'),int) or records[position]['tokens']<=0:raise RuntimeError(f'curve validation record mismatch at step {step}')
   position+=1
 if position!=state['curve_lines'] or order!=state['order'] or cursor!=state['cursor'] or replay_rng.getstate()!=state['python_rng_state']:raise RuntimeError('curve/resume cursor-RNG mismatch')
def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=('current','shared','recurtrace'),required=True);a=p.parse_args();from datasets import load_dataset;import transformers;from transformers import AutoModelForCausalLM,AutoTokenizer
 c=json.loads(CONFIG.read_text());commit=git();config_sha=sha(CONFIG);dirty=subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
 if dirty:raise RuntimeError(f'worktree dirty: {dirty}')
 if transformers.__version__!=c['transformers_version'] or torch.cuda.get_device_name()!=c['hardware']:raise RuntimeError('environment mismatch')
 root=Path(c['output_root']);gate_path=root/'correctness_gate.json';gate=json.loads(gate_path.read_text());gate_sha=sha(gate_path)
 if gate.get('status')!='pass' or gate.get('git_commit')!=commit or gate.get('config_sha256')!=config_sha or gate.get('transformers_version')!=c['transformers_version'] or gate.get('hardware')!=c['hardware']:raise RuntimeError('correctness gate provenance mismatch')
 out=root/a.variant;out.mkdir(parents=True,exist_ok=True);lock=(out/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(out/'checkpoints').mkdir(exist_ok=True);curve=out/'training_curve.jsonl';summary_path=out/'training_summary.json'
 if summary_path.exists():raise FileExistsError(f'completed training already exists: {summary_path}')
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);data=load_dataset(c['dataset_id'],split='train',revision=c['dataset_revision'],trust_remote_code=True);validation_data=load_dataset(c['dataset_id'],split='validation',revision=c['dataset_revision'],trust_remote_code=True);base=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],dtype=torch.bfloat16,local_files_only=True).eval().cuda();wrapper=initialize(base,a.variant,c);optimizer=torch.optim.AdamW(list(wrapper.trainable_parameters()),lr=c['learning_rate'],weight_decay=c['weight_decay']);rng=random.Random(c['training_shuffle_seed']);order=list(range(c['train_ids'][0],c['train_ids'][1]+1));rng.shuffle(order);cursor=0;best=math.inf;best_step=None;start_step=1;resume_path=out/'resume.pt';kept_lines=0
 if resume_path.exists():
  state=torch.load(resume_path,map_location='cpu',weights_only=False);expected={'protocol':c['protocol']+'-training-resume','variant':a.variant,'git_commit':commit,'config_sha256':config_sha,'gate_sha256':gate_sha}
  if any(state.get(k)!=v for k,v in expected.items()):raise RuntimeError('resume provenance mismatch')
  wrapper.module.load_state_dict(state['module']);optimizer.load_state_dict(state['optimizer']);order=state['order'];cursor=state['cursor'];rng.setstate(state['python_rng_state']);best=state['best'];best_step=state['best_step'];start_step=state['step']+1;kept_lines=state['curve_lines'];torch.set_rng_state(state['torch_rng_state']);torch.cuda.set_rng_state_all(state['cuda_rng_states'])
  lines=curve.read_text().splitlines() if curve.exists() else []
  if len(lines)<kept_lines:raise RuntimeError('training curve shorter than resume checkpoint')
  validate_resume_curve(lines,state,c)
  if len(lines)>kept_lines:
   interrupted=out/f'training_curve.interrupted-{uuid.uuid4().hex}.jsonl'
   with curve.open('rb') as source,interrupted.open('xb') as target:shutil.copyfileobj(source,target);target.flush();os.fsync(target.fileno())
   tmp=curve.with_name(f'.{curve.name}.{uuid.uuid4().hex}.tmp')
   with tmp.open('xb') as f:f.write(('\n'.join(lines[:kept_lines])+('\n' if kept_lines else '')).encode());f.flush();os.fsync(f.fileno())
   os.replace(tmp,curve);fd=os.open(out,os.O_RDONLY);os.fsync(fd);os.close(fd)
 started=time.time()
 for step in range(start_step,c['optimizer_steps']+1):
  wrapper.train();wrapper.qwen.eval();optimizer.zero_grad(set_to_none=True);batch=[]
  for _ in range(c['gradient_accumulation_examples']):
   if cursor==len(order):rng.shuffle(order);cursor=0
   batch.append(order[cursor]);cursor+=1
  losses=[]
  for i in batch:
   ids,start=encode(tok,data[i],c)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss,n=loss_for(wrapper,ids[None].cuda(),start)
   (loss/c['gradient_accumulation_examples']).backward();losses.append(float(loss.detach()))
  torch.nn.utils.clip_grad_norm_(list(wrapper.trainable_parameters()),c['gradient_clip_norm']);optimizer.step();append(curve,{'type':'train','step':step,'example_ids':batch,'mean_loss':sum(losses)/len(losses)});kept_lines+=1
  if step%c['validation_every_steps']==0 or step==c['optimizer_steps']:
   val_ce,val_tokens=validate(wrapper,tok,validation_data,c);record={'type':'validation','step':step,'answer_token_ce':val_ce,'tokens':val_tokens};append(curve,record);kept_lines+=1
   checkpoint={'protocol':c['protocol']+'-checkpoint','variant':a.variant,'step':step,'validation_ce':val_ce,'module':wrapper.module.state_dict(),'git_commit':commit,'config_sha256':config_sha,'model_revision':c['model_revision'],'dataset_revision':c['dataset_revision'],'gate_sha256':gate_sha};path=out/'checkpoints'/f'step_{step:06d}.pt'
   if path.exists():
    existing=torch.load(path,map_location='cpu',weights_only=False)
    if any(existing.get(k)!=checkpoint[k] for k in ('protocol','variant','step','validation_ce','git_commit','config_sha256','model_revision','dataset_revision','gate_sha256')) or set(existing.get('module',{}))!=set(checkpoint['module']) or any(not torch.equal(existing['module'][k],checkpoint['module'][k].cpu()) for k in checkpoint['module']):raise RuntimeError(f'existing validation checkpoint differs: {path}')
   else:torch_publish(path,checkpoint)
   if val_ce<best:best=val_ce;best_step=step
  if step%c['resume_every_steps']==0 or step==c['optimizer_steps']:
   state={'protocol':c['protocol']+'-training-resume','variant':a.variant,'step':step,'module':wrapper.module.state_dict(),'optimizer':optimizer.state_dict(),'order':order,'cursor':cursor,'python_rng_state':rng.getstate(),'best':best,'best_step':best_step,'curve_lines':kept_lines,'torch_rng_state':torch.get_rng_state(),'cuda_rng_states':torch.cuda.get_rng_state_all(),'git_commit':commit,'config_sha256':config_sha,'gate_sha256':gate_sha};torch_publish(resume_path,state,no_overwrite=False)
 if best_step is None:raise RuntimeError('no validation checkpoint selected')
 selected=out/'checkpoints'/f'step_{best_step:06d}.pt';best_path=out/'best.pt'
 if best_path.exists():
  if sha(best_path)!=sha(selected):raise RuntimeError('existing best checkpoint differs')
 else:os.link(selected,best_path);fd=os.open(out,os.O_RDONLY);os.fsync(fd);os.close(fd)
 summary={'protocol':c['protocol']+'-training-summary','status':'complete','variant':a.variant,'best_step':best_step,'best_validation_ce':best,'best_checkpoint_sha256':sha(best_path),'optimizer_steps':c['optimizer_steps'],'examples_exposed':c['optimizer_steps']*c['gradient_accumulation_examples'],'trainable_parameters':sum(p.numel() for p in wrapper.trainable_parameters()),'elapsed_seconds_this_invocation':time.time()-started,'git_commit':commit,'config_sha256':config_sha,'gate_sha256':gate_sha,'model_revision':c['model_revision'],'dataset_revision':c['dataset_revision'],'transformers_version':c['transformers_version'],'hardware':c['hardware'],'backbone_frozen':all(not p.requires_grad for p in base.parameters()),'halting_head':False};atomic_json(summary_path,summary);print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
