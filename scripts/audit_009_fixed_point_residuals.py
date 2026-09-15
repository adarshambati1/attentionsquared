#!/usr/bin/env python3
"""Compare true Huginn residuals by depth with the failed one-shot checkpoint."""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.models.fixed_point_jump_huginn import FixedPointJumpHuginn
from src.training.latent_history import answer_cross_entropy,encode_supervised_example,materialize_h0_schedule
from scripts.train_009_fixed_point_jump import fp_loss,atomic_json

def recurrent_states_and_images(huginn,ids,h0,depths):
 frequencies=huginn.freqs_cis[:,:ids.shape[1]];index=torch.tensor(-1,device='cpu',dtype=torch.long);x=huginn.transformer.wte(ids)
 if huginn.emb_scale!=1:x=x*huginn.emb_scale
 for block in huginn.transformer.prelude:index+=1;x=block(x,frequencies,index,None,None)
 state=h0;states={0:state};images={}
 for loop in range(max(depths)+1):
  next_state,index=huginn.core_block_forward(state,x,frequencies,None,None,index,loop)
  if loop in depths:images[loop]=next_state
  state=next_state
  if loop+1 in depths:states[loop+1]=state
 return states,images
def relative_residual(h,image,answer_start,count,epsilon):
 s=slice(answer_start-1,answer_start-1+count);a=h[:,s].float();b=image[:,s].float();return float((b-a).square().sum()/(a.square().sum()+epsilon))
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/009_fixed_point_residual_audit.json'));a=p.parse_args();from datasets import load_dataset;from transformers import AutoModelForCausalLM,AutoTokenizer
 ap=ROOT/a.config;ac=json.loads(ap.read_text());source_path=ROOT/ac['source_config'];c=json.loads(source_path.read_text());audit_config_sha=hashlib.sha256(ap.read_bytes()).hexdigest();source_config_sha=hashlib.sha256(source_path.read_bytes()).hexdigest();frozen_root=ROOT/'results/009_fixed_point_jump';manifest=json.loads((frozen_root/'manifest.json').read_text());frozen_summary_path=frozen_root/'training_summary.json'
 if hashlib.sha256(frozen_summary_path.read_bytes()).hexdigest()!=manifest['files']['training_summary.json']:raise RuntimeError('frozen Experiment009 summary fails manifest')
 frozen_summary=json.loads(frozen_summary_path.read_text());binding={'audit_config_sha256':audit_config_sha,'source_config_sha256':source_config_sha,'source_manifest_sha256':hashlib.sha256((frozen_root/'manifest.json').read_bytes()).hexdigest(),'model_revision':c['model_revision'],'dataset_revision':c['dataset_revision'],'depths':ac['depths'],'residual_estimand':ac['residual_estimand']};binding_sha=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest();output=Path(ac['output_root']);output.mkdir(exist_ok=True);records_path=output/'records.jsonl'
 if (output/'summary.json').exists():raise FileExistsError(output/'summary.json')
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);test=load_dataset(c['dataset_id'],c['dataset_config'],split='test',revision=c['dataset_revision']);huginn=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();wrapper=FixedPointJumpHuginn(huginn,c['bottleneck_size']).cuda().eval();checkpoint_path=Path(ac['source_checkpoint']);checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True);checkpoint_sha=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
 if checkpoint_sha!=manifest['best_checkpoint_sha256'] or checkpoint_sha!=frozen_summary['best_checkpoint_sha256'] or checkpoint['step']!=frozen_summary['best_step'] or checkpoint['config_sha256']!=source_config_sha or checkpoint['git_commit']!=manifest['source_commit']:raise RuntimeError('source checkpoint provenance mismatch')
 wrapper.predictor.load_state_dict(checkpoint['predictor_state_dict'])
 # Pinned-H100 gate: manual raw-state capture must reproduce native normalized latents.
 gate_id=ac['test_ids'][0];gate_item=encode_supervised_example(tok,test[gate_id]['question'],test[gate_id]['answer'],c['system_instruction']);gate_ids=gate_item.input_ids[None].cuda();gate_schedule=materialize_h0_schedule(huginn,device=gate_ids.device,example_id=gate_id,base_seed=c['h0_base_seed']);gate_h0=gate_schedule[:,:gate_ids.shape[1]]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  gate_states,_=recurrent_states_and_images(huginn,gate_ids,gate_h0,ac['depths']);native_checks={}
  for depth in ac['depths']:
   if depth==0:native_checks['0']={'initial_state_exact':torch.equal(gate_states[0],gate_h0)};continue
   native=huginn(input_ids=gate_ids,input_states=gate_h0,attention_mask=torch.ones_like(gate_ids,dtype=torch.bool),num_steps=depth,use_cache=False,return_dict=True,output_details={'return_logits':False,'return_latents':True,'return_head':False,'return_stats':False});manual=huginn.transformer.ln_f(gate_states[depth]);native_checks[str(depth)]={'normalized_latent_bitwise_exact':torch.equal(manual,native.latent_states)}
 if not all(next(iter(value.values())) for value in native_checks.values()):raise RuntimeError(f'native state capture gate failed: {native_checks}')
 records=[]
 if records_path.exists():
  for line_number,line in enumerate(records_path.read_text().splitlines(),1):
   record=json.loads(line)
   if record.get('example_id')!=ac['test_ids'][0]+len(records) or record.get('source_checkpoint_sha256')!=checkpoint_sha or record.get('protocol_binding_sha256')!=binding_sha:raise RuntimeError(f'audit resume provenance failure line {line_number}')
   records.append(record)
 for example_id in range(ac['test_ids'][0]+len(records),ac['test_ids'][1]+1):
  item=encode_supervised_example(tok,test[example_id]['question'],test[example_id]['answer'],c['system_instruction']);ids=item.input_ids[None].cuda();schedule=materialize_h0_schedule(huginn,device=ids.device,example_id=example_id,base_seed=c['h0_base_seed']);h0=schedule[:,:ids.shape[1]];count=ids.shape[1]-item.answer_start
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   states,images=recurrent_states_and_images(huginn,ids,h0,ac['depths']);depth_residuals={str(depth):relative_residual(states[depth],images[depth],item.answer_start,count,ac['fixed_point_epsilon']) for depth in ac['depths']}
   jump=wrapper(ids,h0,compute_fixed_point=True);task,_=answer_cross_entropy(jump.logits,ids,item.answer_start);jump_residual=float(fp_loss(jump,item.answer_start,count,ac['fixed_point_epsilon']));selected_logits=jump.logits[:,item.answer_start-1:ids.shape[1]-1];targets=ids[:,item.answer_start:];token_correct=int((selected_logits.argmax(-1)==targets).sum());first_logits=jump.logits[:,item.answer_start-1];first_target=ids[:,item.answer_start];first_ce=float(F.cross_entropy(first_logits,first_target));first_correct=bool(first_logits.argmax(-1).eq(first_target).item())
  record={'example_id':example_id,'source_checkpoint_sha256':checkpoint_sha,'protocol_binding_sha256':binding_sha,'answer_tokens':count,'huginn_relative_fixed_point_residuals':depth_residuals,'jump_relative_fixed_point_residual':jump_residual,'jump_answer_token_cross_entropy':float(task),'jump_first_answer_token_cross_entropy':first_ce,'jump_answer_tokens_correct':token_correct,'jump_first_answer_token_correct':first_correct};records.append(record)
  with records_path.open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
  print(json.dumps(record),flush=True)
 total_tokens=sum(r['answer_tokens'] for r in records);summary_depths={str(d):sum(r['huginn_relative_fixed_point_residuals'][str(d)]*r['answer_tokens'] for r in records)/total_tokens for d in ac['depths']};summary={'protocol':ac['protocol'],'status':'complete','examples':len(records),'answer_tokens':total_tokens,'huginn_relative_fixed_point_residuals':summary_depths,'one_shot_checkpoint':{'relative_fixed_point_residual':sum(r['jump_relative_fixed_point_residual']*r['answer_tokens'] for r in records)/total_tokens,'answer_token_cross_entropy':sum(r['jump_answer_token_cross_entropy']*r['answer_tokens'] for r in records)/total_tokens,'first_answer_token_cross_entropy':sum(r['jump_first_answer_token_cross_entropy'] for r in records)/len(records),'answer_token_accuracy':sum(r['jump_answer_tokens_correct'] for r in records)/total_tokens,'first_answer_token_accuracy':sum(r['jump_first_answer_token_correct'] for r in records)/len(records)},'source_checkpoint_sha256':checkpoint_sha,'native_state_capture_gate':native_checks,'source_checkpoint_step':checkpoint['step'],'config_sha256':audit_config_sha,'protocol_binding':binding,'protocol_binding_sha256':binding_sha,'git_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip(),'full_vocabulary_logits_persisted':False};atomic_json(output/'summary.json',summary);print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
