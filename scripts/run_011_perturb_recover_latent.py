#!/usr/bin/env python3
"""Primary latent contraction and next-token audit for perturb-and-recover."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,math,os,statistics,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.correctness import build_chat_prompt,tokenize_prompt
from src.models.perturbed_huginn import derived_noise_seed,token_relative_noise
from src.training.latent_history import materialize_h0_schedule
from scripts.train_009_fixed_point_jump import atomic_json

def trajectory(huginn,ids,h0,max_depth):
 frequencies=huginn.freqs_cis[:,:ids.shape[1]];index=torch.tensor(-1,device='cpu',dtype=torch.long);x=huginn.transformer.wte(ids)
 if huginn.emb_scale!=1:x=x*huginn.emb_scale
 for block in huginn.transformer.prelude:index+=1;x=block(x,frequencies,index,None,None)
 state=h0;states={0:state};indices={0:int(index)}
 for loop in range(max_depth):state,index=huginn.core_block_forward(state,x,frequencies,None,None,index,loop);states[loop+1]=state;indices[loop+1]=int(index)
 return states,indices,x,frequencies
def coda_logits(huginn,state,frequencies):
 value=huginn.transformer.ln_f(state);index=torch.tensor(0,device='cpu',dtype=torch.long)
 for block in huginn.transformer.coda:index-=1;value=block(value,frequencies,index,None,None)
 return huginn.lm_head(huginn.transformer.ln_f(value)).float()
def write_exclusive_json(path,value):
 data=(json.dumps(value,sort_keys=True)+'\n').encode();fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
 with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def atomic_torch_save_exclusive(path,value):
 temporary=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
 with temporary.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
 try:os.link(temporary,path)
 finally:temporary.unlink(missing_ok=True)
def norm(value):return torch.linalg.vector_norm(value.float())
def cosine(a,b):return float(torch.nn.functional.cosine_similarity(a.float().reshape(1,-1),b.float().reshape(1,-1)).item())
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/011_perturb_recover.json'));a=p.parse_args();from datasets import load_dataset;from transformers import AutoModelForCausalLM,AutoTokenizer
 cp=ROOT/a.config;c=json.loads(cp.read_text());config_sha=hashlib.sha256(cp.read_bytes()).hexdigest();commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();binding={'config_sha256':config_sha,'model_revision':c['model_revision'],'dataset_revision':c['dataset_revision'],'h0_base_seed':c['h0_base_seed'],'depths':c['perturb_depths'],'sigmas':c['noise_strengths'],'perturbation_seeds':c['perturbation_seed_indices']};binding_sha=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest();root=Path(c['output_root']);root.mkdir(exist_ok=True);lock=(root/'latent.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);record_dir=root/'latent_records';record_dir.mkdir(exist_ok=True);clean_dir=root/'clean_trajectories';clean_dir.mkdir(exist_ok=True);records_path=root/'latent_records.jsonl'
 if (root/'latent_summary.json').exists():raise FileExistsError(root/'latent_summary.json')
 gate=json.loads((root/'correctness_gate.json').read_text())
 if gate.get('status')!='pass' or gate.get('config_sha256')!=config_sha or gate.get('git_commit')!=commit or not all(gate.get('sigma_zero_native_exact',{}).values()) or not all(gate.get('manual_trajectory_native_exact',{}).values()):raise RuntimeError('correctness gate provenance/content mismatch')
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);test=load_dataset(c['dataset_id'],c['dataset_config'],split='test',revision=c['dataset_revision']);huginn=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();conditions=[(i,k,sigma,seed) for i in range(c['test_ids'][0],c['test_ids'][1]+1) for k in c['perturb_depths'] for seed in c['perturbation_seed_indices'] for sigma in c['noise_strengths']];records=[]
 for record_path in sorted(record_dir.glob('*.json')):
  expected_name=f'{len(records):06d}.json'
  if record_path.name!=expected_name:raise RuntimeError(f'noncontiguous latent record: {record_path}')
  r=json.loads(record_path.read_text());expected=conditions[len(records)]
  if (r['example_id'],r['perturb_depth'],r['sigma'],r['perturbation_seed_index'])!=expected or r.get('protocol_binding_sha256')!=binding_sha:raise RuntimeError(f'latent resume mismatch {record_path}')
  records.append(r)
 grouped={}
 for condition_index in range(len(records),len(conditions)):
  example_id,k,sigma,seed=conditions[condition_index]
  if example_id not in grouped:
   grouped.clear();prompt=tokenize_prompt(tok,build_chat_prompt(tok,test[example_id]['question'],c['system_instruction']))['input_ids'].cuda();schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=example_id,base_seed=c['h0_base_seed'])
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):grouped[example_id]=(prompt,*trajectory(huginn,prompt,schedule[:,:prompt.shape[1]],c['clean_depth']+1))
   clean_path=clean_dir/f'{example_id:03d}.pt'
   if not clean_path.exists():
    clean_states=grouped[example_id][1];atomic_torch_save_exclusive(clean_path,{'protocol':c['protocol']+'-clean-trajectory','example_id':example_id,'prompt_token_ids':prompt.cpu(),'states':{depth:value.cpu() for depth,value in clean_states.items() if depth<=c['clean_depth']},'native_dtypes':{depth:str(value.dtype) for depth,value in clean_states.items() if depth<=c['clean_depth']},'config_sha256':config_sha})
  prompt,clean,indices,x,frequencies=grouped[example_id];positions=torch.arange(prompt.shape[1],device='cuda').unsqueeze(0)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   perturbed,relative=token_relative_noise(clean[k],sigma=sigma,example_id=example_id,perturb_depth=k,seed_index=seed,base_seed=c['perturbation_base_seed'],token_positions=positions);initial_error=norm(perturbed-clean[k]);state=perturbed;index=torch.tensor(indices[k],device='cpu',dtype=torch.long);curve=[]
   for depth in range(k,c['clean_depth']+1):
    error=norm(state-clean[depth]);entry={'depth':depth,'resumed_loops':depth-k,'error_normalized_to_initial':float(error/initial_error),'cosine_to_clean':cosine(state,clean[depth])};next_state,next_index=huginn.core_block_forward(state,x,frequencies,None,None,index,depth);entry['relative_huginn_residual']=float(norm(next_state-state)/norm(state).clamp_min(1e-12))
    if depth<c['clean_depth']:
     next_error=norm(next_state-clean[depth+1]);entry['per_step_contraction']=float(next_error/error.clamp_min(1e-12))
    curve.append(entry);state,index=next_state,next_index
   pert_d16=curve[16-k];clean_logits=coda_logits(huginn,clean[16],frequencies)[:,-1];# Recompute perturbed D16 because state now advanced past D32.
   state16=perturbed;index16=torch.tensor(indices[k],device='cpu',dtype=torch.long)
   for depth in range(k,16):state16,index16=huginn.core_block_forward(state16,x,frequencies,None,None,index16,depth)
   pert_logits=coda_logits(huginn,state16,frequencies)[:,-1];clean_logp=clean_logits.log_softmax(-1);pert_logp=pert_logits.log_softmax(-1);kl=float((clean_logp.exp()*(clean_logp-pert_logp)).sum())
  direction_seeds=[derived_noise_seed(base_seed=c['perturbation_base_seed'],example_id=example_id,perturb_depth=k,seed_index=seed,token_position=position) for position in range(prompt.shape[1])];direction_seed_sha=hashlib.sha256(json.dumps(direction_seeds,separators=(',',':')).encode()).hexdigest();record={'example_id':example_id,'perturb_depth':k,'sigma':sigma,'perturbation_seed_index':seed,'noise_direction_seed_sha256':direction_seed_sha,'protocol_binding_sha256':binding_sha,'mean_actual_token_relative_perturbation':float(relative.mean()),'curve':curve,'next_token_at_d16':{'argmax_matches_clean':bool(torch.equal(clean_logits.argmax(-1),pert_logits.argmax(-1))),'clean_to_perturbed_kl':kl}}
  write_exclusive_json(record_dir/f'{condition_index:06d}.json',record)
  records.append(record);print(json.dumps({'condition':condition_index+1,'total_conditions':len(conditions),'example_id':example_id,'perturb_depth':k,'sigma':sigma,'seed':seed,'E_D16':pert_d16['error_normalized_to_initial'],'E_D32':curve[-1]['error_normalized_to_initial'],'fallbacks':0}),flush=True)
 def aggregate(subset,depth):
  values=[r['curve'][depth-r['perturb_depth']]['error_normalized_to_initial'] for r in subset];cosines=[r['curve'][depth-r['perturb_depth']]['cosine_to_clean'] for r in subset];return {'E_mean':sum(values)/len(values),'E_median':statistics.median(values),'cosine_mean':sum(cosines)/len(cosines),'cosine_median':statistics.median(cosines)}
 table=[];curves={}
 for k in c['perturb_depths']:
  for sigma in c['noise_strengths']:
   subset=[r for r in records if r['perturb_depth']==k and r['sigma']==sigma];row={'perturb_depth':k,'sigma':sigma,'samples':len(subset),'at_d16':aggregate(subset,16),'at_d32':aggregate(subset,32),'d16_argmax_agreement':sum(r['next_token_at_d16']['argmax_matches_clean'] for r in subset)/len(subset),'d16_clean_to_perturbed_kl_mean':sum(r['next_token_at_d16']['clean_to_perturbed_kl'] for r in subset)/len(subset)};table.append(row);curves[f'h{k}:sigma{sigma}']=[{'depth':depth,'resumed_loops':depth-k,'E_mean':sum(r['curve'][depth-k]['error_normalized_to_initial'] for r in subset)/len(subset),'E_median':statistics.median(r['curve'][depth-k]['error_normalized_to_initial'] for r in subset),'q_mean':sum(r['curve'][depth-k]['per_step_contraction'] for r in subset)/len(subset) if depth<32 else None,'q_median':statistics.median(r['curve'][depth-k]['per_step_contraction'] for r in subset) if depth<32 else None,'cosine_mean':sum(r['curve'][depth-k]['cosine_to_clean'] for r in subset)/len(subset),'cosine_median':statistics.median(r['curve'][depth-k]['cosine_to_clean'] for r in subset),'relative_residual_mean':sum(r['curve'][depth-k]['relative_huginn_residual'] for r in subset)/len(subset),'relative_residual_median':statistics.median(r['curve'][depth-k]['relative_huginn_residual'] for r in subset)} for depth in range(k,33)]
 clean_manifest={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(clean_dir.glob('*.pt'))}
 if len(clean_manifest)!=250:raise RuntimeError('clean trajectory archive incomplete')
 if records_path.exists():raise FileExistsError(records_path)
 write_exclusive_json(records_path,records)
 summary={'protocol':c['protocol']+'-latent','status':'complete','examples':250,'samples_per_table_row':750,'table':table,'curves':curves,'protocol_binding':binding,'protocol_binding_sha256':binding_sha,'git_commit':commit,'clean_trajectory_files':len(clean_manifest),'clean_trajectory_manifest':clean_manifest,'latent_record_files':len(records),'full_vocabulary_logits_persisted':False};write_exclusive_json(root/'latent_summary.json',summary);print(json.dumps(summary['table'],indent=2))
if __name__=='__main__':main()
