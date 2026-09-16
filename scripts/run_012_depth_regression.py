#!/usr/bin/env python3
"""Step 2/11: frozen plain-Huginn depth regression through D512."""
from __future__ import annotations
import argparse,csv,fcntl,hashlib,io,json,os,statistics,subprocess,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.correctness import build_chat_prompt,find_repetition_onset,generation_status,score_generation,stop_token_ids,tokenize_prompt
from src.models.plain_depth_huginn import PlainDepthHuginn
from src.training.latent_history import generate_cached,materialize_h0_schedule

def publish_bytes(path,data):
 if path.exists():
  if path.read_bytes()!=data:raise RuntimeError(f'existing artifact differs: {path}')
  return
 tmp=path.with_name(f'.{path.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp')
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 try:os.link(tmp,path)
 finally:tmp.unlink(missing_ok=True)
def write_exclusive(path,value,*,pretty=False):publish_bytes(path,(json.dumps(value,indent=2 if pretty else None,sort_keys=True)+'\n').encode())
def promote_plot(tmp,path):
 if path.exists():
  if path.stat().st_size<100 or path.read_bytes()[:8]!=b'\x89PNG\r\n\x1a\n':raise RuntimeError(f'invalid existing plot: {path}')
  tmp.unlink(missing_ok=True);return
 try:os.link(tmp,path)
 finally:tmp.unlink(missing_ok=True)
def measure_prompt_residual(model,prompt,schedule,depth):
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(prompt,schedule[:,:prompt.shape[1]],depth=depth,mode='plain',compute_fixed_point_residual=True)
 if out.fixed_point_residual is None or not torch.isfinite(torch.tensor(out.fixed_point_residual)):raise FloatingPointError('non-finite prompt fixed-point residual')
 return out.fixed_point_residual
def validate_record(r,*,depth,example_id,config_sha,commit,c,tok,test):
 required=('generated_text','generated_token_ids','extracted_answer','gold_answer','correct','generated_tokens','hit_max_new_tokens','degeneration','generation_latency_seconds','fixed_point_residual','peak_memory_bytes')
 if any(key not in r for key in required) or r.get('protocol')!=c['protocol']+'-record' or r.get('depth')!=depth or r.get('example_id')!=example_id or r.get('config_sha256')!=config_sha or r.get('git_commit')!=commit or r.get('model_revision')!=c['model_revision'] or r.get('dataset_revision')!=c['dataset_revision'] or r.get('h0_base_seed')!=c['h0_base_seed'] or r.get('h0_seed_index')!=c['h0_seed_index']:raise RuntimeError(f'record provenance/schema mismatch D={depth} ID={example_id}')
 if r['generated_tokens']!=len(r['generated_token_ids']) or r['generated_text']!=tok.decode(r['generated_token_ids'],skip_special_tokens=False):raise RuntimeError(f'record token/text mismatch D={depth} ID={example_id}')
 if not all(torch.isfinite(torch.tensor(float(r[key]))).item() for key in ('generation_latency_seconds','fixed_point_residual')) or r['generation_latency_seconds']<0 or r['fixed_point_residual']<0:raise RuntimeError(f'non-finite record D={depth} ID={example_id}')
 status=generation_status(r['generated_token_ids'],max_new_tokens=c['max_new_tokens'],stop_token_ids=stop_token_ids(tok));score=score_generation(r['generated_text'],test[example_id]['answer'],hit_max_new_tokens=status.hit_max_new_tokens);repetition=find_repetition_onset(r['generated_token_ids'],chunk_size=c['degeneration_chunk_size'],repetitions=c['degeneration_repetitions'])
 if r['hit_max_new_tokens']!=status.hit_max_new_tokens or r['extracted_answer']!=score.predicted_answer or r['gold_answer']!=score.gold_answer or r['correct']!=score.correct or r['repetition_onset']!=repetition or r['degeneration']!=(repetition is not None):raise RuntimeError(f'record reconstruction mismatch D={depth} ID={example_id}')
def summarize(records):return {'correct':sum(r['correct'] for r in records),'total':len(records),'accuracy':sum(r['correct'] for r in records)/len(records),'fixed_point_residual_mean':sum(r['fixed_point_residual'] for r in records)/len(records),'fixed_point_residual_median':statistics.median(r['fixed_point_residual'] for r in records),'cap_hit_rate':sum(r['hit_max_new_tokens'] for r in records)/len(records),'degeneration_rate':sum(r['repetition_onset'] is not None for r in records)/len(records),'mean_generated_tokens':sum(r['generated_tokens'] for r in records)/len(records),'mean_generation_latency_seconds':sum(r['generation_latency_seconds'] for r in records)/len(records),'median_generation_latency_seconds':statistics.median(r['generation_latency_seconds'] for r in records),'peak_memory_bytes':max(r['peak_memory_bytes'] for r in records)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/012_depth_regression.json'));a=p.parse_args();from datasets import load_dataset;from transformers import AutoModelForCausalLM,AutoTokenizer
 cp=ROOT/a.config;c=json.loads(cp.read_text());config_sha=hashlib.sha256(cp.read_bytes()).hexdigest();commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();root=Path(c['output_root']);root.mkdir(exist_ok=True);lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gate=json.loads((root/'correctness_gate.json').read_text())
 if gate.get('status')!='pass' or gate.get('config_sha256')!=config_sha or gate.get('git_commit')!=commit:raise RuntimeError('correctness gate provenance mismatch')
 if (root/'summary.json').exists():raise FileExistsError(root/'summary.json')
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);test=load_dataset(c['dataset_id'],c['dataset_config'],split='test',revision=c['dataset_revision']);huginn=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();model=PlainDepthHuginn(huginn).cuda().eval();reference_path=ROOT/'results/005_latent_history_attention/comparison.json';reference=json.loads(reference_path.read_text());reference_d16={r['example_id']:r for r in reference['records'] if r['model']=='original_huginn_d16'}
 if len(reference_d16)!=250:raise RuntimeError('frozen D16 reference incomplete')
 all_records={}
 for depth in c['depths']:
  directory=root/f'depth_{depth:03d}';directory.mkdir(exist_ok=True);records=[]
  for path in sorted(directory.glob('*.json')):
   expected_id=c['test_ids'][0]+len(records)
   if path.name!=f'{expected_id:03d}.json':raise RuntimeError(f'noncontiguous depth record {path}')
   r=json.loads(path.read_text())
   validate_record(r,depth=depth,example_id=expected_id,config_sha=config_sha,commit=commit,c=c,tok=tok,test=test)
   records.append(r)
  for example_id in range(c['test_ids'][0]+len(records),c['test_ids'][1]+1):
   prompt=tokenize_prompt(tok,build_chat_prompt(tok,test[example_id]['question'],c['system_instruction']))['input_ids'].cuda()
   if prompt.shape[1]+c['max_new_tokens']>2048:raise RuntimeError(f'h0 schedule overflow ID={example_id}')
   schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=example_id,base_seed=c['h0_base_seed']);residual=measure_prompt_residual(model,prompt,schedule,depth);g=generate_cached(model,tok,prompt,schedule,depth=depth,mode='plain',max_new_tokens=c['max_new_tokens']);score=score_generation(g.text,test[example_id]['answer'],hit_max_new_tokens=g.hit_max_new_tokens);r={'protocol':c['protocol']+'-record','depth':depth,'example_id':example_id,'config_sha256':config_sha,'git_commit':commit,'model_revision':c['model_revision'],'dataset_revision':c['dataset_revision'],'h0_base_seed':c['h0_base_seed'],'h0_seed_index':c['h0_seed_index'],'generated_text':g.text,'generated_token_ids':g.token_ids,'extracted_answer':score.predicted_answer,'gold_answer':score.gold_answer,'correct':score.correct,'generated_tokens':g.generated_tokens,'hit_max_new_tokens':g.hit_max_new_tokens,'repetition_onset':find_repetition_onset(g.token_ids,chunk_size=c['degeneration_chunk_size'],repetitions=c['degeneration_repetitions']),'degeneration':False,'generation_latency_seconds':g.latency_seconds,'peak_memory_bytes':g.peak_memory_bytes,'fixed_point_residual':residual};r['degeneration']=r['repetition_onset'] is not None;validate_record(r,depth=depth,example_id=example_id,config_sha=config_sha,commit=commit,c=c,tok=tok,test=test);write_exclusive(directory/f'{example_id:03d}.json',r);records.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ('generated_text','generated_token_ids')}),flush=True)
  all_records[depth]=records
  if depth==16:
   mismatches=[]
   for r in records:
    ref=reference_d16[r['example_id']]
    if r['generated_token_ids']!=ref['generated_token_ids'] or r['extracted_answer']!=ref['predicted_answer'] or r['gold_answer']!=ref['gold_answer'] or r['correct']!=ref['correct'] or r['hit_max_new_tokens']!=ref['hit_max_new_tokens']:mismatches.append(r['example_id'])
   if mismatches:raise RuntimeError(f'D16 frozen-reference replay mismatch IDs={mismatches[:10]} total={len(mismatches)}')
   replay={'status':'pass','examples':len(records),'exact_generated_tokens':True,'exact_stopping_cap_scoring':True,'reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest()}
   replay_path=root/'d16_replay_gate.json'
   if not replay_path.exists():write_exclusive(replay_path,replay,pretty=True)
  depth_summary=summarize(records);depth_summary.update({'depth':depth,'status':'complete','config_sha256':config_sha,'git_commit':commit});summary_path=root/f'depth_{depth:03d}_summary.json'
  if not summary_path.exists():write_exclusive(summary_path,depth_summary,pretty=True)
 transitions=[]
 for left,right in zip(c['depths'],c['depths'][1:]):
  counts={'wrong_to_correct':0,'correct_to_wrong':0,'correct_to_correct':0,'wrong_to_wrong':0}
  for a_record,b_record in zip(all_records[left],all_records[right]):
   key=('correct' if a_record['correct'] else 'wrong')+'_to_'+('correct' if b_record['correct'] else 'wrong');counts[key]+=1
  transitions.append({'from_depth':left,'to_depth':right,**counts})
 table=[{'depth':depth,**summarize(all_records[depth])} for depth in c['depths']];raw=[r for depth in c['depths'] for r in all_records[depth]]
 write_exclusive(root/'raw_results.json',raw)
 fields=['depth','example_id','extracted_answer','gold_answer','correct','generated_tokens','hit_max_new_tokens','degeneration','repetition_onset','generation_latency_seconds','peak_memory_bytes','fixed_point_residual','generated_text'];csv_buffer=io.StringIO(newline='');w=csv.DictWriter(csv_buffer,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in raw);publish_bytes(root/'raw_results.csv',csv_buffer.getvalue().encode())
 write_exclusive(root/'transitions.json',transitions,pretty=True)
 summary={'protocol':c['protocol'],'step':c['step'],'status':'complete','table':table,'transitions':transitions,'config_sha256':config_sha,'git_commit':commit,'raw_records':len(raw),'do_not_start_step_3':True,'full_vocabulary_logits_persisted':False}
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;import numpy as np
 depths=c['depths'];accuracies=[row['accuracy'] for row in table];residuals=[row['fixed_point_residual_mean'] for row in table]
 plt.figure();plt.plot(depths,accuracies,marker='o');plt.xscale('log',base=2);plt.xticks(depths,depths);plt.xlabel('Recurrent depth');plt.ylabel('GSM8K accuracy');plt.grid(alpha=.3);plt.tight_layout();tmp=root/f'.accuracy.{os.getpid()}.tmp.png';plt.savefig(tmp,dpi=180);plt.close();promote_plot(tmp,root/'accuracy_vs_depth.png')
 plt.figure();plt.plot(depths,residuals,marker='o');plt.xscale('log',base=2);plt.yscale('log');plt.xticks(depths,depths);plt.xlabel('Recurrent depth');plt.ylabel('Fixed-point residual');plt.grid(alpha=.3);plt.tight_layout();tmp=root/f'.residual.{os.getpid()}.tmp.png';plt.savefig(tmp,dpi=180);plt.close();promote_plot(tmp,root/'residual_vs_depth.png')
 labels=[f"{r['from_depth']}→{r['to_depth']}" for r in transitions];x=np.arange(len(labels));plt.figure(figsize=(9,4));plt.bar(x-.2,[r['wrong_to_correct'] for r in transitions],.4,label='wrong→correct');plt.bar(x+.2,[r['correct_to_wrong'] for r in transitions],.4,label='correct→wrong');plt.xticks(x,labels);plt.ylabel('Examples');plt.legend();plt.tight_layout();tmp=root/f'.transitions.{os.getpid()}.tmp.png';plt.savefig(tmp,dpi=180);plt.close();promote_plot(tmp,root/'correctness_transitions.png')
 matrix=np.array([[int(r['correct']) for r in all_records[d]] for d in depths]);plt.figure(figsize=(14,4));plt.imshow(matrix,aspect='auto',interpolation='nearest',cmap='binary',vmin=0,vmax=1);plt.yticks(range(len(depths)),depths);plt.xlabel('Held-out example ID');plt.ylabel('Depth');plt.colorbar(label='Correct');plt.tight_layout();tmp=root/f'.heatmap.{os.getpid()}.tmp.png';plt.savefig(tmp,dpi=180);plt.close();promote_plot(tmp,root/'correctness_heatmap.png');write_exclusive(root/'summary.json',summary,pretty=True);print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
