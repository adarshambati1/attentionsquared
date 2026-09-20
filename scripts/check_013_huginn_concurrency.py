#!/usr/bin/env python3
"""Proportional A100 gate for concurrent, strictly batch-size-one Huginn workers."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CONFIG=ROOT/'configs/013_step3_huginn_robustness.json'
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.step3_concurrency import production_waves,validate_correctness_gate_contract,validate_gate_contract
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def publish(path,value):
 data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
 if path.exists():
  if path.read_bytes()!=data:raise RuntimeError(f'existing artifact differs: {path}')
  return
 tmp=path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 try:os.link(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 finally:tmp.unlink(missing_ok=True)
def load(path):return json.loads(path.read_text())
def main():
 import torch
 c=json.loads(CONFIG.read_text());commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();dirty=subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
 if dirty:raise RuntimeError(f'worktree dirty: {dirty}')
 if torch.cuda.get_device_name()!=c['hardware']:raise RuntimeError('hardware mismatch')
 checkpoint_hashes_before={k:sha(v) for k,v in c['checkpoints'].items()}
 if checkpoint_hashes_before!=c['checkpoint_sha256']:raise RuntimeError({'checkpoint_hash_mismatch':checkpoint_hashes_before})
 root=Path(c['output_root']);correctness=root/'correctness_gate.json';correctness_sha=sha(correctness);serial_gate=json.loads(correctness.read_text())
 validate_correctness_gate_contract(serial_gate,config=c,commit=commit,config_sha256=sha(CONFIG),hardware=c['hardware'])
 completed=root/'concurrency_gate.json'
 if completed.exists():
  existing=load(completed);validate_gate_contract(existing,config=c,commit=commit,config_sha256=sha(CONFIG),correctness_gate_sha256=correctness_sha,hardware=c['hardware']);print(json.dumps(existing,indent=2,sort_keys=True));return
 attempt=root/'concurrency_probe_attempts'/f'attempt-{uuid.uuid4().hex}';attempt.mkdir(parents=True);waves=production_waves(c)
 def run_wave(*,tier,wave,variants,max_new_tokens):
  conditions=wave['members'];base_path=attempt/f'{tier}-{wave["wave_id"]}-baseline.json';command=[sys.executable,str(ROOT/'scripts/_013_huginn_serial_probe.py'),'--conditions',','.join(conditions),'--variants',','.join(variants),'--max-new-tokens',str(max_new_tokens),'--output',str(base_path)];start=time.perf_counter();subprocess.run(command,cwd=ROOT,check=True);baseline_seconds=time.perf_counter()-start;outputs=[];processes=[];start=time.perf_counter()
  for condition in conditions:
   output=attempt/f'{tier}-{wave["wave_id"]}-{condition}.json';outputs.append(output);processes.append(subprocess.Popen([sys.executable,str(ROOT/'scripts/_013_huginn_serial_probe.py'),'--conditions',condition,'--variants',','.join(variants),'--max-new-tokens',str(max_new_tokens),'--output',str(output)],cwd=ROOT))
  codes=[process.wait() for process in processes];concurrent_seconds=time.perf_counter()-start
  if any(codes):raise RuntimeError({'wave':wave,'exit_codes':codes,'attempt':str(attempt)})
  base=load(base_path)
  expected_probe={'protocol':c['protocol']+'-serial-concurrency-probe','git_commit':commit,'config_sha256':sha(CONFIG),'hardware':c['hardware'],'variants':variants,'max_new_tokens':max_new_tokens}
  if any(base.get(key)!=value for key,value in expected_probe.items()) or base.get('conditions')!=conditions:raise RuntimeError(f'baseline probe provenance mismatch: {base_path}')
  baseline={(r['condition'],r['variant'],r['example_id'],r['seed_index']):r for r in base['records']};concurrent={};artifact_hashes={'baseline':sha(base_path)}
  for output,condition in zip(outputs,conditions):
   artifact=load(output);artifact_hashes[output.stem]=sha(output)
   if any(artifact.get(key)!=value for key,value in expected_probe.items()) or artifact.get('conditions')!=[condition]:raise RuntimeError(f'probe provenance mismatch: {output}')
   for record in artifact['records']:concurrent[(record['condition'],record['variant'],record['example_id'],record['seed_index'])]=record
  fields=('h0_seed','h0_schedule_sha256','token_ids','hit_max_new_tokens','ended_naturally','fallback','fallback_onset');no_duplicates=len(base['records'])==len(baseline)==len(concurrent)==sum(len(load(output)['records']) for output in outputs);exact=no_duplicates and set(baseline)==set(concurrent) and all(all(baseline[key][field]==concurrent[key][field] for field in fields) for key in baseline);natural=any(record['variant']=='natural' and record['ended_naturally'] for record in baseline.values());forced=all(record['variant']!='forced_cap' or (len(record['token_ids'])==max_new_tokens and record['hit_max_new_tokens']) for record in baseline.values());record_keys=[{'condition':key[0],'variant':key[1],'example_id':key[2],'seed_index':key[3]} for key in baseline];return {**wave,'tier':tier,'variants':variants,'max_new_tokens':max_new_tokens,'records':len(baseline),'record_keys':record_keys,'exact':exact,'natural_stopping_observed':natural,'forced_cap_completed':forced,'baseline_seconds_including_load':baseline_seconds,'concurrent_seconds_including_load':concurrent_seconds,'observed_wall_speedup':baseline_seconds/concurrent_seconds,'probe_artifact_sha256':artifact_hashes}
 short=[run_wave(tier='all-waves-short',wave=wave,variants=['forced_cap'],max_new_tokens=c['concurrency_gate_short_tokens']) for wave in waves]
 by_depth=[]
 for depth in sorted({wave['depth'] for wave in waves}):
  wave=max((item for item in waves if item['depth']==depth),key=lambda item:item['workers']);by_depth.append(run_wave(tier='natural-stop-by-depth',wave=wave,variants=['natural'],max_new_tokens=c['max_new_tokens']))
 stress=[]
 for depth in (32,64):
  wave=max((item for item in waves if item['depth']==depth),key=lambda item:item['workers']);stress.append(run_wave(tier='worst-case-forced-cap',wave=wave,variants=['forced_cap'],max_new_tokens=c['max_new_tokens']))
 checkpoint_hashes_after={k:sha(v) for k,v in c['checkpoints'].items()}
 if checkpoint_hashes_after!=checkpoint_hashes_before:raise RuntimeError({'checkpoint_mutation':{'before':checkpoint_hashes_before,'after':checkpoint_hashes_after}})
 result={'protocol':c['protocol']+'-concurrency-gate','status':'pass','gate_policy':c['concurrency_gate_policy'],'production_wave_manifest':waves,'tiers':{'all_waves_short':short,'natural_stop_by_depth':by_depth,'worst_case_forced_cap':stress},'all_exact':all(item['exact'] for item in short+by_depth+stress),'attempt_directory':str(attempt),'correctness_gate_sha256':correctness_sha,'checkpoint_sha256_before':checkpoint_hashes_before,'checkpoint_sha256_after':checkpoint_hashes_after,'git_commit':commit,'config_sha256':sha(CONFIG),'hardware':torch.cuda.get_device_name(),'full_vocabulary_logits_persisted':False}
 validate_gate_contract(result,config=c,commit=commit,config_sha256=sha(CONFIG),correctness_gate_sha256=correctness_sha,hardware=c['hardware']);publish(root/'concurrency_gate.json',result);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
