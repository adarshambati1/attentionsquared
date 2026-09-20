#!/usr/bin/env python3
"""Validate, stage, and atomically freeze completed Step 3; never starts Step 4."""
from __future__ import annotations
import hashlib,json,os,shutil,subprocess,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HCONFIG=ROOT/'configs/013_step3_huginn_robustness.json';QCONFIG=ROOT/'configs/013_step3_qwen_recurtrace.json'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write_sync(path,data):
 with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def inventory(root,exclude_manifest=False):return [{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file() and not p.name.endswith('.lock') and not (exclude_manifest and p.name=='sha256_manifest.json')]
def verify_frozen(final,commit):
 manifest_path=final/'sha256_manifest.json';manifest=json.loads(manifest_path.read_text())
 if manifest.get('protocol')!='step3-frozen-manifest-v1' or manifest.get('git_commit')!=commit or manifest.get('files')!=inventory(final,exclude_manifest=True):raise RuntimeError(f'frozen manifest verification failed: {final}')
def main():
 dirty=subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
 if dirty:raise RuntimeError(f'worktree dirty: {dirty}')
 commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();hc=json.loads(HCONFIG.read_text());qc=json.loads(QCONFIG.read_text());hroot=Path(hc['output_root']);qroot=Path(qc['output_root']);hpath=hroot/'summary_3abc.json';qpath=qroot/'evaluation.json';h=json.loads(hpath.read_text());q=json.loads(qpath.read_text());hsha=sha(HCONFIG);qsha=sha(QCONFIG);hgate_path=hroot/'correctness_gate.json';qgate_path=qroot/'correctness_gate.json';hgate=json.loads(hgate_path.read_text());qgate=json.loads(qgate_path.read_text())
 if hgate.get('status')!='pass' or hgate.get('protocol')!=hc['protocol']+'-correctness-gate' or hgate.get('git_commit')!=commit or hgate.get('config_sha256')!=hsha or hgate.get('hardware')!=hc['hardware'] or hgate.get('checkpoint_sha256')!=hc['checkpoint_sha256'] or not all(hgate.get('checkpoints_unchanged',{}).values()):raise RuntimeError('Huginn correctness gate provenance mismatch')
 if qgate.get('status')!='pass' or qgate.get('protocol')!=qc['protocol']+'-correctness-gate' or qgate.get('git_commit')!=commit or qgate.get('config_sha256')!=qsha or qgate.get('hardware')!=qc['hardware'] or qgate.get('transformers_version')!=qc['transformers_version']:raise RuntimeError('Qwen correctness gate provenance mismatch')
 if h.get('status')!='complete' or h.get('protocol')!=hc['protocol']+'-summary-3abc' or h.get('git_commit')!=commit or h.get('config_sha256')!=hsha or h.get('checkpoint_sha256')!=hc['checkpoint_sha256'] or h.get('gate_sha256')!=sha(hgate_path) or h.get('hardware')!=hc['hardware'] or h.get('do_not_start_step_4') is not True:raise RuntimeError('Huginn summary provenance mismatch')
 expected_depths={f'D{x}' for x in (4,8,16,32,64)}
 if set(h.get('table3_depth_generalization',{}))!=expected_depths or any(set(v)!={f'plain_d{d[1:]}',f'current_d{d[1:]}',f'shared_d{d[1:]}'} for d,v in h['table3_depth_generalization'].items()):raise RuntimeError('Step 3C depth/model coverage mismatch')
 if q.get('status')!='complete' or q.get('protocol')!=qc['protocol']+'-evaluation' or q.get('git_commit')!=commit or q.get('config_sha256')!=qsha or q.get('hardware')!=qc['hardware'] or q.get('fixed_loop_count')!=2 or q.get('loop_layers')!=[12,13,14] or q.get('halting_head') is not False or q.get('do_not_start_step_4') is not True or q.get('gate_sha256')!=sha(qroot/'correctness_gate.json'):raise RuntimeError('Qwen evaluation provenance mismatch')
 qrows={row['variant']:row for row in q.get('table4',[])}
 if set(qrows)!=set(qc['evaluation_variants']) or q.get('records')!=(qc['test_ids'][1]-qc['test_ids'][0]+1)*len(qc['evaluation_variants']) or qrows['current']['correct']!=qrows['shared']['correct'] or q.get('current_shared_duplicate_control') is not True:raise RuntimeError('Qwen coverage/current-shared equivalence mismatch')
 for variant in qc['variants']:
  summary_path=qroot/variant/'training_summary.json';summary=json.loads(summary_path.read_text())
  if summary.get('status')!='complete' or summary.get('variant')!=variant or summary.get('git_commit')!=commit or summary.get('config_sha256')!=qsha or summary.get('best_checkpoint_sha256')!=sha(qroot/variant/'best.pt') or summary.get('gate_sha256')!=sha(qgate_path):raise RuntimeError(f'{variant} training provenance mismatch')
 final=ROOT/'results/013_step3_robustness'
 if final.exists():
  manifest=final/'sha256_manifest.json'
  if not manifest.exists():raise RuntimeError(f'incomplete existing final directory: {final}')
  verify_frozen(final,commit);print(json.dumps({'status':'already_frozen','path':str(final),'manifest_sha256':sha(manifest)},indent=2));return
 stage=final.parent/f'.{final.name}.staging-{uuid.uuid4().hex}';stage.mkdir()
 try:
  copies={'huginn_summary_3abc.json':hpath,'qwen_table4.json':qpath,'huginn_correctness_gate.json':hroot/'correctness_gate.json','qwen_correctness_gate.json':qroot/'correctness_gate.json','huginn_config.json':HCONFIG,'qwen_config.json':QCONFIG}
  for name,source in copies.items():write_sync(stage/name,Path(source).read_bytes())
  source_manifest={'protocol':'step3-source-artifact-manifest-v1','git_commit':commit,'huginn_root':str(hroot),'qwen_root':str(qroot),'huginn_files':inventory(hroot),'qwen_files':inventory(qroot)};write_sync(stage/'source_artifact_manifest.json',(json.dumps(source_manifest,indent=2,sort_keys=True)+'\n').encode())
  hacc={name:value['pooled_accuracy'] for name,value in h['table1_gsm8k_paired_seeds'].items()};lines=['# Step 3 conclusion','','Step 3 robustified the frozen Huginn findings across paired initializations, datasets, and D4/D8/D16/D32/D64 transfer, and tested the requested fixed-two-loop Qwen3-1.7B controls.','','## Huginn GSM8K (three-seed pooled accuracy)']+[f'- {k}: {v:.3%}' for k,v in sorted(hacc.items())]+['','## Qwen3-1.7B MathQA']+[f"- {v}: {qrows[v]['correct']}/{qrows[v]['total']} = {qrows[v]['accuracy']:.3%}" for v in qc['evaluation_variants']]+['','Current and shared Qwen conditions are an algebraically duplicate control at exactly two loops and are reported as such; no nonzero shared-minus-current mechanism claim is permitted.','',f"Within-backbone RecurTrace gain versus plain: {qrows['recurtrace']['delta_vs_plain']:+.3%}.",f"Within-backbone RecurTrace gain versus current: {qrows['recurtrace']['delta_vs_current']:+.3%}.",'','Absolute Qwen and Huginn accuracies are not compared. Step 4 was not started.'];write_sync(stage/'conclusion.md',('\n'.join(lines)+'\n').encode())
  entries=inventory(stage);manifest={'protocol':'step3-frozen-manifest-v1','git_commit':commit,'files':entries};write_sync(stage/'sha256_manifest.json',(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode());fd=os.open(stage,os.O_RDONLY);os.fsync(fd);os.close(fd);os.rename(stage,final);fd=os.open(final.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 except Exception:
  raise
 print(json.dumps({'status':'frozen','path':str(final),'manifest_sha256':sha(final/'sha256_manifest.json')},indent=2))
if __name__=='__main__':main()
