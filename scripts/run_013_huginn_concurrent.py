#!/usr/bin/env python3
"""Run approved Step 3A-C conditions as concurrent batch-size-one workers."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CONFIG=ROOT/'configs/013_step3_huginn_robustness.json'
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.step3_concurrency import production_waves
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('gsm8k','svamp','math500'),required=True);a=p.parse_args();c=json.loads(CONFIG.read_text());root=Path(c['output_root']);scope='gsm8k' if a.dataset=='gsm8k' else 'cross_dataset';waves=[wave for wave in production_waves(c) if wave['scope']==scope];logs=root/'worker_logs'/a.dataset;logs.mkdir(parents=True,exist_ok=True)
 for wave in waves:
   processes=[];opened=[]
   for condition in wave['members']:
    log=logs/f'{condition}.attempt-{uuid.uuid4().hex}.log';handle=log.open('xb');opened.append(handle);command=[sys.executable,str(ROOT/'scripts/run_013_huginn_robustness.py'),'--dataset',a.dataset,'--condition-names',condition,'--no-finalize'];processes.append((condition,subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)))
   codes=[]
   for condition,process in processes:codes.append((condition,process.wait()))
   for handle in opened:handle.flush();os.fsync(handle.fileno());handle.close()
   if any(code for _,code in codes):raise RuntimeError({'dataset':a.dataset,'wave':wave,'worker_exit_codes':codes})
 subprocess.run([sys.executable,str(ROOT/'scripts/run_013_huginn_robustness.py'),'--dataset',a.dataset],cwd=ROOT,check=True)
if __name__=='__main__':main()
