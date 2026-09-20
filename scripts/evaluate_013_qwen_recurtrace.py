#!/usr/bin/env python3
"""Evaluate fixed-two-loop Qwen3-1.7B mechanism controls on MathQA."""
from __future__ import annotations
import fcntl,hashlib,json,math,os,random,re,statistics,subprocess,sys,time,uuid
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.models.qwen_loop_memory import FixedLoopQwen3,match_corresponding_initialization
from src.evaluation.step3_robustness import wilson_interval
CONFIG=ROOT/'configs/013_step3_qwen_recurtrace.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def publish(path,value):
 data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
 if path.exists():
  if path.read_bytes()!=data:raise RuntimeError(f'existing differs {path}')
  return
 tmp=path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 try:os.link(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 finally:tmp.unlink(missing_ok=True)
def prompt(tok,item,c):return tok.apply_chat_template([{'role':'system','content':c['system_instruction']},{'role':'user','content':item['Problem']+'\nOptions: '+item['options']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
def extract(text):
 m=re.search(r'(?i)(?:^|[^a-z])([abcde])(?:[^a-z]|$)',text.strip());return m.group(1).lower() if m else None
def initialize(base,variant,c):
 torch.manual_seed(c['module_initialization_seed']);w=FixedLoopQwen3(base,variant,c['memory_width'],c['memory_heads'],*c['loop_layers_zero_indexed_inclusive']).cuda().eval()
 if variant!='plain':match_corresponding_initialization(w,c['module_initialization_seed'])
 return w
def validate_record(r,variant,i,tok,item,c,commit,checkpoint_sha,training_summary_sha):
 required=('generated_token_ids','generated_text','predicted_answer','gold_answer','correct','generated_tokens','hit_max_new_tokens','generation_latency_seconds','peak_memory_bytes')
 fixed={'protocol':c['protocol']+'-record','variant':variant,'example_id':i,'git_commit':commit,'config_sha256':sha(CONFIG),'checkpoint_sha256':checkpoint_sha,'training_summary_sha256':training_summary_sha,'hardware':c['hardware']}
 if any(k not in r for k in required) or any(r.get(k)!=v for k,v in fixed.items()):raise RuntimeError(f'record provenance/schema mismatch {variant}/{i}')
 prediction=extract(r['generated_text']);hit=len(r['generated_token_ids'])==c['max_new_tokens'] and (not r['generated_token_ids'] or r['generated_token_ids'][-1]!=tok.eos_token_id)
 if r['generated_tokens']!=len(r['generated_token_ids']) or r['generated_text']!=tok.decode(r['generated_token_ids'],skip_special_tokens=True) or r['predicted_answer']!=prediction or r['gold_answer']!=item['correct'] or r['correct']!=(prediction==item['correct']) or r['hit_max_new_tokens']!=hit or not math.isfinite(float(r['generation_latency_seconds'])) or r['generation_latency_seconds']<0 or not isinstance(r['peak_memory_bytes'],int) or r['peak_memory_bytes']<0:raise RuntimeError(f'record reconstruction mismatch {variant}/{i}')
def generate(model,tok,ids,max_new):
 generated=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for _ in range(max_new):
   out=model(ids);token=int(out.logits[0,-1].argmax());generated.append(token);ids=torch.cat((ids,torch.tensor([[token]],device=ids.device)),1)
   if token==tok.eos_token_id:break
 torch.cuda.synchronize();return generated,time.perf_counter()-start,int(torch.cuda.max_memory_allocated())
def bootstrap(records,c):
 variants=list(records);n=len(next(iter(records.values())));rng=random.Random(c['bootstrap_seed']);model_draws={v:[] for v in variants};pairs=[(v,'plain') for v in variants if v!='plain']+[(v,'current') for v in ('shared','recurtrace')];delta={f'{a}_minus_{b}':[] for a,b in pairs}
 for _ in range(c['bootstrap_samples']):
  chosen=[rng.randrange(n) for _ in range(n)];vals={v:sum(records[v][i]['correct'] for i in chosen)/n for v in variants}
  for v in variants:model_draws[v].append(vals[v])
  for a,b in pairs:delta[f'{a}_minus_{b}'].append(vals[a]-vals[b])
 def interval(xs):xs.sort();return [xs[int(.025*(len(xs)-1))],xs[int(.975*(len(xs)-1))]]
 return {v:interval(xs) for v,xs in model_draws.items()},{k:interval(xs) for k,xs in delta.items()}
def main():
 from datasets import load_dataset;import transformers;from transformers import AutoModelForCausalLM,AutoTokenizer
 c=json.loads(CONFIG.read_text());commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();config_sha=sha(CONFIG);dirty=subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
 if dirty:raise RuntimeError(f'worktree dirty: {dirty}')
 if transformers.__version__!=c['transformers_version'] or torch.cuda.get_device_name()!=c['hardware']:raise RuntimeError('environment mismatch')
 root=Path(c['output_root']);gate_path=root/'correctness_gate.json';gate_sha=sha(gate_path);lock=(root/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gate=json.loads(gate_path.read_text())
 if gate.get('status')!='pass' or gate.get('git_commit')!=commit or gate.get('config_sha256')!=config_sha or gate.get('hardware')!=c['hardware'] or gate.get('transformers_version')!=c['transformers_version']:raise RuntimeError('gate mismatch')
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);data=load_dataset(c['dataset_id'],split='test',revision=c['dataset_revision'],trust_remote_code=True);base=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],dtype=torch.bfloat16,local_files_only=True).eval().cuda();records_by_variant={};summaries={};loaded_states={}
 for variant in c['evaluation_variants']:
  wrapper=initialize(base,variant,c);checkpoint_sha=None;training_summary_sha=None;checkpoint=None
  if variant!='plain':
   summary_path=root/variant/'training_summary.json';summary=json.loads(summary_path.read_text());training_summary_sha=sha(summary_path);best_path=root/variant/'best.pt';checkpoint_sha=sha(best_path);checkpoint=torch.load(best_path,map_location='cpu',weights_only=False);expected={'protocol':c['protocol']+'-training-summary','status':'complete','variant':variant,'git_commit':commit,'config_sha256':config_sha,'best_checkpoint_sha256':checkpoint_sha,'model_revision':c['model_revision'],'dataset_revision':c['dataset_revision'],'transformers_version':c['transformers_version'],'hardware':c['hardware'],'backbone_frozen':True,'halting_head':False,'gate_sha256':gate_sha}
   if any(summary.get(k)!=v for k,v in expected.items()) or checkpoint.get('protocol')!=c['protocol']+'-checkpoint' or checkpoint.get('variant')!=variant or checkpoint.get('step')!=summary['best_step'] or checkpoint.get('git_commit')!=commit or checkpoint.get('config_sha256')!=config_sha or checkpoint.get('gate_sha256')!=gate_sha:raise RuntimeError(f'{variant} checkpoint/training provenance mismatch')
   wrapper.module.load_state_dict(checkpoint['module']);loaded_states[variant]=checkpoint['module']
  directory=root/'records'/variant;directory.mkdir(parents=True,exist_ok=True);records=[]
  for path in sorted(directory.glob('*.json')):
   if path.name!=f'{len(records):04d}.json':raise RuntimeError(f'noncontiguous {path}')
   r=json.loads(path.read_text());validate_record(r,variant,len(records),tok,data[len(records)],c,commit,checkpoint_sha,training_summary_sha);records.append(r)
  for i in range(len(records),c['test_ids'][1]+1):
   ids=torch.tensor([prompt(tok,data[i],c)],device='cuda');generated,latency,peak=generate(wrapper,tok,ids,c['max_new_tokens']);text=tok.decode(generated,skip_special_tokens=True);prediction=extract(text);r={'protocol':c['protocol']+'-record','variant':variant,'example_id':i,'generated_token_ids':generated,'generated_text':text,'predicted_answer':prediction,'gold_answer':data[i]['correct'],'correct':prediction==data[i]['correct'],'generated_tokens':len(generated),'hit_max_new_tokens':len(generated)==c['max_new_tokens'] and (not generated or generated[-1]!=tok.eos_token_id),'generation_latency_seconds':latency,'peak_memory_bytes':peak,'git_commit':commit,'config_sha256':config_sha,'checkpoint_sha256':checkpoint_sha,'training_summary_sha256':training_summary_sha,'hardware':c['hardware']};validate_record(r,variant,i,tok,data[i],c,commit,checkpoint_sha,training_summary_sha);publish(directory/f'{i:04d}.json',r);records.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ('generated_text','generated_token_ids')}),flush=True)
  correct=sum(r['correct'] for r in records);summaries[variant]={'correct':correct,'total':len(records),'accuracy':correct/len(records),'wilson_95ci':wilson_interval(correct,len(records)),'mean_generation_latency_seconds':sum(r['generation_latency_seconds'] for r in records)/len(records),'median_generation_latency_seconds':statistics.median(r['generation_latency_seconds'] for r in records),'mean_generated_tokens':sum(r['generated_tokens'] for r in records)/len(records),'cap_hit_rate':sum(r['hit_max_new_tokens'] for r in records)/len(records),'peak_memory_bytes':max(r['peak_memory_bytes'] for r in records),'checkpoint_step':None if variant=='plain' else checkpoint['step'],'checkpoint_sha256':checkpoint_sha,'training_summary_sha256':training_summary_sha};records_by_variant[variant]=records
 if any(not torch.equal(loaded_states['current'][k],loaded_states['shared'][k]) for k in ('v_proj.weight','out_proj.weight','injection')) or any(a['generated_token_ids']!=b['generated_token_ids'] for a,b in zip(records_by_variant['current'],records_by_variant['shared'])):raise RuntimeError('trained current/shared duplicate control diverged')
 model_ci,delta_ci=bootstrap(records_by_variant,c);plain=summaries['plain']['accuracy'];current=summaries['current']['accuracy'];table=[];paired={}
 for v in c['evaluation_variants']:table.append({'variant':v,**summaries[v],'example_bootstrap_95ci':model_ci[v],'delta_vs_plain':None if v=='plain' else summaries[v]['accuracy']-plain,'delta_vs_current':None if v in ('plain','current') else summaries[v]['accuracy']-current})
 for a,b in [(v,'plain') for v in c['evaluation_variants'] if v!='plain']+[(v,'current') for v in ('shared','recurtrace')]:
  wins=sum(x['correct'] and not y['correct'] for x,y in zip(records_by_variant[a],records_by_variant[b]));losses=sum(not x['correct'] and y['correct'] for x,y in zip(records_by_variant[a],records_by_variant[b]));paired[f'{a}_minus_{b}']={'wins':wins,'losses':losses,'ties':len(records_by_variant[a])-wins-losses,'bootstrap_95ci':delta_ci[f'{a}_minus_{b}']}
 result={'protocol':c['protocol']+'-evaluation','status':'complete','table4':table,'paired_comparisons':paired,'records':sum(map(len,records_by_variant.values())),'fixed_loop_count':2,'loop_layers':[12,13,14],'halting_head':False,'current_shared_duplicate_control':True,'inference_estimand':'Held-out MathQA test-example accuracy conditional on the single preregistered training initialization; intervals resample test examples and do not generalize over training seeds.','git_commit':commit,'config_sha256':config_sha,'gate_sha256':gate_sha,'hardware':c['hardware'],'do_not_start_step_4':True};publish(root/'evaluation.json',result);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
