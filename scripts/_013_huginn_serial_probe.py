#!/usr/bin/env python3
"""Internal batch-size-one worker used only by the Step-3 concurrency gate."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,uuid
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.correctness import build_chat_prompt,tokenize_prompt
from src.models.latent_history_huginn import LatentHistoryHuginn
from src.models.per_layer_history_huginn import PerLayerHistoryHuginn
from src.evaluation.correctness import per_example_seed
from src.training.latent_history import generate_cached,materialize_h0_schedule
CONFIG=ROOT/'configs/013_step3_huginn_robustness.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class NoStopTokenizer:
 def __init__(self,tokenizer):self.tokenizer=tokenizer;self.eos_token_id=None
 def convert_tokens_to_ids(self,token):return None
 def decode(self,*args,**kwargs):return self.tokenizer.decode(*args,**kwargs)
def publish(path,value):
 data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode();tmp=path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 try:os.link(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 finally:tmp.unlink(missing_ok=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--conditions',required=True);p.add_argument('--variants',required=True);p.add_argument('--max-new-tokens',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();from datasets import load_dataset;from transformers import AutoModelForCausalLM,AutoTokenizer
 c=json.loads(CONFIG.read_text());commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);huginn=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();checkpoints={k:torch.load(v,map_location='cpu',weights_only=False) for k,v in c['checkpoints'].items()};wrappers={'plain':LatentHistoryHuginn(huginn,c['projection_size'],c['attention_heads']).cuda().eval(),'current':LatentHistoryHuginn(huginn,c['projection_size'],c['attention_heads']).cuda().eval(),'projected_uniform':LatentHistoryHuginn(huginn,c['projection_size'],c['attention_heads']).cuda().eval(),'shared':LatentHistoryHuginn(huginn,c['projection_size'],c['attention_heads']).cuda().eval(),'per_layer':PerLayerHistoryHuginn(huginn,c['projection_size'],c['attention_heads']).cuda().eval()}
 for name in ('current','projected_uniform','shared','per_layer'):wrappers[name].history_attention.load_state_dict(checkpoints[name]['history_attention_state_dict'])
 modes={'plain':'disabled','current':'current_only','projected_uniform':'uniform','shared':'learned','per_layer':'learned'};data=load_dataset(c['gsm8k']['dataset_id'],c['gsm8k']['config'],split='test',revision=c['gsm8k']['revision']);records=[]
 for condition in a.conditions.split(','):
  model,depth_text=condition.rsplit('_d',1);depth=int(depth_text)
  for example_id in c['concurrency_gate_examples']:
   prompt=tokenize_prompt(tok,build_chat_prompt(tok,data[example_id]['question'],c['system_instruction']))['input_ids'].cuda();schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=example_id,base_seed=c['h0_base_seed'],seed_index=0);schedule_sha256=hashlib.sha256(schedule.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest();h0_seed=per_example_seed(c['h0_base_seed'],example_id,step=0,seed_index=0)
   for variant in a.variants.split(','):
    if variant not in ('natural','forced_cap'):raise ValueError(variant)
    result=generate_cached(wrappers[model],tok if variant=='natural' else NoStopTokenizer(tok),prompt,schedule,depth=depth,mode=modes[model],max_new_tokens=a.max_new_tokens,max_cache_allocated_bytes=c['generation_cache_allocated_limit_bytes']);records.append({'condition':condition,'variant':variant,'example_id':example_id,'seed_index':0,'h0_seed':h0_seed,'h0_schedule_sha256':schedule_sha256,'token_ids':result.token_ids,'hit_max_new_tokens':result.hit_max_new_tokens,'ended_naturally':result.ended_naturally,'fallback':result.used_full_prefix_fallback,'fallback_onset':result.full_prefix_fallback_onset,'peak_memory_bytes':result.peak_memory_bytes})
 publish(a.output,{'protocol':c['protocol']+'-serial-concurrency-probe','conditions':a.conditions.split(','),'variants':a.variants.split(','),'max_new_tokens':a.max_new_tokens,'records':records,'git_commit':commit,'config_sha256':sha(CONFIG),'hardware':torch.cuda.get_device_name()})
if __name__=='__main__':main()
