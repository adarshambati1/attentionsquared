#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from types import MethodType
import numpy as np,torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer

def main(config_path,cache,manifest):
 c=json.loads(config_path.read_text()); files=sorted((cache/'train').glob('*.npz'))+sorted((cache/'val').glob('*.npz'))+sorted((cache/'test').glob('*.npz')); assert len(files)==2500
 samples=[]; dtype_set=set(); max_errs=[]
 for p in files[::max(1,len(files)//5)]:
  with np.load(p) as a:
   assert a['h0'].shape==a['x'].shape and a['trajectory'].shape[1:]==a['h0'].shape and a['trajectory'].shape[0]==16; dtype_set.update([str(a['h0'].dtype),str(a['x'].dtype),str(a['trajectory'].dtype)]); samples.append({'path':str(p),'T':int(a['h0'].shape[0]),'H':int(a['h0'].shape[1])})
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision']); ds=load_dataset(c['dataset_id'],c['dataset_config'],split=c['dataset_split'],revision=c['dataset_revision']); model=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True).eval().cuda(); convention=[]; coda=[]
 for p in files[::max(1,len(files)//5)]:
  i=int(p.stem); with_np=np.load(p); ids=torch.from_numpy(with_np['input_ids']).cuda().unsqueeze(0); target=torch.from_numpy(with_np['trajectory']).cuda(); torch.manual_seed(c['seed']+i); torch.cuda.manual_seed_all(c['seed']+i); states=[]; original=model.core_block_forward
  def wrapped(this,x,input_embeds,*args,**kwargs):
   r=original(x,input_embeds,*args,**kwargs); states.append(r[0].detach().cpu()); return r
  model.core_block_forward=MethodType(wrapped,model)
  try:
   with torch.inference_mode(): live=model(input_ids=ids,num_steps=16,use_cache=False,output_details={'return_logits':False,'return_latents':False,'return_head':False,'return_stats':False})
  finally: model.core_block_forward=original
  convention.append({'id':i,'h1_max_abs_error':float((states[0][0].cuda().float()-target[0].float()).abs().max().item()),'h16_max_abs_error':float((states[-1][0].cuda().float()-target[-1].float()).abs().max().item())})
  with torch.inference_mode(): cached_out=model(input_ids=ids,input_states=target[-1:].to(torch.bfloat16),num_steps=0,use_cache=False); torch.manual_seed(c['seed']+i); torch.cuda.manual_seed_all(c['seed']+i); live_out=model(input_ids=ids,num_steps=16,use_cache=False)
  coda.append(float((cached_out.logits-live_out.logits).abs().max().item()))
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path.cwd(),text=True).strip(); out={'examples':len(files),'splits':{'train':len(list((cache/'train').glob('*.npz'))),'val':len(list((cache/'val').glob('*.npz'))),'test':len(list((cache/'test').glob('*.npz')))},'sample_shapes':samples,'cache_dtypes':sorted(dtype_set),'source_model_dtype':c['dtype'],'state_capture':'h0 initialized state; trajectory[d-1] is state after d core_block_forward applications, before final ln_f/coda','h1_h16_convention_checks':convention,'max_coda_self_difference':max(coda),'model_revision':c['model_revision'],'dataset_revision':c['dataset_revision'],'tokenizer_chat_template':'native tokenizer apply_chat_template(add_generation_prompt=True), add_special_tokens=False','config_commit':commit}
 manifest.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004b_fixed_k.json')); p.add_argument('--cache',type=Path,default=Path('results/004b_fixed_k/cache')); p.add_argument('--manifest',type=Path,default=Path('results/004b_fixed_k/cache_manifest.json')); a=p.parse_args(); main(a.config,a.cache,a.manifest)
