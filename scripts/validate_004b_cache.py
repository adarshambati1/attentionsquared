#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from types import MethodType
import numpy as np,torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
from src.evaluation.correctness import SEED_PROTOCOL,build_chat_prompt,ensure_new_output_path,per_example_seed,tokenize_prompt

def main(config_path,cache,manifest):
 ensure_new_output_path(manifest)
 c=json.loads(config_path.read_text())
 expected=[]
 for split,(start,end) in c['splits'].items():
  expected.extend(cache/split/f'{i:05d}.npz' for i in range(start,end+1))
 files=sorted((cache/'train').glob('*.npz'))+sorted((cache/'val').glob('*.npz'))+sorted((cache/'test').glob('*.npz'))
 if set(files) != set(expected) or len(files) != len(expected):
  raise ValueError(f'cache completeness mismatch: expected {len(expected)} exact files, found {len(files)}')
 samples=[]; dtype_set=set(); max_errs=[]
 expected_protocol=SEED_PROTOCOL
 for p in files:
  with np.load(p) as a:
   required={'h0','x','trajectory','input_ids','base_seed','derived_seed','seed_protocol','example_id','split','model_revision','dataset_revision','depth'}
   if set(a.files) != required:
    raise ValueError(f'{p} has keys {sorted(a.files)}, expected {sorted(required)}')
   example_id=int(p.stem)
   expected_split=next((split for split,(start,end) in c['splits'].items() if start <= example_id <= end),None)
   if int(a['example_id'].item()) != example_id or str(a['split'].item()) != expected_split:
    raise ValueError(f'{p} has invalid example identity metadata')
   if str(a['model_revision'].item()) != c['model_revision'] or str(a['dataset_revision'].item()) != c['dataset_revision'] or int(a['depth'].item()) != 16:
    raise ValueError(f'{p} has invalid source metadata')
   if str(a['seed_protocol'].item()) != expected_protocol:
    raise ValueError(f'{p} has seed protocol {a["seed_protocol"].item()!r}, expected {expected_protocol!r}')
   if int(a['base_seed'].item()) != int(c['seed']):
    raise ValueError(f'{p} has base seed {a["base_seed"].item()}, expected {c["seed"]}')
   if int(a['derived_seed'].item()) != per_example_seed(c['seed'],int(p.stem)):
    raise ValueError(f'{p} has an invalid derived seed')
   if a['h0'].shape != a['x'].shape or a['trajectory'].shape[1:] != a['h0'].shape or a['trajectory'].shape[0] != 16 or a['input_ids'].ndim != 1 or a['input_ids'].shape[0] != a['h0'].shape[0]:
    raise ValueError(f'{p} has invalid state/token shapes')
   if a['h0'].dtype != np.float16 or a['x'].dtype != np.float16 or a['trajectory'].dtype != np.float16 or not all(np.isfinite(a[k]).all() for k in ('h0','x','trajectory')):
    raise ValueError(f'{p} has invalid state dtype or non-finite values')
   dtype_set.update([str(a['h0'].dtype),str(a['x'].dtype),str(a['trajectory'].dtype)])
 for p in files[::max(1,len(files)//5)]:
  with np.load(p) as a:
   samples.append({'path':str(p),'T':int(a['h0'].shape[0]),'H':int(a['h0'].shape[1])})
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision']); ds=load_dataset(c['dataset_id'],c['dataset_config'],split=c['dataset_split'],revision=c['dataset_revision']); model=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True).eval().cuda(); convention=[]; coda=[]
 for p in files:
  i=int(p.stem)
  archive=np.load(p)
  derived_seed=int(archive['derived_seed'].item()); ids=torch.from_numpy(archive['input_ids'].copy()).cuda().unsqueeze(0); target=torch.from_numpy(archive['trajectory'].copy()).cuda()
  prompt=build_chat_prompt(tok,ds[i]['question'],c['system_instruction']); expected_ids=tokenize_prompt(tok,prompt)['input_ids'].cuda()
  if not torch.equal(ids,expected_ids):
   raise ValueError(f'{p} input_ids do not match the pinned prompt/tokenizer')
  torch.manual_seed(derived_seed); torch.cuda.manual_seed_all(derived_seed); states=[]; original=model.core_block_forward
  captured={}
  def wrapped(this,x,input_embeds,*args,**kwargs):
   if not states:
    captured['h0']=x.detach().float().cpu()
    captured['x']=input_embeds.detach().float().cpu()
   r=original(x,input_embeds,*args,**kwargs); states.append(r[0].detach().cpu()); return r
  model.core_block_forward=MethodType(wrapped,model)
  try:
   with torch.inference_mode(): live=model(input_ids=ids,num_steps=16,use_cache=False,output_details={'return_logits':False,'return_latents':False,'return_head':False,'return_stats':False})
  finally: model.core_block_forward=original
  errors={'h0':float((captured['h0'][0].cuda().float()-torch.from_numpy(archive['h0']).cuda().float()).abs().max().item()),'x':float((captured['x'][0].cuda().float()-torch.from_numpy(archive['x']).cuda().float()).abs().max().item()),'trajectory':float((torch.stack([s[0] for s in states]).cuda().float()-target.float()).abs().max().item())}
  convention.append({'id':i,**errors})
  with torch.inference_mode(): cached_out=model(input_ids=ids,input_states=target[-1:].to(torch.bfloat16),num_steps=0,use_cache=False); torch.manual_seed(derived_seed); torch.cuda.manual_seed_all(derived_seed); live_out=model(input_ids=ids,num_steps=16,use_cache=False)
  coda_error=float((cached_out.logits-live_out.logits).abs().max().item())
  if max(errors.values()) > 1e-2 or coda_error > 0.1:
   raise ValueError(f'{p} replay exceeds validation tolerance')
  coda.append(coda_error)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path.cwd(),text=True).strip(); out={'examples':len(files),'splits':{'train':len(list((cache/'train').glob('*.npz'))),'val':len(list((cache/'val').glob('*.npz'))),'test':len(list((cache/'test').glob('*.npz')))},'sample_shapes':samples,'cache_dtypes':sorted(dtype_set),'source_model_dtype':c['dtype'],'state_capture':'h0 initialized state; trajectory[d-1] is state after d core_block_forward applications, before final ln_f/coda','h1_h16_convention_checks':convention,'max_coda_self_difference':max(coda),'model_revision':c['model_revision'],'dataset_revision':c['dataset_revision'],'seed_protocol':SEED_PROTOCOL,'tokenizer_chat_template':'native tokenizer apply_chat_template(add_generation_prompt=True), add_special_tokens=False','config_commit':commit}
 with manifest.open('x',encoding='utf-8') as stream: json.dump(out,stream,indent=2); print(json.dumps(out,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004b_fixed_k.json')); p.add_argument('--cache',type=Path,default=Path('results/004b_fixed_k/cache')); p.add_argument('--manifest',type=Path,default=Path('results/004b_fixed_k/cache_manifest_phase1_validation.json')); a=p.parse_args(); main(a.config,a.cache,a.manifest)
