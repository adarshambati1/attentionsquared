#!/usr/bin/env python3
"""Validate and live-replay only the Phase 9R eight-example cache-v3 smoke."""
from __future__ import annotations
import json, stat, sys
from pathlib import Path
from types import MethodType
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.build_004_functional_cache_v3_smoke import CONFIG, EXPECTED_HASHES, file_sha256, load_raw_source
from src.data.functional_cache_v3 import CACHE_V3_ITEM_KEYS,SMOKE_EXAMPLE_IDS,schedule_sha256,validate_item,validate_manifest
from src.data.functional_extraction_v3 import capture_scheduled_huginn_states
from src.data.valid_end_manifest import load_and_validate_manifest
from src.evaluation.functional_autoregressive import frozen_coda_logits_from_normalized_state
from src.training.functional_protocol import fixed_huginn_h0_schedule,schedule_prefix
CACHE_ROOT=Path('/workspace/functional_cache_v3_smoke')
def validate_file_set(root=CACHE_ROOT):
 if root.is_symlink() or not root.is_dir():raise ValueError('cache root must be a real directory')
 expected={root/'manifest.json',*(root/f'{i:05d}.npz' for i in SMOKE_EXAMPLE_IDS)};allowed={root/'.functional-cache-v3.lock'}
 actual={p for p in root.rglob('*') if p.is_file()}
 if actual not in (expected,expected|allowed) or any(p.is_dir() for p in root.rglob('*')):raise ValueError('exact file set must be manifest plus eight items; no K directories')
 if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('symlinks forbidden')
 if stat.S_IMODE(root.stat().st_mode)!=0o555:raise ValueError('cache root must be mode 0555')
 for path in actual:
  if stat.S_IMODE(path.stat().st_mode)!=0o444:raise ValueError(f'cache file must be mode 0444: {path}')
 return [root/f'{i:05d}.npz' for i in SMOKE_EXAMPLE_IDS]
def validate_static(root=CACHE_ROOT):
 items=validate_file_set(root);manifest=json.loads((root/'manifest.json').read_text());validate_manifest(manifest)
 for key,value in EXPECTED_HASHES.items():
  if manifest[key]!=value:raise ValueError(f'locked source mismatch: {key}')
 config=json.loads((ROOT/CONFIG).read_text());rows=load_and_validate_manifest(ROOT/config['valid_end_manifest'])[:8]
 for example_id,item in zip(SMOKE_EXAMPLE_IDS,items):
  meta=validate_item(item,manifest);source=ROOT/config['source_continuations']/rows[example_id]['source_relative_path'];source_ids=load_raw_source(source,rows[example_id])
  with np.load(item,allow_pickle=False) as a:
   if set(a.files)!=CACHE_V3_ITEM_KEYS or any('logit' in k.lower() for k in a.files):raise ValueError('forbidden item key')
   if not np.array_equal(a['input_ids'],source_ids):raise ValueError(f'cached tokens differ from raw ID={example_id}')
 return manifest,items
def replay(manifest,items):
 from transformers import AutoModelForCausalLM
 model=AutoModelForCausalLM.from_pretrained(manifest['huginn_model_id'],revision=manifest['huginn_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda().requires_grad_(False);out=[]
 for example_id,item in zip(SMOKE_EXAMPLE_IDS,items):
  with np.load(item,allow_pickle=False) as a: values={k:a[k].copy() for k in a.files}
  ids=torch.from_numpy(values['input_ids']).unsqueeze(0).cuda();mask=torch.ones_like(ids,dtype=torch.bool);calls=0;original=model.initialize_state
  def counted(this,*args,**kwargs):
   nonlocal calls;calls+=1;return original(*args,**kwargs)
  model.initialize_state=MethodType(counted,model)
  cpu_before=torch.random.get_rng_state().clone();cuda_before=[s.clone() for s in torch.cuda.get_rng_state_all()]
  try:
   template=torch.empty((1,2048,values['h0'].shape[1]),device='cuda',dtype=torch.bfloat16)
   schedule,seed=fixed_huginn_h0_schedule(model,template,base_seed=manifest['base_seed'],example_id=example_id)
   if not torch.equal(cpu_before,torch.random.get_rng_state()) or any(not torch.equal(a,b) for a,b in zip(cuda_before,torch.cuda.get_rng_state_all())):raise ValueError('RNG was not restored')
   if seed!=int(values['derived_seed']) or schedule_sha256(schedule)!=str(values['full_schedule_sha256']):raise ValueError('schedule provenance replay mismatch')
   for t in range(1,len(values['input_ids'])+1):
    observed=schedule_prefix(schedule,t).float().cpu().numpy().astype(np.float16)
    if observed.tobytes()!=values['h0'][:t][None].tobytes():raise ValueError(f'prefix mismatch ID={example_id} t={t}')
   prefix=schedule_prefix(schedule,len(values['input_ids']));states=capture_scheduled_huginn_states(model,ids,mask,prefix)
   for key in ('h0','x','h16'):
    if states[key].tobytes()!=values[key].tobytes():raise ValueError(f'live replay mismatch {key}')
   with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
    live=model(input_ids=ids,attention_mask=mask,input_states=prefix,num_steps=16,use_cache=False,return_dict=True,output_details={'return_logits':True,'return_latents':True,'return_head':False,'return_stats':False})
    coda=frozen_coda_logits_from_normalized_state(model,live.latent_states,model.freqs_cis[:,:ids.shape[1]])
   error=float((live.logits.float()-coda.float()).abs().max())
   if not torch.allclose(live.logits.float(),coda.float(),rtol=1e-5,atol=1e-5):raise ValueError('exact normalized pre-coda replay semantics failed')
  finally:model.initialize_state=original
  if calls!=1:raise ValueError(f'expected exactly one initialize_state call, got {calls}')
  out.append({'example_id':example_id,'prefixes_checked':len(values['input_ids']),'initialize_state_calls':1,'coda_max_abs':error})
  del schedule,prefix,states,live,coda,template,ids,mask;torch.cuda.empty_cache()
 return out
def main():
 manifest,items=validate_static();print(json.dumps({'protocol':'functional-cache-v3-smoke-validation-v1','status':'pass','prefix_bit_identity_every_t':True,'rng_restored':True,'full_schedule_persisted':False,'logits_persisted':False,'K_directories':False,'replay':replay(manifest,items)},indent=2,sort_keys=True))
if __name__=='__main__':main()
