"""Frozen depth schedule and launch authorization for Step-3D Stage 1."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import torch

def sample_loop_depths(*,steps:int,seed:int,target_mean:float=4.0,log_sigma:float=.5,minimum:int=1,maximum:int=8)->list[int]:
 if steps<1 or target_mean<=1 or not 1<=minimum<=maximum:raise ValueError('invalid sampler configuration')
 generator=torch.Generator(device='cpu');generator.manual_seed(seed);mu=math.log(target_mean-1)-.5*log_sigma**2;z=torch.normal(mu,log_sigma,size=(steps,),generator=generator);extra=torch.poisson(z.exp(),generator=generator);return (extra.add(1).clamp(minimum,maximum).to(torch.int64).tolist())

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def validate_training_authorization(config:dict)->dict:
 """Fail closed until an immutable exact or explicitly paper-guided asset manifest exists."""
 path=Path(config['asset_authorization_manifest']);expected_sha=config.get('asset_authorization_manifest_sha256')
 if not isinstance(expected_sha,str) or len(expected_sha)!=64 or any(ch not in '0123456789abcdef' for ch in expected_sha):raise RuntimeError('Step 3D training blocked: authorization manifest digest is not frozen in config')
 if not path.is_file() or sha256(path)!=expected_sha:raise RuntimeError(f'Step 3D training blocked: missing or hash-mismatched authorization manifest {path}')
 manifest=json.loads(path.read_text());required=('protocol','status','mode','repository_commit','artifacts','training_seed_values','mathqa_evaluation_seed_values','approved_by_user','bound_config_sha256')
 if any(k not in manifest for k in required) or manifest['protocol']!='step3d-training-authorization-v1' or manifest['status']!='approved' or manifest['mode'] not in ('exact','paper-guided-controlled-replication') or manifest['approved_by_user'] is not True:raise RuntimeError('Step 3D authorization manifest invalid')
 binding=dict(config);binding.pop('_config_path',None);binding['asset_authorization_manifest_sha256']=None;binding_sha=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if manifest['bound_config_sha256']!=binding_sha:raise RuntimeError('authorization manifest is not bound to this protocol config')
 if len(manifest['training_seed_values'])!=3 or len(set(manifest['training_seed_values']))!=3 or len(manifest['mathqa_evaluation_seed_values'])!=8 or len(set(manifest['mathqa_evaluation_seed_values']))!=8:raise RuntimeError('authorization seed cardinality mismatch')
 required_artifacts={'repository_snapshot','dataset_manifest','prompt_protocol','packing_protocol','optimizer_protocol','training_seed_map','mathqa_evaluation_seed_map'}
 if set(manifest['artifacts'])!=required_artifacts:raise RuntimeError('authorization artifact set mismatch')
 for name,spec in manifest['artifacts'].items():
  artifact=Path(spec.get('path',''));digest=spec.get('sha256')
  if not artifact.is_file() or not isinstance(digest,str) or len(digest)!=64 or sha256(artifact)!=digest:raise RuntimeError(f'authorization artifact mismatch: {name}')
 if manifest['mode']=='exact' and not manifest['repository_commit']:raise RuntimeError('exact mode requires authoritative repository commit')
 return manifest
