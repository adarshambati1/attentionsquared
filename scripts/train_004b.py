#!/usr/bin/env python3
"""Train independent fixed-K A² models on the shared frozen trajectory cache."""
from __future__ import annotations
import argparse,json,random,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np,torch
from src.models.attention2 import Attention2Lite

def load(path,device):
 with np.load(path) as a:
  return tuple(torch.from_numpy(a[k].astype(np.float32)).to(device) for k in ('h0','x','trajectory')), torch.from_numpy(a['input_ids'])

def loss_fn(z,target,mask):
 p=z[0].float(); y=target.float(); valid=mask.reshape(-1).expand(16,mask.numel()); rel=((p-y).norm(dim=-1)/(y.norm(dim=-1)+1e-8))[valid].mean(); cos=(1-torch.nn.functional.cosine_similarity(p[valid],y[valid],dim=-1)).mean(); return rel+cos

def evaluate(model,paths,device):
 vals=[]; model.eval()
 with torch.inference_mode():
  for path in paths:
   (h0,x,target),_=load(path,device); mask=torch.ones(target.shape[1],dtype=torch.bool,device=device); z,_,_=model(h0[None],x[None]); p=z[0].float(); y=target.float(); valid=mask[None,:].expand(16,mask.numel()); vals.append({'loss':loss_fn(z,target,mask).item(),'cosine':torch.nn.functional.cosine_similarity(p[valid],y[valid],dim=-1).mean().item()})
 return {k:float(np.mean([v[k] for v in vals])) for k in vals[0]}

def main(config_path,cache,output,steps):
 c=json.loads(config_path.read_text()); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); train=sorted((cache/'train').glob('*.npz')); val=sorted((cache/'val').glob('*.npz'))[:32]; output.mkdir(parents=True,exist_ok=True); summary=[]
 for k in [1,2,4,8]:
  torch.manual_seed(5000+k); torch.cuda.manual_seed_all(5000+k); random.seed(5000+k); np.random.seed(5000+k); sample,_=load(train[0],device); h=sample[0].shape[-1]; model=Attention2Lite(h,16,16,k,1,16.0,False).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.0); log=[]; start=time.perf_counter(); model.train()
  for step in range(1,steps+1):
   path=train[random.randrange(len(train))]; (h0,x,target),_=load(path,device); mask=torch.ones(target.shape[1],dtype=torch.bool,device=device); opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16): z,_,_=model(h0[None],x[None])
   loss=loss_fn(z,target,mask); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); log.append(loss.item())
   if step%100==0:
    v=evaluate(model,val,device); print(f'K={k} step={step}/{steps} train={np.mean(log[-100:]):.5f} val_loss={v["loss"]:.5f} val_cos={v["cosine"]:.5f} seconds={time.perf_counter()-start:.1f}',flush=True)
  result={'K':k,'steps':steps,'parameters':sum(p.numel() for p in model.parameters()),'train_loss_final':float(log[-1]),'val':evaluate(model,val,device),'seconds':time.perf_counter()-start}
  kd=output/f'K{k}'; kd.mkdir(exist_ok=True); torch.save({'state_dict':model.state_dict(),'config':{**c,'K':k},'result':result},kd/'model.pt'); (kd/'result.json').write_text(json.dumps(result,indent=2)); summary.append(result); del model; torch.cuda.empty_cache()
 (output/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004b_fixed_k.json')); p.add_argument('--cache',type=Path,default=Path('results/004b_fixed_k/cache')); p.add_argument('--output',type=Path,default=Path('results/004b_fixed_k/models')); p.add_argument('--steps',type=int,default=1000); a=p.parse_args(); main(a.config,a.cache,a.output,a.steps)
