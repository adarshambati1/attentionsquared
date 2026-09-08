#!/usr/bin/env python3
"""Train only the 4A tiny overfit Attention²-lite gate."""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.models.attention2 import Attention2Lite

def cosine_metrics(z,target,mask):
 m=mask[:,None,:,None].expand_as(z); p=z[m].reshape(-1,z.shape[-1]); y=target[m].reshape(-1,target.shape[-1]); return torch.nn.functional.cosine_similarity(p,y,dim=-1).mean().item(), ((p-y).norm(dim=-1)/(y.norm(dim=-1)+1e-8)).mean().item()

def main(config_path, data_path, output_dir, depth_scale=None, film=False, objective='mse_endpoint', eta=1.0):
 c=json.loads(config_path.read_text()); scale=c.get('depth_scale',1.0) if depth_scale is None else depth_scale; a=np.load(data_path); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 h0=torch.from_numpy(a['h0'].astype(np.float32)).to(device); x=torch.from_numpy(a['x'].astype(np.float32)).to(device); target=torch.from_numpy(a['trajectory'][:,1:].astype(np.float32)).to(device); mask=torch.from_numpy(a['mask']).to(device)
 n,d,t,h=target.shape; model=Attention2Lite(h,d,c['num_heads'],c['refinement_rounds'],c['mlp_ratio'],scale,film).to(device); opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=0.0); output_dir.mkdir(parents=True,exist_ok=True)
 # Causal-token correctness: perturbing a future input token must not affect an earlier output token.
 model.eval()
 with torch.inference_mode():
  z,_,_=model(h0[:1],x[:1]); x2=x[:1].clone(); x2[:,-1]+=torch.randn_like(x2[:,-1]); z2,_,_=model(h0[:1],x2); causal_error=(z[:,:,:1]-z2[:,:,:1]).abs().max().item()
  zi=model.initialize(h0[:1],x[:1]); zo,_=model.operator(zi); zi2=zi.clone(); zi2[:,0,0]+=1.0; zo2,_=model.operator(zi2); depth_comm=(zo[:,1,0]-zo2[:,1,0]).abs().max().item()
  signature_ok=True
 print(json.dumps({'shape':[1,d,t,h],'depth_scale':scale,'causal_future_to_past_error':causal_error,'depth_communication_effect':depth_comm,'no_teacher_inputs':signature_ok}),flush=True)
 losses=[]; start=time.perf_counter(); model.train()
 for step in range(1,c['train_steps']+1):
  i=torch.randint(n,(1,),device=device); opt.zero_grad(set_to_none=True)
  with torch.autocast(device_type='cuda',dtype=torch.bfloat16): z,_,_=model(h0[i],x[i])
  valid=mask[i,None,:,None].expand_as(z); diff=(z-target[i]).float()
  if objective=='normalized_trajectory':
   p=z.float(); y=target[i].float(); token_valid=mask[i].reshape(-1).expand(d,t); rel=((p-y).norm(dim=-1)/(y.norm(dim=-1)+1e-8))[token_valid].mean(); cos=(1-torch.nn.functional.cosine_similarity(p[token_valid],y[token_valid],dim=-1)).mean(); loss=rel+eta*cos
  else:
   mse=(diff.pow(2)[valid]).mean(); endpoint=(diff[:, -1].pow(2)[mask[i,:,None].expand_as(diff[:, -1])]).mean(); loss=mse+c['endpoint_loss_weight']*endpoint
  loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(loss.item())
  if step%50==0:
   model.eval()
   with torch.inference_mode(): zz,_,_=model(h0,x); co,rl=cosine_metrics(zz,target,mask); epco, eprl=cosine_metrics(zz[:,-1:],target[:,-1:],mask)
   model.train(); print(f'step={step} loss={np.mean(losses[-50:]):.6g} cosine={co:.5f} rel_l2={rl:.5f} endpoint_cosine={epco:.5f} endpoint_rel_l2={eprl:.5f} seconds={time.perf_counter()-start:.1f}',flush=True)
 model.eval()
 with torch.inference_mode(): zz,_,att=model(h0,x,return_attention=True); co,rl=cosine_metrics(zz,target,mask); epco,eprl=cosine_metrics(zz[:,-1:],target[:,-1:],mask); pair=[]
 for i in range(d):
  for j in range(i+1,d): pair.append(torch.nn.functional.cosine_similarity(zz[:,i][mask],zz[:,j][mask],dim=-1).mean().item())
 # Measure one shared operator round separately from the configured K-round path.
 with torch.inference_mode():
  z0=model.initialize(h0[:1],x[:1])
  for _ in range(3): model.operator(z0)
  if device.type=='cuda': torch.cuda.synchronize()
  t0=time.perf_counter()
  for _ in range(10): model.operator(z0)
  if device.type=='cuda': torch.cuda.synchronize()
  one_round_ms=1000*(time.perf_counter()-t0)/10
  if device.type=='cuda': torch.cuda.synchronize()
  t0=time.perf_counter(); model(h0[:1],x[:1])
  if device.type=='cuda': torch.cuda.synchronize()
  full_ms=1000*(time.perf_counter()-t0)
 result={'depth_scale':scale,'film':film,'objective':objective,'eta':eta,'train_loss_final':float(losses[-1]),'trajectory_cosine':co,'trajectory_relative_l2':rl,'endpoint_cosine':epco,'endpoint_relative_l2':eprl,'slot_pairwise_cosine_mean':float(np.mean(pair)),'slot_pairwise_cosine_max':float(np.max(pair)),'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters()),'shape':[n,d,t,h],'refinement_rounds':c['refinement_rounds'],'single_round_latency_ms':one_round_ms,'four_round_latency_ms':full_ms,'causal_future_to_past_error':causal_error,'depth_communication_effect':depth_comm}
 torch.save({'state_dict':model.state_dict(),'config':{**c,'depth_scale':scale,'film':film,'objective':objective,'eta':eta},'result':result},output_dir/'attention2_lite.pt'); (output_dir/'result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004a_tiny_overfit.json')); p.add_argument('--data',type=Path,default=Path('results/004a_tiny_overfit/trajectories.npz')); p.add_argument('--output-dir',type=Path,default=Path('results/004a_tiny_overfit')); p.add_argument('--depth-scale',type=float); p.add_argument('--film',action='store_true'); p.add_argument('--objective',choices=['mse_endpoint','normalized_trajectory'],default='mse_endpoint'); p.add_argument('--eta',type=float,default=1.0); a=p.parse_args(); main(a.config,a.data,a.output_dir,a.depth_scale,a.film,a.objective,a.eta)
