#!/usr/bin/env python3
"""Diagnostics for the frozen 4A checkpoint; does not retrain."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np, torch
from src.models.attention2 import Attention2Lite

def cosmat(states,mask):
 d=states.shape[1]; out=np.zeros((d,d))
 flat=[states[:,i][mask].float() for i in range(d)]
 for i in range(d):
  for j in range(d): out[i,j]=torch.nn.functional.cosine_similarity(flat[i],flat[j],dim=-1).mean().item()
 return out

def per_depth(pred,target,mask):
 rows=[]
 for d in range(pred.shape[1]):
  p=pred[:,d][mask].float(); y=target[:,d][mask].float(); rows.append({'depth':d+1,'cosine':torch.nn.functional.cosine_similarity(p,y,dim=-1).mean().item(),'relative_l2':((p-y).norm(dim=-1)/(y.norm(dim=-1)+1e-8)).mean().item(),'mse':(p-y).pow(2).mean().item()})
 return rows

def main(config,data,checkpoint,outdir):
 c=json.loads(config.read_text()); a=np.load(data); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); h0=torch.from_numpy(a['h0'].astype('float32')).to(device); x=torch.from_numpy(a['x'].astype('float32')).to(device); target=torch.from_numpy(a['trajectory'][:,1:].astype('float32')).to(device); mask=torch.from_numpy(a['mask']).to(device)
 ck=torch.load(checkpoint,map_location='cpu'); model=Attention2Lite(h0.shape[-1],16,c['num_heads'],c['refinement_rounds'],c['mlp_ratio']).to(device); model.load_state_dict(ck['state_dict']); model.eval(); outdir.mkdir(parents=True,exist_ok=True)
 with torch.inference_mode():
  z0=model.initialize(h0,x); states=[z0]; attns=[]; z=z0
  for _ in range(c['refinement_rounds']): z,att=model.operator(z,return_attention=True); states.append(z); attns.append(att['depth'])
 pred=states[-1]; rows=per_depth(pred,target,mask); (outdir/'per_depth.json').write_text(json.dumps(rows,indent=2));
 np.savetxt(outdir/'teacher_pairwise_cosine.csv',cosmat(target,mask),delimiter=','); np.savetxt(outdir/'predicted_pairwise_cosine.csv',cosmat(pred,mask),delimiter=',')
 round_stats=[]
 for k,s in enumerate(states):
  m=cosmat(s,mask); round_stats.append({'stage':'Z0' if k==0 else f'Z{k}','pairwise_cosine_mean_offdiag':float((m.sum()-np.trace(m))/(m.size-len(m))),'pairwise_cosine_max_offdiag':float((m-np.eye(16)).max()),'init_or_round':k})
 np.savetxt(outdir/'predicted_slot_cosine_by_stage.csv',np.array([[r['init_or_round'],r['pairwise_cosine_mean_offdiag'],r['pairwise_cosine_max_offdiag']] for r in round_stats]),delimiter=',',header='stage_index,mean_offdiag,max_offdiag',comments='')
 distance=[]
 for d in range(16):
  p=pred[:,d][mask].float(); y=target[:,-1][mask].float(); distance.append({'depth_slot':d+1,'cosine_to_teacher_h16':torch.nn.functional.cosine_similarity(p,y,dim=-1).mean().item(),'relative_l2_to_teacher_h16':((p-y).norm(dim=-1)/(y.norm(dim=-1)+1e-8)).mean().item()})
 (outdir/'distance_to_teacher_h16.json').write_text(json.dumps(distance,indent=2)); (outdir/'stage_slot_cosine.json').write_text(json.dumps(round_stats,indent=2))
 att_stats=[]
 for k,w in enumerate(attns,1):
  # [B*T, heads, D, D], retain only valid token rows
  ww=w.reshape(h0.shape[0],-1,w.shape[1],16,16); valid=mask[:,:,None,None,None].expand_as(ww); vals=ww[valid].reshape(-1,16,16); probs=vals.clamp_min(1e-8); entropy=-(probs*probs.log()).sum(-1).mean().item(); normalized=entropy/np.log(16); avg=vals.mean(0).mean(0).cpu().numpy(); variation=vals.std(0).mean().item(); np.savetxt(outdir/f'depth_attention_round{k}.csv',avg,delimiter=','); att_stats.append({'round':k,'entropy':entropy,'normalized_entropy':normalized,'mean_attention_std_across_tokens_heads':variation})
 (outdir/'attention_stats.json').write_text(json.dumps(att_stats,indent=2));
 base=model.init_proj(torch.cat([h0,x],-1)); emb=model.depth_embedding; emb_norm=emb.norm(dim=-1).detach().cpu().numpy(); base_norm=base[mask].norm(dim=-1).mean().item(); (outdir/'embedding_stats.json').write_text(json.dumps({'depth_embedding_norms':emb_norm.tolist(),'depth_embedding_norm_mean':float(emb_norm.mean()),'shared_initialization_norm_mean':base_norm,'embedding_to_base_norm_ratio':float(emb_norm.mean()/base_norm)},indent=2))
 losses={'trajectory_mse':float(sum(r['mse'] for r in rows)/16),'endpoint_mse':rows[-1]['mse'],'total_with_endpoint_weight':float(sum(r['mse'] for r in rows)/16+c['endpoint_loss_weight']*rows[-1]['mse'])}; (outdir/'loss_decomposition.json').write_text(json.dumps(losses,indent=2))
 # One diagnostic backward pass for gradient norms; this is not optimization.
 model.train(); model.zero_grad(set_to_none=True); z,_,_=model(h0[:1],x[:1]); valid=mask[:1,None,:,None].expand_as(z); diff=(z-target[:1]).float(); loss=diff.pow(2)[valid].mean()+c['endpoint_loss_weight']*diff[:,-1].pow(2)[mask[:1,:,None].expand_as(diff[:,-1])].mean(); loss.backward(); groups={}
 for name,module in [('depth_embedding',model.depth_embedding),('init_proj',model.init_proj),('token_qkv',model.token_attn),('depth_qkv',model.depth_attn),('mlp',model.mlp)]: groups[name]=float(torch.sqrt(sum((p.grad.detach().float().pow(2).sum() for p in module.parameters() if p.grad is not None))).item())
 (outdir/'gradient_norms.json').write_text(json.dumps(groups,indent=2))
 try:
  import matplotlib.pyplot as plt
  ds=[r['depth'] for r in rows]; plt.figure(); plt.plot(ds,[r['cosine'] for r in rows],label='cosine'); plt.plot(ds,[r['relative_l2'] for r in rows],label='relative L2'); plt.legend(); plt.xlabel('Teacher depth'); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(outdir/'per_depth_reconstruction.png',dpi=160); plt.close()
  for name,m in [('teacher',cosmat(target,mask)),('predicted',cosmat(pred,mask))]: plt.figure(); plt.imshow(m,vmin=0,vmax=1,cmap='viridis'); plt.colorbar(); plt.title(name); plt.xlabel('slot'); plt.ylabel('slot'); plt.tight_layout(); plt.savefig(outdir/f'{name}_pairwise_cosine.png',dpi=160); plt.close()
 except ImportError: pass
 print(json.dumps({'per_depth':rows,'stages':round_stats,'attention':att_stats,'losses':losses},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004a_tiny_overfit.json')); p.add_argument('--data',type=Path,default=Path('results/004a_tiny_overfit/trajectories.npz')); p.add_argument('--checkpoint',type=Path,default=Path('results/004a_tiny_overfit/attention2_lite.pt')); p.add_argument('--output-dir',type=Path,default=Path('results/004a_tiny_overfit/diagnostics')); a=p.parse_args(); main(a.config,a.data,a.checkpoint,a.output_dir)
