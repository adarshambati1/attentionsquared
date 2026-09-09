#!/usr/bin/env python3
"""QUARANTINED functional-v1 training (off-by-one cached targets)."""
from __future__ import annotations
import argparse,json,random,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np,torch
from transformers import AutoModelForCausalLM
from src.evaluation.correctness import INVALID_FUNCTIONAL_V1_LABEL, ensure_audit_rerun_root, require_invalid_v1_opt_in, token_normalized_kl
from src.models.attention2 import Attention2Lite

def load(path,device):
 with np.load(path) as a:
  return (torch.from_numpy(a['h0'].astype(np.float32)).to(device),torch.from_numpy(a['x'].astype(np.float32)).to(device),torch.from_numpy(a['teacher_logits'].astype(np.float32)).to(device),torch.from_numpy(a['input_ids']).to(device),int(a['answer_start']),int(a['valid_end']))
def main(config,cache,output,min_steps,max_steps,allow_invalid_v1=False):
 require_invalid_v1_opt_in(allow_invalid_v1, Path(__file__).name)
 ensure_audit_rerun_root(output)
 c=json.loads(config.read_text()); device=torch.device('cuda'); train=sorted((cache/'train').glob('*.npz')); val=sorted((cache/'val').glob('*.npz'))[:16]; hug=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True).eval().to(device)
 for p in hug.parameters(): p.requires_grad_(False)
 summary=[]
 for k in [1,2,4]:
  torch.manual_seed(7000+k); torch.cuda.manual_seed_all(7000+k); random.seed(7000+k); np.random.seed(7000+k); h0,x,tl,ids,s,e=load(train[0],device); a2=Attention2Lite(h0.shape[-1],16,16,k,1,16.0,False).to(device); opt=torch.optim.AdamW(a2.parameters(),lr=1e-4,weight_decay=0.0); logs=[]; start=time.perf_counter()
  best_val=float('inf'); best_state=None
  no_improve=0
  for step in range(1,max_steps+1):
   h0,x,tl,ids,s,e=load(random.choice(train),device); opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16): z,_,_=a2(h0[None],x[None]); student=hug(input_ids=ids[None],input_states=z[:,15].to(torch.bfloat16),num_steps=0,use_cache=False).logits[:,s:e]; loss=token_normalized_kl(student,tl); total_kl=float(loss.item()*(e-s)); valid_tokens=e-s
   loss.backward(); torch.nn.utils.clip_grad_norm_(a2.parameters(),1.0); opt.step(); logs.append(loss.item())
   if step%200==0:
    vals=[]
    with torch.inference_mode():
     for vp in val:
      vh,vx,vt,vi,vs,ve=load(vp,device); vz,_,_=a2(vh[None],vx[None]); sl=hug(input_ids=vi[None],input_states=vz[:,15].to(torch.bfloat16),num_steps=0,use_cache=False).logits[:,vs:ve]; vals.append(token_normalized_kl(sl,vt).item())
    vm=float(np.mean(vals)); print(f'functional-full K={k} step={step}/{max_steps} train_kl_token={np.mean(logs[-100:]):.6g} val_kl_token={vm:.6g} last_total_kl={total_kl:.6g} last_valid_tokens={valid_tokens} teacher_self_kl=0 seconds={time.perf_counter()-start:.1f}',flush=True)
    if vm < best_val-1e-3: best_val=vm; best_state={n:v.detach().cpu().clone() for n,v in a2.state_dict().items()}; no_improve=0
    else: no_improve += 1
    if step>=min_steps and no_improve>=5: print(f'K={k} plateau stop at step={step}',flush=True); break
  if best_state is not None: a2.load_state_dict(best_state)
  result={'K':k,'scientific_validity':INVALID_FUNCTIONAL_V1_LABEL,'objective':'full_answer_teacher_KL_only','steps':step,'parameters':sum(p.numel() for p in a2.parameters()),'train_kl_token_final':float(logs[-1]),'best_val_kl_token':best_val,'teacher_self_kl':0.0,'seconds':time.perf_counter()-start}; kd=output/f'K{k}'; kd.mkdir(exist_ok=True); torch.save({'state_dict':a2.state_dict(),'config':{**c,'K':k,'objective':'full_answer_teacher_KL_only'},'result':result},kd/'model.pt'); (kd/'result.json').write_text(json.dumps(result,indent=2)); summary.append(result); del a2; torch.cuda.empty_cache()
 (output/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004b_fixed_k.json')); p.add_argument('--cache',type=Path,default=Path('results/004_functional/training_cache')); p.add_argument('--output',type=Path,default=Path('results/004_functional/models_full')); p.add_argument('--min-steps',type=int,default=2000); p.add_argument('--max-steps',type=int,default=5000); p.add_argument('--allow-invalid-v1',action='store_true',help='audit rerun only; outputs remain scientifically invalid'); a=p.parse_args(); main(a.config,a.cache,a.output,a.min_steps,a.max_steps,a.allow_invalid_v1)
