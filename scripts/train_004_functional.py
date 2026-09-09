#!/usr/bin/env python3
"""QUARANTINED functional-v1 training on invalid historical inputs."""
from __future__ import annotations
import argparse,json,random,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np,torch
from transformers import AutoModelForCausalLM
from src.evaluation.correctness import INVALID_FUNCTIONAL_V1_LABEL, ensure_audit_rerun_root, require_invalid_v1_opt_in
from src.models.attention2 import Attention2Lite

def load(path,device):
 with np.load(path) as a: return tuple(torch.from_numpy(a[k].astype(np.float32)).to(device) for k in ('h0','x','trajectory')), torch.from_numpy(a['input_ids']).to(device)
def logits_kl(student,teacher):
 t=torch.softmax(teacher.float(),-1); return torch.nn.functional.kl_div(torch.log_softmax(student.float(),-1),t,reduction='batchmean')
def main(config,cache,output,steps,allow_invalid_v1=False):
 require_invalid_v1_opt_in(allow_invalid_v1, Path(__file__).name)
 ensure_audit_rerun_root(output)
 c=json.loads(config.read_text()); device=torch.device('cuda'); train=sorted((cache/'train').glob('*.npz')); val=sorted((cache/'val').glob('*.npz'))[:32]; hug=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True).eval().to(device)
 for p in hug.parameters(): p.requires_grad_(False)
 tok_results=[]
 for k in [1,2,4]:
  torch.manual_seed(6000+k); torch.cuda.manual_seed_all(6000+k); random.seed(6000+k); np.random.seed(6000+k); (h0,x,target),ids=load(train[0],device); a2=Attention2Lite(h0.shape[-1],16,16,k,1,16.0,False).to(device); opt=torch.optim.AdamW(a2.parameters(),lr=1e-4,weight_decay=0.0); log=[]; start=time.perf_counter()
  for step in range(1,steps+1):
   path=random.choice(train); (h0,x,target),ids=load(path,device); ids=ids.unsqueeze(0); opt.zero_grad(set_to_none=True)
   with torch.inference_mode(): teacher=hug(input_ids=ids,input_states=target[-1].to(torch.bfloat16).unsqueeze(0),num_steps=0,use_cache=False).logits.detach()
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16): z,_,_=a2(h0[None],x[None]); pred=z[:,15]; student=hug(input_ids=ids,input_states=pred.to(torch.bfloat16),num_steps=0,use_cache=False).logits; loss=logits_kl(student,teacher)
   loss.backward(); torch.nn.utils.clip_grad_norm_(a2.parameters(),1.0); opt.step(); log.append(loss.item())
   if step%100==0:
    vals=[]
    with torch.inference_mode():
     for vp in val:
      (vh,vx,vt),vi=load(vp,device); vi=vi.unsqueeze(0); tz=hug(input_ids=vi,input_states=vt[-1].to(torch.bfloat16).unsqueeze(0),num_steps=0,use_cache=False).logits; vz,_,_=a2(vh[None],vx[None]); sz=hug(input_ids=vi,input_states=vz[:,15].to(torch.bfloat16),num_steps=0,use_cache=False).logits; vals.append(logits_kl(sz,tz).item())
    print(f'functional K={k} step={step}/{steps} train_kl={np.mean(log[-100:]):.6g} val_kl={np.mean(vals):.6g} seconds={time.perf_counter()-start:.1f}',flush=True)
  result={'K':k,'scientific_validity':INVALID_FUNCTIONAL_V1_LABEL,'objective':'teacher_logit_KL_only','steps':steps,'parameters':sum(p.numel() for p in a2.parameters()),'train_kl_final':float(log[-1]),'seconds':time.perf_counter()-start}; kd=output/f'K{k}'; kd.mkdir(exist_ok=True); torch.save({'state_dict':a2.state_dict(),'config':{**c,'K':k,'objective':'teacher_logit_KL_only'},'result':result},kd/'model.pt'); (kd/'result.json').write_text(json.dumps(result,indent=2)); tok_results.append(result); del a2; torch.cuda.empty_cache()
 (output/'summary.json').write_text(json.dumps(tok_results,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004b_fixed_k.json')); p.add_argument('--cache',type=Path,default=Path('results/004b_fixed_k/cache')); p.add_argument('--output',type=Path,default=Path('results/004_functional/models')); p.add_argument('--steps',type=int,default=500); p.add_argument('--allow-invalid-v1',action='store_true',help='audit rerun only; outputs remain scientifically invalid'); a=p.parse_args(); main(a.config,a.cache,a.output,a.steps,a.allow_invalid_v1)
