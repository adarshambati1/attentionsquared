#!/usr/bin/env python3
"""Functional-only A² training on full teacher-forced answer trajectories."""
from __future__ import annotations
import argparse,json,random,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np,torch
from transformers import AutoModelForCausalLM
from src.models.attention2 import Attention2Lite

def load(path,device):
 with np.load(path) as a:
  return (torch.from_numpy(a['h0'].astype(np.float32)).to(device),torch.from_numpy(a['x'].astype(np.float32)).to(device),torch.from_numpy(a['teacher_logits'].astype(np.float32)).to(device),torch.from_numpy(a['input_ids']).to(device),int(a['answer_start']),int(a['valid_end']))
def main(config,cache,output,steps):
 c=json.loads(config.read_text()); device=torch.device('cuda'); train=sorted((cache/'train').glob('*.npz')); val=sorted((cache/'val').glob('*.npz'))[:16]; hug=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True).eval().to(device)
 for p in hug.parameters(): p.requires_grad_(False)
 output.mkdir(parents=True,exist_ok=True); summary=[]
 for k in [1,2,4]:
  torch.manual_seed(7000+k); torch.cuda.manual_seed_all(7000+k); random.seed(7000+k); np.random.seed(7000+k); h0,x,tl,ids,s,e=load(train[0],device); a2=Attention2Lite(h0.shape[-1],16,16,k,1,16.0,False).to(device); opt=torch.optim.AdamW(a2.parameters(),lr=1e-4,weight_decay=0.0); logs=[]; start=time.perf_counter()
  for step in range(1,steps+1):
   h0,x,tl,ids,s,e=load(random.choice(train),device); opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.bfloat16): z,_,_=a2(h0[None],x[None]); student=hug(input_ids=ids[None],input_states=z[:,15].to(torch.bfloat16),num_steps=0,use_cache=False).logits[:,s:e]; token_kl=torch.nn.functional.kl_div(torch.log_softmax(student.float(),-1),torch.softmax(tl,-1),reduction='none').sum(-1); loss=token_kl.mean()
   total_kl=float(token_kl.sum().item()); valid_tokens=int(token_kl.numel())
   loss.backward(); torch.nn.utils.clip_grad_norm_(a2.parameters(),1.0); opt.step(); logs.append(loss.item())
   if step%100==0: print(f'functional-full K={k} step={step}/{steps} train_kl_token={np.mean(logs[-100:]):.6g} last_total_kl={total_kl:.6g} last_valid_tokens={valid_tokens} teacher_self_kl=0 seconds={time.perf_counter()-start:.1f}',flush=True)
  result={'K':k,'objective':'full_answer_teacher_KL_only','steps':steps,'parameters':sum(p.numel() for p in a2.parameters()),'train_kl_token_final':float(logs[-1]),'teacher_self_kl':0.0,'seconds':time.perf_counter()-start}; kd=output/f'K{k}'; kd.mkdir(exist_ok=True); torch.save({'state_dict':a2.state_dict(),'config':{**c,'K':k,'objective':'full_answer_teacher_KL_only'},'result':result},kd/'model.pt'); (kd/'result.json').write_text(json.dumps(result,indent=2)); summary.append(result); del a2; torch.cuda.empty_cache()
 (output/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004b_fixed_k.json')); p.add_argument('--cache',type=Path,default=Path('results/004_functional/training_cache')); p.add_argument('--output',type=Path,default=Path('results/004_functional/models_full')); p.add_argument('--steps',type=int,default=500); a=p.parse_args(); main(a.config,a.cache,a.output,a.steps)
