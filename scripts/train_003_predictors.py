#!/usr/bin/env python3
"""Train the two deliberately boring tokenwise direct-jump predictors."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

class JumpMLP(nn.Module):
    def __init__(self, hidden: int, residual: bool):
        super().__init__()
        self.residual = residual
        self.in_proj = nn.Linear(2 * hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
    def forward(self, h0, x):
        delta = self.out_proj(torch.nn.functional.gelu(self.in_proj(torch.cat([h0, x], dim=-1))))
        return h0 + delta if self.residual else delta

def load_split(directory: Path):
    by_key = {k: [] for k in ("h0", "x", "h16")}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path) as archive:
            for key in by_key:
                by_key[key].append(archive[key].astype(np.float32))
    return tuple(np.concatenate(by_key[key], axis=0) for key in ("h0", "x", "h16"))

def metrics(model, arrays, device):
    h0, x, target = [torch.from_numpy(a) for a in arrays]
    vals=[]
    with torch.inference_mode():
        for i in range(0, len(h0), 2048):
            pred=model(h0[i:i+2048].to(device), x[i:i+2048].to(device)).float().cpu()
            y=target[i:i+2048]
            vals.append((pred-y).pow(2).mean(dim=-1).sum())
    pred_all=[]
    with torch.inference_mode():
        for i in range(0,len(h0),2048): pred_all.append(model(h0[i:i+2048].to(device),x[i:i+2048].to(device)).float().cpu())
    p=torch.cat(pred_all); y=target
    cosine=torch.nn.functional.cosine_similarity(p,y,dim=-1).mean().item()
    rel=((p-y).norm(dim=-1)/(y.norm(dim=-1)+1e-8)).mean().item()
    return float(sum(v.item() for v in vals)/len(target)), cosine, rel

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cache",type=Path,default=Path("results/003_direct_jump/cache")); ap.add_argument("--output",type=Path,default=Path("results/003_direct_jump/models")); ap.add_argument("--epochs",type=int,default=8); ap.add_argument("--batch-size",type=int,default=1024); args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=load_split(args.cache/"train"); val=load_split(args.cache/"val")
    hidden=train[0].shape[-1]; args.output.mkdir(parents=True,exist_ok=True)
    report=[]
    for residual in (False,True):
        name="residual" if residual else "direct"; model=JumpMLP(hidden,residual).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        h0,x,y=[torch.from_numpy(a) for a in train]; n=len(h0); best=1e99; best_state=None
        for epoch in range(args.epochs):
            order=torch.randperm(n); total=0.; start=time.perf_counter(); model.train()
            for ix in order.split(args.batch_size):
                pred=model(h0[ix].to(device),x[ix].to(device)); mse=(pred-y[ix].to(device)).pow(2).mean(); cos=(1-torch.nn.functional.cosine_similarity(pred,y[ix].to(device),dim=-1)).mean(); loss=mse+0.1*cos
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total += loss.item()*len(ix)
            model.eval(); vm,vc,vr=metrics(model,val,device); print(f"{name} epoch={epoch+1} train={total/n:.6g} val_mse={vm:.6g} val_cos={vc:.6f} val_rel_l2={vr:.6f} seconds={time.perf_counter()-start:.1f}",flush=True)
            if vm<best: best=vm; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_state); torch.save({"state_dict":model.state_dict(),"hidden":hidden,"residual":residual},args.output/f"{name}.pt")
        tm,tc,tr=metrics(model,train,device); vm,vc,vr=metrics(model,val,device)
        params=sum(p.numel() for p in model.parameters()); flops_per_token=2*((2*hidden)*hidden + hidden*hidden)
        report.append({"model":name,"parameters":params,"approx_flops_per_token":flops_per_token,"train_mse":tm,"train_cosine":tc,"train_relative_l2":tr,"val_mse":vm,"val_cosine":vc,"val_relative_l2":vr})
    (args.output/"training_summary.json").write_text(json.dumps(report,indent=2))
if __name__=="__main__": main()
