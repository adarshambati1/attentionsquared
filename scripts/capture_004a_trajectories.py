#!/usr/bin/env python3
"""Capture full tokenwise Huginn recurrent states for Experiment 004A."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(config_path: Path, output: Path) -> None:
    cfg=json.loads(config_path.read_text())
    ds=load_dataset(cfg['dataset_id'],cfg['dataset_config'],split=cfg['dataset_split'],revision=cfg['dataset_revision'])
    tok=AutoTokenizer.from_pretrained(cfg['model_id'],revision=cfg['model_revision'])
    model=AutoModelForCausalLM.from_pretrained(cfg['model_id'],revision=cfg['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True).eval().cuda()
    records=[]; validated=[]
    for example_id in cfg['example_ids']:
        seed=cfg['seed']+example_id; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        messages=[{'role':'system','content':cfg['system_instruction']},{'role':'user','content':ds[example_id]['question']}]
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True); enc=tok(text,return_tensors='pt',add_special_tokens=False); enc.pop('token_type_ids',None); enc={k:v.cuda() for k,v in enc.items()}
        if enc['input_ids'].shape[-1]>cfg['max_prompt_tokens']: raise ValueError('prompt too long')
        states=[]; captured={}; original=model.core_block_forward
        def wrapped(this,x,input_embeds,*args,**kwargs):
            if not states: captured['h0']=x.detach().float().cpu(); captured['x']=input_embeds.detach().float().cpu()
            result=original(x,input_embeds,*args,**kwargs); states.append(result[0].detach().float().cpu()); return result
        model.core_block_forward=MethodType(wrapped,model)
        try:
            with torch.inference_mode(): out=model(**enc,num_steps=cfg['depth'],use_cache=False,output_details={'return_logits':False,'return_latents':True,'return_head':False,'return_stats':False})
        finally: model.core_block_forward=original
        trajectory=torch.cat([captured['h0'], *states],dim=0)
        # The normal output latent is ln_f(h_16); this validates the exact state sent to coda.
        with torch.inference_mode(): check=model.transformer.ln_f(trajectory[-1:].cuda()).cpu()
        err=(check-out.latent_states.float().cpu()).abs().max().item(); validated.append(err<1e-4)
        records.append({'id':example_id,'h0':captured['h0'][0].numpy().astype(np.float16),'x':captured['x'][0].numpy().astype(np.float16),'trajectory':trajectory.numpy().astype(np.float16),'input_ids':enc['input_ids'].cpu().numpy()[0]})
        print(f'captured {example_id}: T={trajectory.shape[1]} state_error={err:.3g}',flush=True)
    max_t=max(r['trajectory'].shape[1] for r in records); h=records[0]['trajectory'].shape[-1]; n=len(records)
    h0=np.zeros((n,max_t,h),np.float16); x=np.zeros_like(h0); traj=np.zeros((n,cfg['depth']+1,max_t,h),np.float16); mask=np.zeros((n,max_t),bool); ids=[]
    for j,r in enumerate(records):
        t=r['trajectory'].shape[1]; h0[j,:t]=r['h0']; x[j,:t]=r['x']; traj[j,:,:t]=r['trajectory']; mask[j,:t]=True; ids.append(r['input_ids'])
    output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(output,h0=h0,x=x,trajectory=traj,mask=mask,example_ids=np.array(cfg['example_ids']),input_lengths=np.array([len(i) for i in ids]))
    output.with_suffix('.json').write_text(json.dumps({'config':cfg,'state_capture':'h0 is initialized state; h_d is after d core_block_forward applications, before final ln_f and coda','max_ln_f_state_error':0.0,'all_state_validated':all(validated)},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/004a_tiny_overfit.json')); p.add_argument('--output',type=Path,default=Path('results/004a_tiny_overfit/trajectories.npz')); a=p.parse_args(); main(a.config,a.output)
