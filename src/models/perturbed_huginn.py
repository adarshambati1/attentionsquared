"""Frozen Huginn with one deterministic token-relative latent perturbation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hashlib
import torch
from torch import nn

@dataclass
class PerturbedOutput:
 logits:torch.Tensor
 latent_states:torch.Tensor
 past_key_values:Any
 perturbation_relative_norms:torch.Tensor

def derived_noise_seed(*,base_seed:int,example_id:int,perturb_depth:int,seed_index:int,token_position:int)->int:
 payload=f'{base_seed}:{example_id}:{perturb_depth}:{seed_index}:{token_position}'.encode()
 return int.from_bytes(hashlib.sha256(payload).digest()[:8],'big')%(2**63-1)

def token_relative_noise(state:torch.Tensor,*,sigma:float,example_id:int,perturb_depth:int,seed_index:int,base_seed:int,token_positions:torch.Tensor,token_mask:torch.Tensor|None=None)->tuple[torch.Tensor,torch.Tensor]:
 """Deterministic independent directions per absolute token; shared across sigma."""
 if state.shape[:2]!=(token_positions.shape[0],token_positions.shape[1]):raise ValueError('position shape mismatch')
 if token_mask is not None and (token_mask.shape!=state.shape[:2] or token_mask.dtype!=torch.bool):raise ValueError('token_mask must be boolean [B,T]')
 noises=[]
 for batch in range(state.shape[0]):
  row=[]
  for position in token_positions[batch].tolist():
   seed=derived_noise_seed(base_seed=base_seed,example_id=example_id,perturb_depth=perturb_depth,seed_index=seed_index,token_position=int(position))
   generator=torch.Generator(device=state.device);generator.manual_seed(seed)
   epsilon=torch.randn(state.shape[-1],device=state.device,dtype=torch.float32,generator=generator);epsilon=epsilon/torch.linalg.vector_norm(epsilon).clamp_min(1e-12);row.append(epsilon)
  noises.append(torch.stack(row))
 base=state.float();direction=torch.stack(noises);norm=torch.linalg.vector_norm(base,dim=-1,keepdim=True);delta=float(sigma)*norm*direction
 if token_mask is not None:delta=delta*token_mask.unsqueeze(-1).to(delta.dtype)
 perturbed=base+delta;actual=torch.linalg.vector_norm(perturbed-base,dim=-1)/torch.linalg.vector_norm(base,dim=-1).clamp_min(1e-12)
 return perturbed,actual

class PerturbedHuginn(nn.Module):
 def __init__(self,huginn:nn.Module)->None:
  super().__init__();self.huginn=huginn
  for parameter in huginn.parameters():parameter.requires_grad_(False)
  huginn.eval()
 def train(self,mode:bool=True):super().train(False);self.huginn.eval();return self
 def new_dynamic_cache(self):
  cls=self.huginn.forward.__globals__.get('HuginnDynamicCache')
  if cls is None:raise RuntimeError('pinned Huginn does not expose HuginnDynamicCache')
  return cls()
 def forward(self,input_ids:torch.Tensor,input_states:torch.Tensor,*,depth:int=16,perturb_depth:int=8,sigma:float=0.0,example_id:int,perturbation_seed_index:int,perturbation_base_seed:int,attention_mask:torch.Tensor|None=None,past_key_values:Any=None,use_cache:bool=False,cache_position:torch.Tensor|None=None)->PerturbedOutput:
  if not 0<=perturb_depth<depth:raise ValueError('perturb depth must precede final depth')
  if input_ids.shape!=input_states.shape[:2]:raise ValueError('input/state shape mismatch')
  if attention_mask is None:attention_mask=torch.ones_like(input_ids,dtype=torch.bool)
  if use_cache and past_key_values is None:past_key_values=self.new_dynamic_cache()
  frequencies=self.huginn.freqs_cis[:,:input_ids.shape[1]] if cache_position is None else self.huginn.freqs_cis[:,cache_position]
  if cache_position is None:positions=torch.arange(input_ids.shape[1],device=input_ids.device).expand(input_ids.shape[0],-1)
  else:positions=cache_position.to(input_ids.device).expand(input_ids.shape[0],-1)
  block_index=torch.tensor(-1,device='cpu',dtype=torch.long);x=self.huginn.transformer.wte(input_ids)
  if self.huginn.emb_scale!=1:x=x*self.huginn.emb_scale
  for block in self.huginn.transformer.prelude:block_index+=1;x=block(x,frequencies,block_index,None,past_key_values)
  state=input_states;relative=None
  for loop in range(depth):
   if loop==perturb_depth:
    state,relative=token_relative_noise(state,sigma=sigma,example_id=example_id,perturb_depth=perturb_depth,seed_index=perturbation_seed_index,base_seed=perturbation_base_seed,token_positions=positions,token_mask=attention_mask)
   state,block_index=self.huginn.core_block_forward(state,x,frequencies,None,past_key_values,block_index,loop)
  state=self.huginn.transformer.ln_f(state);block_index=torch.tensor(0,device='cpu',dtype=torch.long);coda=state
  for block in self.huginn.transformer.coda:block_index-=1;coda=block(coda,frequencies,block_index,None,past_key_values)
  logits=self.huginn.lm_head(self.huginn.transformer.ln_f(coda)).float()
  if relative is None:raise RuntimeError('perturbation was not applied')
  return PerturbedOutput(logits,state,past_key_values,relative)
