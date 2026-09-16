"""Exact frozen plain-Huginn path at arbitrary deterministic recurrent depth."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn

@dataclass
class PlainDepthOutput:
 logits:torch.Tensor
 latent_states:torch.Tensor
 past_key_values:Any
 fixed_point_residual:float|None

class PlainDepthHuginn(nn.Module):
 def __init__(self,huginn:nn.Module)->None:
  super().__init__();self.huginn=huginn
  for parameter in huginn.parameters():parameter.requires_grad_(False)
  huginn.eval()
 def train(self,mode:bool=True):super().train(False);self.huginn.eval();return self
 def new_dynamic_cache(self):
  cls=self.huginn.forward.__globals__.get('HuginnDynamicCache')
  if cls is None:raise RuntimeError('pinned Huginn does not expose HuginnDynamicCache')
  return cls()
 def forward(self,input_ids:torch.Tensor,input_states:torch.Tensor,*,depth:int,mode:str='plain',attention_mask:torch.Tensor|None=None,past_key_values:Any=None,use_cache:bool=False,cache_position:torch.Tensor|None=None,compute_fixed_point_residual:bool=False)->PlainDepthOutput:
  if depth<=0:raise ValueError('depth must be positive')
  if mode!='plain':raise ValueError('plain depth wrapper supports only mode=plain')
  if input_ids.shape!=input_states.shape[:2]:raise ValueError('input/state shape mismatch')
  if attention_mask is None:attention_mask=torch.ones_like(input_ids,dtype=torch.bool)
  if attention_mask.shape!=input_ids.shape or attention_mask.dtype!=torch.bool:raise ValueError('attention_mask must be boolean [B,T]')
  if use_cache and past_key_values is None:past_key_values=self.new_dynamic_cache()
  frequencies=self.huginn.freqs_cis[:,:input_ids.shape[1]] if cache_position is None else self.huginn.freqs_cis[:,cache_position]
  index=torch.tensor(-1,device='cpu',dtype=torch.long);x=self.huginn.transformer.wte(input_ids)
  if self.huginn.emb_scale!=1:x=x*self.huginn.emb_scale
  for block in self.huginn.transformer.prelude:index+=1;x=block(x,frequencies,index,None,past_key_values)
  state=input_states
  for loop in range(depth):state,index=self.huginn.core_block_forward(state,x,frequencies,None,past_key_values,index,loop)
  residual=None
  if compute_fixed_point_residual:
   image,_=self.huginn.core_block_forward(state,x,frequencies,None,None,index,depth);mask=attention_mask.unsqueeze(-1);delta=torch.where(mask,(image-state).float(),0.0);base=torch.where(mask,state.float(),0.0);residual=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(base).clamp_min(1e-12))
  normalized=self.huginn.transformer.ln_f(state);coda=normalized;index=torch.tensor(0,device='cpu',dtype=torch.long)
  for block in self.huginn.transformer.coda:index-=1;coda=block(coda,frequencies,index,None,past_key_values)
  logits=self.huginn.lm_head(self.huginn.transformer.ln_f(coda)).float()
  return PlainDepthOutput(logits,normalized,past_key_values,residual)
