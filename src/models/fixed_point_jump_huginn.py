"""Functional one-shot fixed-point jump into frozen Huginn coda."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn

class FixedPointJumpPredictor(nn.Module):
    """One shared tokenwise residual bottleneck map from (h0, x) to h*."""
    def __init__(self, hidden_size: int, bottleneck_size: int = 256) -> None:
        super().__init__()
        self.in_proj=nn.Linear(2*hidden_size,bottleneck_size)
        self.out_proj=nn.Linear(bottleneck_size,hidden_size)
        nn.init.zeros_(self.out_proj.weight);nn.init.zeros_(self.out_proj.bias)
    def forward(self,h0:torch.Tensor,x:torch.Tensor)->torch.Tensor:
        delta=self.out_proj(torch.nn.functional.gelu(self.in_proj(torch.cat((h0,x),dim=-1))))
        return h0+delta

@dataclass
class FixedPointJumpOutput:
    logits:torch.Tensor
    latent_states:torch.Tensor
    past_key_values:Any
    fixed_point_image:torch.Tensor|None
    core_steps_executed:int

class FixedPointJumpHuginn(nn.Module):
    def __init__(self,huginn:nn.Module,bottleneck_size:int=256)->None:
        super().__init__();self.huginn=huginn;self.predictor=FixedPointJumpPredictor(int(huginn.config.n_embd),bottleneck_size)
        for p in huginn.parameters():p.requires_grad_(False)
        huginn.eval()
    def train(self,mode:bool=True):
        super().train(mode);self.huginn.eval();return self
    def trainable_parameters(self):return self.predictor.parameters()
    def new_dynamic_cache(self):
        cls=self.huginn.forward.__globals__.get('HuginnDynamicCache')
        if cls is None:raise RuntimeError('pinned Huginn does not expose HuginnDynamicCache')
        return cls()
    def forward(self,input_ids:torch.Tensor,input_states:torch.Tensor,*,compute_fixed_point:bool=False,depth:int=0,mode:str='jump',attention_mask:torch.Tensor|None=None,past_key_values:Any=None,use_cache:bool=False,cache_position:torch.Tensor|None=None)->FixedPointJumpOutput:
        if depth!=0 or mode!='jump':raise ValueError('one-shot jump executes no recurrent inference loops')
        if input_ids.shape!=input_states.shape[:2]:raise ValueError('input/state shape mismatch')
        if compute_fixed_point and use_cache:raise ValueError('fixed-point training image cannot mutate an inference cache')
        if use_cache and past_key_values is None:past_key_values=self.new_dynamic_cache()
        frequencies=self.huginn.freqs_cis[:,:input_ids.shape[1]] if cache_position is None else self.huginn.freqs_cis[:,cache_position]
        block_index=torch.tensor(-1,device='cpu',dtype=torch.long);x=self.huginn.transformer.wte(input_ids)
        if self.huginn.emb_scale!=1:x=x*self.huginn.emb_scale
        for block in self.huginn.transformer.prelude:
            block_index+=1;x=block(x,frequencies,block_index,None,past_key_values)
        h_star=self.predictor(input_states,x);fp_image=None;core_steps=0
        if compute_fixed_point:
            fp_image,_=self.huginn.core_block_forward(h_star,x,frequencies,None,None,block_index,0);core_steps=1
        state=self.huginn.transformer.ln_f(h_star);block_index=torch.tensor(0,device='cpu',dtype=torch.long)
        for block in self.huginn.transformer.coda:
            block_index-=1;state=block(state,frequencies,block_index,None,past_key_values)
        logits=self.huginn.lm_head(self.huginn.transformer.ln_f(state)).float()
        return FixedPointJumpOutput(logits,h_star,past_key_values,fp_image,core_steps)
