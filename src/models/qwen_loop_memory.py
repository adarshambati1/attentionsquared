"""Fixed-two-loop Qwen3 backbone controls for Step 3D."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn
import torch.nn.functional as F
Variant=Literal['plain','current','shared','recurtrace']
@dataclass
class QwenLoopOutput:
 logits:torch.Tensor
 hidden_states:torch.Tensor
class CurrentStateAdapter(nn.Module):
 def __init__(self,hidden:int,width:int):
  super().__init__();self.v_proj=nn.Linear(hidden,width,bias=False);self.out_proj=nn.Linear(width,hidden,bias=False);self.injection=nn.Parameter(torch.zeros(()))
 def forward(self,state):return self.injection*self.out_proj(self.v_proj(state))
class SharedLoopHistory(nn.Module):
 def __init__(self,hidden:int,width:int,heads:int):
  super().__init__();assert width%heads==0;self.heads=heads;self.dim=width//heads;self.q_proj=nn.Linear(hidden,width,bias=False);self.k_proj=nn.Linear(hidden,width,bias=False);self.v_proj=nn.Linear(hidden,width,bias=False);self.out_proj=nn.Linear(width,hidden,bias=False);self.injection=nn.Parameter(torch.zeros(()))
 def forward(self,current,history):
  h=torch.stack(history,2);b,t,l,_=h.shape;q=self.q_proj(current).view(b,t,self.heads,self.dim);k=self.k_proj(h).view(b,t,l,self.heads,self.dim);v=self.v_proj(h).view(b,t,l,self.heads,self.dim);w=torch.einsum('bthd,btlhd->bthl',q,k)*(self.dim**-.5);mixed=torch.einsum('bthl,btlhd->bthd',w.softmax(-1),v);return self.injection*self.out_proj(mixed.reshape(b,t,-1))
class RecurTraceLayerMemory(nn.Module):
 """Same-position loop-time memory with QK norm, W=3, signed distance bias and gates."""
 def __init__(self,hidden:int,width:int,heads:int,window:int=3):
  super().__init__();assert width%heads==0;self.heads=heads;self.dim=width//heads;self.window=window;self.q_proj=nn.Linear(hidden,width,bias=False);self.k_proj=nn.Linear(hidden,width,bias=False);self.v_proj=nn.Linear(hidden,width,bias=False);self.out_proj=nn.Linear(width,hidden,bias=False);self.scalar_gate=nn.Parameter(torch.ones(()));self.injection=nn.Parameter(torch.zeros(()));self.distance_slope=nn.Parameter(torch.tensor([2**(-8*i/heads) for i in range(heads)]));self.token_gate_in=nn.Linear(2*hidden,hidden);self.token_gate_out=nn.Linear(hidden,1);nn.init.zeros_(self.token_gate_out.weight);nn.init.constant_(self.token_gate_out.bias,-3.0)
 def forward(self,query_state,history):
  memory=history[-self.window:];h=torch.stack(memory,2);b,t,l,_=h.shape;q=self.q_proj(query_state).view(b,t,self.heads,self.dim);k=self.k_proj(h).view(b,t,l,self.heads,self.dim);v=self.v_proj(h).view(b,t,l,self.heads,self.dim);q=F.normalize(q.float(),dim=-1).to(q.dtype);k=F.normalize(k.float(),dim=-1).to(k.dtype);distance=torch.arange(l-1,-1,-1,device=h.device,dtype=q.dtype);score=torch.einsum('bthd,btlhd->bthl',q,k)*(self.dim**-.5)-self.distance_slope.to(q.dtype)[None,None,:,None]*distance[None,None,None,:];mixed=torch.einsum('bthl,btlhd->bthd',score.softmax(-1),v);attention=self.out_proj(mixed.reshape(b,t,-1));mean=h.mean(2);token_gate=torch.sigmoid(self.token_gate_out(F.silu(self.token_gate_in(torch.cat((query_state,mean),-1)))));return self.injection*self.scalar_gate*token_gate*attention
def match_corresponding_initialization(wrapper,seed:int)->None:
 """Bind V/O/ReZero initialization wherever the three mechanisms correspond."""
 hidden=wrapper.qwen.config.hidden_size;width=wrapper.module.v_proj.out_features if wrapper.variant in ('current','shared') else wrapper.module[0].v_proj.out_features
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed);canonical=CurrentStateAdapter(hidden,width)
 targets=[wrapper.module] if wrapper.variant in ('current','shared') else list(wrapper.module)
 for target in targets:
  target.v_proj.load_state_dict(canonical.v_proj.state_dict());target.out_proj.load_state_dict(canonical.out_proj.state_dict());target.injection.data.copy_(canonical.injection.data)
class FixedLoopQwen3(nn.Module):
 def __init__(self,qwen:nn.Module,variant:Variant,width:int=512,heads:int=4,loop_start:int=12,loop_end:int=14):
  super().__init__();self.qwen=qwen;self.variant=variant;self.loop_start=loop_start;self.loop_end=loop_end;hidden=qwen.config.hidden_size
  for p in qwen.parameters():p.requires_grad_(False)
  qwen.eval()
  if variant=='current':self.module=CurrentStateAdapter(hidden,width)
  elif variant=='shared':self.module=SharedLoopHistory(hidden,width,heads)
  elif variant=='recurtrace':self.module=nn.ModuleList(RecurTraceLayerMemory(hidden,width,heads) for _ in range(loop_end-loop_start+1))
  elif variant!='plain':raise ValueError(variant)
 def train(self,mode=True):super().train(mode);self.qwen.eval();return self
 def trainable_parameters(self):return () if self.variant=='plain' else self.module.parameters()
 def forward(self,input_ids,attention_mask=None):
  from transformers.masking_utils import create_causal_mask
  base=self.qwen.model;hidden=base.embed_tokens(input_ids);seq=input_ids.shape[1];cache_position=torch.arange(seq,device=input_ids.device);position_ids=cache_position.unsqueeze(0);mask=create_causal_mask(config=base.config,input_embeds=hidden,attention_mask=attention_mask,cache_position=cache_position,past_key_values=None,position_ids=position_ids);pos=base.rotary_emb(hidden,position_ids)
  def layer(i,x):return base.layers[i](x,attention_mask=mask,position_ids=position_ids,past_key_values=None,use_cache=False,cache_position=cache_position,position_embeddings=pos)
  for i in range(self.loop_start):hidden=layer(i,hidden)
  layer_history=[[] for _ in range(self.loop_end-self.loop_start+1)]
  for loop in range(2):
   if loop==1:
    if self.variant=='current':hidden=hidden+self.module(hidden)
    elif self.variant=='shared':hidden=hidden+self.module(hidden,[hidden])
   for j,i in enumerate(range(self.loop_start,self.loop_end+1)):
    if loop==1 and self.variant=='recurtrace':hidden=hidden+self.module[j](hidden,[layer_history[j][-1]])
    hidden=layer(i,hidden)
    if loop==0:layer_history[j].append(hidden.detach())
  for i in range(self.loop_end+1,len(base.layers)):hidden=layer(i,hidden)
  hidden=base.norm(hidden);logits=self.qwen.lm_head(hidden).float();return QwenLoopOutput(logits,hidden)
