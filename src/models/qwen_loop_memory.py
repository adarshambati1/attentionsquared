"""Paper-faithful fixed-budget Qwen3 loop controls for Step 3D."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn
import torch.nn.functional as F
Variant=Literal['plain','shared_current','shared_history','per_layer_current','recurtrace']
@dataclass
class QwenLoopOutput:
 logits:torch.Tensor
 hidden_states:torch.Tensor

def rms_norm_no_weight(x:torch.Tensor,eps:float)->torch.Tensor:
 dtype=x.dtype;return (x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+eps)).to(dtype)

def alibi_slopes(heads:int)->torch.Tensor:
 if heads<=0 or heads&(heads-1):raise ValueError('LMA heads must be a positive power of two')
 return torch.tensor([2**(-8*(i+1)/heads) for i in range(heads)])

class LoopMemoryAttention(nn.Module):
 """Same-position attention over one layer's prior-loop states."""
 def __init__(self,hidden:int,width:int,heads:int,window:int=3,eps:float=1e-6):
  super().__init__();assert width%heads==0;self.heads=heads;self.dim=width//heads;self.window=window;self.eps=eps
  self.q_proj=nn.Linear(hidden,width,bias=False);self.k_proj=nn.Linear(hidden,width,bias=False);self.v_proj=nn.Linear(hidden,width,bias=False);self.out_proj=nn.Linear(width,hidden,bias=False)
  self.scalar_gate=nn.Parameter(torch.ones(()));self.distance_slope=nn.Parameter(alibi_slopes(heads));self.token_gate_in=nn.Linear(2*hidden,hidden);self.token_gate_out=nn.Linear(hidden,1);nn.init.zeros_(self.token_gate_out.weight);nn.init.constant_(self.token_gate_out.bias,-3.0)
 def forward(self,query_state:torch.Tensor,history:list[torch.Tensor])->torch.Tensor:
  if not history:return torch.zeros_like(query_state)
  memory=history[-self.window:];stack=torch.stack(memory,2);b,t,m,_=stack.shape;q=self.q_proj(query_state).view(b,t,self.heads,self.dim);k=self.k_proj(stack).view(b,t,m,self.heads,self.dim);v=self.v_proj(stack).view(b,t,m,self.heads,self.dim);q=rms_norm_no_weight(q,self.eps);k=rms_norm_no_weight(k,self.eps);distance=torch.arange(m,0,-1,device=stack.device,dtype=q.dtype);score=torch.einsum('bthd,btmhd->bthm',q,k)*(self.dim**-.5)-self.distance_slope.to(q.dtype)[None,None,:,None]*distance[None,None,None,:];mixed=torch.einsum('bthm,btmhd->bthd',score.softmax(-1),v);attention=self.out_proj(mixed.reshape(b,t,-1));mean=stack.mean(2);token_gate=torch.sigmoid(self.token_gate_out(F.silu(self.token_gate_in(torch.cat((query_state,mean),-1)))));return self.scalar_gate*token_gate*attention

class FixedLoopQwen3(nn.Module):
 def __init__(self,qwen:nn.Module,variant:Variant,width:int=512,heads:int=4,window:int=3,loop_start:int=12,loop_end:int=14,loop_count:int=2):
  super().__init__();self.qwen=qwen;self.variant=variant;self.loop_start=loop_start;self.loop_end=loop_end;self.loop_count=loop_count;self.eps=float(qwen.config.rms_norm_eps);hidden=qwen.config.hidden_size
  for p in qwen.parameters():p.requires_grad_(False)
  qwen.eval()
  if variant in ('shared_current','shared_history'):self.module=LoopMemoryAttention(hidden,width,heads,window,self.eps)
  elif variant in ('per_layer_current','recurtrace'):self.module=nn.ModuleList(LoopMemoryAttention(hidden,width,heads,window,self.eps) for _ in range(loop_end-loop_start+1))
  elif variant!='plain':raise ValueError(variant)
  if variant!='plain':self.input_injection=nn.Parameter(torch.zeros(()))
 def train(self,mode=True):super().train(mode);self.qwen.eval();return self
 def trainable_parameters(self):return () if self.variant=='plain' else (p for name,p in self.named_parameters() if not name.startswith('qwen.'))
 def forward(self,input_ids,attention_mask=None,*,loop_count:int|None=None):
  loops=self.loop_count if loop_count is None else loop_count
  if loops<1:raise ValueError('loop_count must be positive')
  from transformers.masking_utils import create_causal_mask
  base=self.qwen.model;hidden=base.embed_tokens(input_ids);seq=input_ids.shape[1];cache_position=torch.arange(seq,device=input_ids.device);position_ids=cache_position.unsqueeze(0);mask=create_causal_mask(config=base.config,input_embeds=hidden,attention_mask=attention_mask,cache_position=cache_position,past_key_values=None,position_ids=position_ids);pos=base.rotary_emb(hidden,position_ids)
  def layer(i,x):return base.layers[i](x,attention_mask=mask,position_ids=position_ids,past_key_values=None,use_cache=False,cache_position=cache_position,position_embeddings=pos)
  for i in range(self.loop_start):hidden=layer(i,hidden)
  block_input=hidden;block_history=[];layer_history=[[] for _ in range(self.loop_end-self.loop_start+1)]
  for loop in range(loops):
   if loop>0 and self.variant!='plain':hidden=hidden+self.input_injection*rms_norm_no_weight(block_input,self.eps)
   if loop>0 and self.variant in ('shared_current','shared_history'):
    query=block_history[-1];memory=[query] if self.variant=='shared_current' else block_history;hidden=hidden+self.module(query,memory)
   for j,i in enumerate(range(self.loop_start,self.loop_end+1)):
    if loop>0 and self.variant in ('per_layer_current','recurtrace'):
     query=layer_history[j][-1];memory=[query] if self.variant=='per_layer_current' else layer_history[j];hidden=hidden+self.module[j](query,memory)
    hidden=layer(i,hidden);layer_history[j].append(hidden)
   block_history.append(hidden)
  for i in range(self.loop_end+1,len(base.layers)):hidden=layer(i,hidden)
  hidden=base.norm(hidden);return QwenLoopOutput(self.qwen.lm_head(hidden).float(),hidden)

def match_pair_initialization(left:FixedLoopQwen3,right:FixedLoopQwen3,seed:int)->None:
 """Initialize corresponding B/C or D/E modules exactly alike."""
 if (left.variant,right.variant) not in (('shared_current','shared_history'),('per_layer_current','recurtrace')):raise ValueError('not an approved causal pair')
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed);left.input_injection.data.zero_();right.input_injection.data.copy_(left.input_injection.data)
  left_modules=[left.module] if left.variant=='shared_current' else list(left.module);right_modules=[right.module] if right.variant=='shared_history' else list(right.module)
  for index,(a,b) in enumerate(zip(left_modules,right_modules)):
   torch.manual_seed(seed+index);template=LoopMemoryAttention(left.qwen.config.hidden_size,a.q_proj.out_features,a.heads,a.window,a.eps);a.load_state_dict(template.state_dict());b.load_state_dict(template.state_dict())
