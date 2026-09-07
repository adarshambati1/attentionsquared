"""Minimal shared-round Attention² operator for Experiment 004A."""
from __future__ import annotations
import torch
from torch import nn

class Attention2Lite(nn.Module):
    def __init__(self, hidden: int, depth: int = 16, heads: int = 16, rounds: int = 4, mlp_ratio: int = 1):
        super().__init__(); assert hidden % heads == 0
        self.depth,self.rounds,self.hidden=depth,rounds,hidden
        self.init_proj=nn.Linear(2*hidden,hidden)
        self.depth_embedding=nn.Parameter(torch.zeros(depth,hidden)); nn.init.normal_(self.depth_embedding,std=0.02)
        self.token_norm=nn.LayerNorm(hidden); self.token_attn=nn.MultiheadAttention(hidden,heads,batch_first=True)
        self.depth_norm=nn.LayerNorm(hidden); self.depth_attn=nn.MultiheadAttention(hidden,heads,batch_first=True)
        mid=hidden*mlp_ratio; self.mlp_norm=nn.LayerNorm(hidden); self.mlp=nn.Sequential(nn.Linear(hidden,mid),nn.GELU(),nn.Linear(mid,hidden))
        self.reset_parameters()
    def reset_parameters(self): nn.init.xavier_uniform_(self.init_proj.weight); nn.init.zeros_(self.init_proj.bias)
    def initialize(self,h0,x):
        base=self.init_proj(torch.cat([h0,x],dim=-1)); return base[:,None,:,:]+self.depth_embedding[None,:,None,:]
    def operator(self,z,token_mask=None,return_attention=False):
        b,d,t,h=z.shape; attns={}
        q=self.token_norm(z).reshape(b*d,t,h); causal=torch.triu(torch.ones(t,t,device=z.device,dtype=torch.bool),diagonal=1)
        tok,_=self.token_attn(q,q,q,attn_mask=causal,key_padding_mask=None,need_weights=False); z=z+tok.reshape(b,d,t,h)
        q=self.depth_norm(z).transpose(1,2).reshape(b*t,d,h); dep,w=self.depth_attn(q,q,q,need_weights=return_attention,average_attn_weights=False); z=z+dep.reshape(b,t,d,h).transpose(1,2)
        z=z+self.mlp(self.mlp_norm(z))
        if return_attention: attns['depth']=w
        return z,attns
    def forward(self,h0,x,return_attention=False):
        z=self.initialize(h0,x); all_z=[]; attns=[]
        for _ in range(self.rounds): z,a=self.operator(z,return_attention=return_attention); all_z.append(z); attns.append(a)
        return z,all_z,attns
