import torch
import torch.nn.functional as F
from src.models.qwen_loop_memory import LoopMemoryAttention,alibi_slopes,rms_norm_no_weight

def test_rms_norm_is_per_head_rms_not_l2():
 x=torch.tensor([[[[3.,4.]]]]);y=rms_norm_no_weight(x,0.0);assert torch.allclose(y.pow(2).mean(-1),torch.ones_like(y[...,0]))

def test_alibi_slopes_four_heads():
 assert torch.equal(alibi_slopes(4),torch.tensor([.25,.0625,.015625,.00390625]))

def test_lma_empty_memory_is_exact_zero():
 module=LoopMemoryAttention(8,4,2);x=torch.randn(2,3,8);assert torch.equal(module(x,[]),torch.zeros_like(x))

def test_lma_same_position_shape_and_window():
 module=LoopMemoryAttention(8,4,2,window=3);x=torch.randn(1,5,8);history=[torch.randn_like(x) for _ in range(5)];assert module(x,history).shape==x.shape

def test_lma_active_paths_and_singleton_qk_inactivity():
 torch.manual_seed(3);module=LoopMemoryAttention(8,4,2);torch.nn.init.normal_(module.token_gate_out.weight,std=.01);q=torch.randn(1,3,8);loss=module(q,[q]).sum();loss.backward();named=dict(module.named_parameters())
 for prefix in ('v_proj','out_proj','scalar_gate','token_gate_in','token_gate_out'):
  assert any(p.grad is not None and torch.count_nonzero(p.grad) for name,p in named.items() if prefix in name)
 for prefix in ('q_proj','k_proj','distance_slope'):
  assert all(p.grad is None or not torch.count_nonzero(p.grad) for name,p in named.items() if prefix in name)

def test_lma_matches_independent_equation_with_order_window_distance_and_gates():
 torch.manual_seed(11);module=LoopMemoryAttention(6,4,2,window=3,eps=1e-6);query=torch.randn(1,2,6);history=[torch.randn_like(query) for _ in range(4)];actual=module(query,history);memory=history[-3:];stack=torch.stack(memory,2);q=F.linear(query,module.q_proj.weight).view(1,2,2,2);k=F.linear(stack,module.k_proj.weight).view(1,2,3,2,2);v=F.linear(stack,module.v_proj.weight).view(1,2,3,2,2);q=(q.float()*torch.rsqrt(q.float().square().mean(-1,keepdim=True)+module.eps)).to(q.dtype);k=(k.float()*torch.rsqrt(k.float().square().mean(-1,keepdim=True)+module.eps)).to(k.dtype);distance=torch.tensor([3.,2.,1.]);score=torch.einsum('bthd,btmhd->bthm',q,k)/(2**.5)-module.distance_slope[None,None,:,None]*distance[None,None,None,:];mixed=torch.einsum('bthm,btmhd->bthd',score.softmax(-1),v).reshape(1,2,4);attention=F.linear(mixed,module.out_proj.weight);mean=stack.mean(2);gate=torch.sigmoid(F.linear(F.silu(F.linear(torch.cat((query,mean),-1),module.token_gate_in.weight,module.token_gate_in.bias)),module.token_gate_out.weight,module.token_gate_out.bias));expected=module.scalar_gate*gate*attention
 assert torch.allclose(actual,expected,rtol=1e-6,atol=1e-6)

def test_multiple_history_slots_activate_query_key_and_distance():
 torch.manual_seed(5);module=LoopMemoryAttention(8,4,2);q=torch.randn(1,3,8);history=[torch.randn_like(q),torch.randn_like(q)];module(q,history).sum().backward();named=dict(module.named_parameters())
 for prefix in ('q_proj','k_proj','distance_slope'):
  assert any(p.grad is not None and torch.count_nonzero(p.grad) for name,p in named.items() if prefix in name)
