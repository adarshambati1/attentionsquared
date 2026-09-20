import torch
from src.models.qwen_loop_memory import CurrentStateAdapter,RecurTraceLayerMemory,SharedLoopHistory

def test_current_and_one_state_shared_are_exact_when_corresponding_weights_match():
 torch.manual_seed(1);current=CurrentStateAdapter(8,4);shared=SharedLoopHistory(8,4,2);torch.nn.init.normal_(current.out_proj.weight);shared.v_proj.load_state_dict(current.v_proj.state_dict());shared.out_proj.load_state_dict(current.out_proj.state_dict());current.injection.data.fill_(1);shared.injection.data.fill_(1);x=torch.randn(2,3,8)
 assert torch.equal(current(x),shared(x,[x]))

def test_recurtrace_rezero_is_exact_zero_but_has_gradient():
 torch.manual_seed(2);module=RecurTraceLayerMemory(8,4,2);x=torch.randn(2,3,8);out=module(x,[x]);assert torch.equal(out,torch.zeros_like(out));out.sum().backward();assert module.injection.grad is not None and module.injection.grad!=0

def test_recurtrace_same_position_shapes_and_window():
 module=RecurTraceLayerMemory(8,4,2,window=3);x=torch.randn(1,5,8);history=[torch.randn_like(x) for _ in range(5)];assert module(x,history).shape==x.shape

def test_recurtrace_uses_live_query_not_only_history():
 torch.manual_seed(4);module=RecurTraceLayerMemory(8,4,2);module.injection.data.fill_(1);history=[torch.randn(1,3,8),torch.randn(1,3,8)];q=torch.randn(1,3,8)
 assert not torch.equal(module(q,history),module(q+0.25,history))
