#!/usr/bin/env python3
"""Architecture-only correctness gate for approved five-arm Step 3D."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys,uuid
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.models.qwen_loop_memory import FixedLoopQwen3,match_pair_initialization,rms_norm_no_weight
CONFIG=ROOT/'configs/013_step3_qwen_recurtrace.json'
def publish(path,value):
 data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode();tmp=path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 try:os.link(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 finally:tmp.unlink(missing_ok=True)
def main():
 import transformers;from transformers import AutoModelForCausalLM
 c=json.loads(CONFIG.read_text());commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();dirty=subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
 if dirty:raise RuntimeError(f'worktree dirty: {dirty}')
 if c['training_launch_allowed'] is not False or c['loop_layers_zero_indexed_inclusive']!=[12,14] or c['halting_head'] is not False or c['evaluation_variants']!=['plain','shared_current','shared_history','per_layer_current','recurtrace']:raise RuntimeError('approved architecture/scope mismatch')
 if transformers.__version__!=c['transformers_version'] or torch.cuda.get_device_name()!=c['hardware_for_correctness_gate']:raise RuntimeError('environment mismatch')
 base=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],dtype=torch.bfloat16,local_files_only=True).eval().cuda();ids=torch.tensor([[1,2,3,4,5,6,7,8]],device='cuda');wrappers={v:FixedLoopQwen3(base,v,c['memory_width'],c['memory_heads'],c['memory_window'],12,14,c['primary_evaluation_loop_count']).cuda().eval() for v in c['evaluation_variants']};match_pair_initialization(wrappers['shared_current'],wrappers['shared_history'],13031);match_pair_initialization(wrappers['per_layer_current'],wrappers['recurtrace'],13032)
 corresponding={}
 for left,right in c['causal_pairs']:
  a=dict(wrappers[left].named_parameters());b=dict(wrappers[right].named_parameters());a={k:v for k,v in a.items() if not k.startswith('qwen.')};b={k:v for k,v in b.items() if not k.startswith('qwen.')};corresponding[f'{left}__{right}']={'same_keys':set(a)==set(b),'bitwise':set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a),'same_parameter_count':sum(x.numel() for x in a.values())==sum(x.numel() for x in b.values())}
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):native=base(ids,use_cache=False).logits.float();one={v:wrappers[v](ids,loop_count=1) for v in c['evaluation_variants']};two={v:wrappers[v](ids,loop_count=2) for v in c['evaluation_variants']};four={v:wrappers[v](ids,loop_count=4) for v in c['evaluation_variants']}
 one_loop={v:{'logits_bitwise_native':torch.equal(x.logits,native),'finite':bool(torch.isfinite(x.logits).all())} for v,x in one.items()};t2_duplicate={'shared_pair_logits_bitwise':torch.equal(two['shared_current'].logits,two['shared_history'].logits),'per_layer_pair_logits_bitwise':torch.equal(two['per_layer_current'].logits,two['recurtrace'].logits)};t4_history_active={'shared_pair_differs':not torch.equal(four['shared_current'].logits,four['shared_history'].logits),'per_layer_pair_differs':not torch.equal(four['per_layer_current'].logits,four['recurtrace'].logits),'all_finite':all(bool(torch.isfinite(x.logits).all()) for x in four.values())}
 counts=[0]*len(base.model.layers);hooks=[layer.register_forward_hook(lambda module,args,out,i=i:counts.__setitem__(i,counts[i]+1)) for i,layer in enumerate(base.model.layers)]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):wrappers['recurtrace'](ids,loop_count=4)
 for hook in hooks:hook.remove()
 expected=[4 if 12<=i<=14 else 1 for i in range(len(base.model.layers))];layer_execution={'observed':counts,'expected':expected,'exact':counts==expected}
 layer_outputs=[[] for _ in range(3)];memory_calls=[[] for _ in range(3)];trace_hooks=[]
 for j,layer_index in enumerate(range(12,15)):
  trace_hooks.append(base.model.layers[layer_index].register_forward_hook(lambda module,args,out,j=j:layer_outputs[j].append(out.detach().clone())))
  trace_hooks.append(wrappers['recurtrace'].module[j].register_forward_pre_hook(lambda module,args,j=j:memory_calls[j].append((args[0].detach().clone(),[x.detach().clone() for x in args[1]]))))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):wrappers['recurtrace'](ids,loop_count=4)
 for hook in trace_hooks:hook.remove()
 same_layer_trace={'call_counts':[len(x) for x in memory_calls],'exact_query_and_chronological_history':all(len(memory_calls[j])==3 and all(torch.equal(memory_calls[j][call][0],layer_outputs[j][call]) and len(memory_calls[j][call][1])==call+1 and all(torch.equal(memory_calls[j][call][1][history_index],layer_outputs[j][history_index]) for history_index in range(call+1)) for call in range(3)) for j in range(3))}
 injection_model=wrappers['shared_current'];injection_model.input_injection.data.fill_(.5);injection_model.module.scalar_gate.data.zero_();encoder=[];first_inputs=[];block_outputs=[];injection_hooks=[base.model.layers[11].register_forward_hook(lambda module,args,out:encoder.append(out.detach().clone())),base.model.layers[12].register_forward_pre_hook(lambda module,args:first_inputs.append(args[0].detach().clone())),base.model.layers[14].register_forward_hook(lambda module,args,out:block_outputs.append(out.detach().clone()))]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):injection_model(ids,loop_count=2)
 for hook in injection_hooks:hook.remove()
 expected_second=block_outputs[0]+injection_model.input_injection*rms_norm_no_weight(encoder[0],injection_model.eps);reinjection={'encoder_to_first_loop_exact':torch.equal(first_inputs[0],encoder[0]),'after_first_loop_only_exact':torch.equal(first_inputs[1],expected_second),'call_counts_exact':len(encoder)==1 and len(first_inputs)==2 and len(block_outputs)==2};injection_model.input_injection.data.zero_();injection_model.module.scalar_gate.data.fill_(1)
 gradient={};target=torch.tensor([9],device='cuda')
 for variant in c['variants']:
  model=wrappers[variant].train();model.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=torch.nn.functional.cross_entropy(model(ids,loop_count=4).logits[:,-1],target)
  loss.backward();named={k:p for k,p in model.named_parameters() if not k.startswith('qwen.')};always_names=('input_injection','v_proj','out_proj','scalar_gate','token_gate_out');selection_names=('q_proj','k_proj','distance_slope');always={g:any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for n,p in named.items() if g in n) for g in always_names};selection={g:any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for n,p in named.items() if g in n) for g in selection_names};token_gate_in_initial=any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for n,p in named.items() if 'token_gate_in' in n);expects_selection=variant in ('shared_history','recurtrace');initial_pass=all(always.values()) and not token_gate_in_initial and (all(selection.values()) if expects_selection else not any(selection.values()));model.zero_grad(set_to_none=True)
  modules=[model.module] if variant.startswith('shared_') else list(model.module)
  with torch.no_grad():
   for index,module in enumerate(modules):
    generator=torch.Generator(device=module.token_gate_out.weight.device);generator.manual_seed(1700+index);module.token_gate_out.weight.copy_(torch.randn(module.token_gate_out.weight.shape,generator=generator,device=module.token_gate_out.weight.device,dtype=module.token_gate_out.weight.dtype)*.01)
  with torch.autocast('cuda',dtype=torch.bfloat16):staged_loss=torch.nn.functional.cross_entropy(model(ids,loop_count=4).logits[:,-1],target)
  staged_loss.backward();staged_token_gate_in=any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for n,p in named.items() if 'token_gate_in' in n);gradient[variant]={'loss_finite':bool(torch.isfinite(loss) and torch.isfinite(staged_loss)),'initial_always_active':always,'initial_selection_groups':selection,'initial_token_gate_in_expected_zero':not token_gate_in_initial,'staged_token_gate_in_active':staged_token_gate_in,'expects_selection':expects_selection,'pass':initial_pass and staged_token_gate_in,'base_gradients':sum(p.grad is not None for p in base.parameters()),'trainable_parameters':sum(p.numel() for p in model.trainable_parameters())};model.zero_grad(set_to_none=True)
 no_halting=all('halt' not in n.lower() for v in c['evaluation_variants'] for n,_ in wrappers[v].named_parameters());base_frozen=all(not p.requires_grad for p in base.parameters());result={'protocol':c['protocol']+'-architecture-correctness-gate','status':'pass','scope':'implementation-only; training remains blocked by authoritative asset and compute gates','one_loop_native_parity':one_loop,'corresponding_pair_initialization':corresponding,'two_loop_singleton_history_equivalence':t2_duplicate,'four_loop_history_activation':t4_history_active,'four_loop_layer_execution':layer_execution,'same_layer_history_trace':same_layer_trace,'input_reinjection_trace':reinjection,'gradient_connectivity_at_four_loops':gradient,'base_frozen':base_frozen,'halting_parameters_absent':no_halting,'model_revision':c['model_revision'],'transformers_version':transformers.__version__,'hardware':torch.cuda.get_device_name(),'git_commit':commit,'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),'training_launch_allowed':False,'do_not_start_step_4':True}
 if not all(all(v.values()) for v in corresponding.values()) or not all(all(v.values()) for v in one_loop.values()) or not all(t2_duplicate.values()) or not all(t4_history_active.values()) or not layer_execution['exact'] or not same_layer_trace['exact_query_and_chronological_history'] or same_layer_trace['call_counts']!=[3,3,3] or not all(reinjection.values()) or not all(v['loss_finite'] and v['pass'] and v['base_gradients']==0 for v in gradient.values()) or not base_frozen or not no_halting:raise RuntimeError(result)
 root=Path(c['output_root']);root.mkdir(exist_ok=True);path=root/'architecture_correctness_gate.json'
 if path.exists():raise FileExistsError(path)
 publish(path,result);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
