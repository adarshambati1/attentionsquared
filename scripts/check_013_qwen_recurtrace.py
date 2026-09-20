#!/usr/bin/env python3
"""Correctness gate for the fixed-two-loop Qwen3-1.7B mechanism comparison."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.models.qwen_loop_memory import FixedLoopQwen3,match_corresponding_initialization
CONFIG=ROOT/'configs/013_step3_qwen_recurtrace.json'
def publish(p,v):
 data=(json.dumps(v,indent=2,sort_keys=True)+'\n').encode();tmp=p.with_name(f'.{p.name}.{os.getpid()}.tmp')
 with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 try:os.link(tmp,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 finally:tmp.unlink(missing_ok=True)
def initialize(base,variant,c):
 torch.manual_seed(c['module_initialization_seed']);w=FixedLoopQwen3(base,variant,c['memory_width'],c['memory_heads'],*c['loop_layers_zero_indexed_inclusive']).cuda().eval()
 if variant!='plain':match_corresponding_initialization(w,c['module_initialization_seed'])
 return w
def main():
 from datasets import load_dataset;import transformers;from transformers import AutoModelForCausalLM,AutoTokenizer
 c=json.loads(CONFIG.read_text());
 if c['loop_layers_zero_indexed_inclusive']!=[12,14] or c['fixed_loop_count']!=2 or c['do_not_train_halting_head'] is not True or c['do_not_start_step_4'] is not True:raise RuntimeError('frozen Step-3D architecture/scope mismatch')
 commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();dirty=subprocess.run(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True,capture_output=True,check=True).stdout
 if dirty:raise RuntimeError(f'tracked worktree dirty: {dirty}')
 if transformers.__version__!=c['transformers_version'] or torch.cuda.get_device_name()!=c['hardware']:raise RuntimeError({'transformers':transformers.__version__,'hardware':torch.cuda.get_device_name()})
 root=Path(c['output_root']);root.mkdir(exist_ok=True);path=root/'correctness_gate.json'
 if path.exists():raise FileExistsError(path)
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);data=load_dataset(c['dataset_id'],split='train',revision=c['dataset_revision'],trust_remote_code=True);base=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],dtype=torch.bfloat16,local_files_only=True).eval().cuda();item=data[0];prompt=tok.apply_chat_template([{'role':'system','content':c['system_instruction']},{'role':'user','content':item['Problem']+'\nOptions: '+item['options']}],tokenize=True,add_generation_prompt=True,enable_thinking=False);ids=torch.tensor([prompt],device='cuda');wrappers={v:initialize(base,v,c) for v in c['evaluation_variants']};counts=[0]*len(base.model.layers);hooks=[layer.register_forward_hook(lambda module,args,out,i=i:counts.__setitem__(i,counts[i]+1)) for i,layer in enumerate(base.model.layers)]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):plain=wrappers['plain'](ids)
 for h in hooks:h.remove()
 start,end=c['loop_layers_zero_indexed_inclusive'];expected=[2 if start<=i<=end else 1 for i in range(len(base.model.layers))];layer_execution={'observed':counts,'expected':expected,'exact':counts==expected};initial={}
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for v in ('current','shared','recurtrace'):
   out=wrappers[v](ids);initial[v]={'logits_bitwise_plain':torch.equal(out.logits,plain.logits),'hidden_bitwise_plain':torch.equal(out.hidden_states,plain.hidden_states),'finite':bool(torch.isfinite(out.logits).all())}
 corresponding={'current_shared_v':torch.equal(wrappers['current'].module.v_proj.weight,wrappers['shared'].module.v_proj.weight),'current_shared_out':torch.equal(wrappers['current'].module.out_proj.weight,wrappers['shared'].module.out_proj.weight),'current_shared_injection':torch.equal(wrappers['current'].module.injection,wrappers['shared'].module.injection),'recurtrace_v_all':all(torch.equal(wrappers['current'].module.v_proj.weight,m.v_proj.weight) for m in wrappers['recurtrace'].module),'recurtrace_out_all':all(torch.equal(wrappers['current'].module.out_proj.weight,m.out_proj.weight) for m in wrappers['recurtrace'].module),'recurtrace_injection_all':all(torch.equal(wrappers['current'].module.injection,m.injection) for m in wrappers['recurtrace'].module)};gradient={}
 target=torch.tensor([tok(item['correct'],add_special_tokens=False).input_ids[0]],device='cuda')
 for v in ('current','shared','recurtrace'):
  w=wrappers[v].train();w.qwen.eval();w.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):out=w(ids);loss=torch.nn.functional.cross_entropy(out.logits[:,-1],target)
  loss.backward();params=list(w.trainable_parameters());nonzero=sum(int(p.grad is not None and bool(torch.count_nonzero(p.grad))) for p in params);gradient[v]={'loss_finite':bool(torch.isfinite(loss)),'parameters_with_nonzero_gradient':nonzero,'base_parameters_with_gradient':sum(p.grad is not None for p in base.parameters()),'trainable_parameters':sum(p.numel() for p in params)}
  w.zero_grad(set_to_none=True)
 opts={v:torch.optim.AdamW(list(wrappers[v].trainable_parameters()),lr=c['learning_rate'],weight_decay=c['weight_decay']) for v in ('current','shared')}
 for v in ('current','shared'):
  w=wrappers[v];w.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=torch.nn.functional.cross_entropy(w(ids).logits[:,-1],target)
  loss.backward();opts[v].step()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):post_current=wrappers['current'](ids);post_shared=wrappers['shared'](ids)
 post_update_equivalence={'logits_bitwise':torch.equal(post_current.logits,post_shared.logits),'hidden_bitwise':torch.equal(post_current.hidden_states,post_shared.hidden_states),'v_proj':torch.equal(wrappers['current'].module.v_proj.weight,wrappers['shared'].module.v_proj.weight),'out_proj':torch.equal(wrappers['current'].module.out_proj.weight,wrappers['shared'].module.out_proj.weight),'injection':torch.equal(wrappers['current'].module.injection,wrappers['shared'].module.injection)}
 opened_path_gradients={}
 for v in ('current','shared','recurtrace'):
  w=wrappers[v];modules=[w.module] if v!='recurtrace' else list(w.module)
  for m in modules:
   m.injection.data.fill_(1)
   if v=='recurtrace':torch.manual_seed(17);torch.nn.init.normal_(m.token_gate_out.weight,std=.01)
  w.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=torch.nn.functional.cross_entropy(w(ids).logits[:,-1],target)
  loss.backward();named=dict(w.module.named_parameters());active_prefixes=('v_proj','out_proj','injection') if v in ('current','shared') else ('v_proj','out_proj','injection','scalar_gate','token_gate_in','token_gate_out');inactive_prefixes=('q_proj','k_proj') if v=='shared' else (('q_proj','k_proj','distance_slope') if v=='recurtrace' else ())
  active={prefix:any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for name,p in named.items() if prefix in name) for prefix in active_prefixes};inactive={prefix:all(p.grad is None or not bool(torch.count_nonzero(p.grad)) for name,p in named.items() if prefix in name) for prefix in inactive_prefixes};opened_path_gradients[v]={'active':active,'singleton_history_inactive':inactive,'pass':all(active.values()) and all(inactive.values())}
  w.zero_grad(set_to_none=True)
 for v in ('current','shared'):wrappers[v].module.injection.data.zero_()
 base_frozen=all(not p.requires_grad for p in base.parameters());result={'protocol':c['protocol']+'-correctness-gate','status':'pass','hardware':torch.cuda.get_device_name(),'transformers_version':transformers.__version__,'model_revision':c['model_revision'],'dataset_revision':c['dataset_revision'],'layer_execution':layer_execution,'initial_exact_plain_semantics':initial,'corresponding_initialization':corresponding,'post_representative_update_current_shared_equivalence':post_update_equivalence,'gradient_checks':gradient,'opened_rezero_path_gradient_checks':opened_path_gradients,'base_frozen':base_frozen,'fixed_loop_count':2,'halting_head':False,'recurtrace_two_loop_structural_note':'At exactly two loops each per-layer reader has one prior-loop memory item, so Q/K selection and distance bias are algebraically inactive; the tested distinction is per-layer placement with live-query token gating and V/O memory.','shared_current_two_loop_structural_note':'With one completed whole-loop state at the only injection point, shared whole-block attention reduces exactly to the current-state V/O adapter; Q/K receive no gradient. This is entailed by the requested fixed-two-loop protocol.','git_commit':commit,'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest()}
 if not layer_execution['exact'] or not all(all(x.values()) for x in initial.values()) or not all(corresponding.values()) or not all(post_update_equivalence.values()) or not all(x['pass'] for x in opened_path_gradients.values()) or not all(x['loss_finite'] and x['parameters_with_nonzero_gradient']>0 and x['base_parameters_with_gradient']==0 for x in gradient.values()) or not base_frozen:raise RuntimeError(result)
 publish(path,result);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
