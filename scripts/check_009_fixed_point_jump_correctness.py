#!/usr/bin/env python3
"""Pre-training correctness gate for functional fixed-point jump."""
from __future__ import annotations
import hashlib,json,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.models.fixed_point_jump_huginn import FixedPointJumpHuginn
from src.training.latent_history import answer_cross_entropy,encode_supervised_example,materialize_h0_schedule
from scripts.train_009_fixed_point_jump import fp_loss,tensor_state_sha
CONFIG=ROOT/'configs/009_fixed_point_jump.json';OUTPUT=Path('/workspace/fixed_point_jump/correctness_gate.json')
def sha(module):
 d=hashlib.sha256()
 for n,v in module.state_dict().items():d.update(n.encode());d.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return d.hexdigest()
def main():
 from datasets import load_dataset;from transformers import AutoModelForCausalLM,AutoTokenizer
 c=json.loads(CONFIG.read_text());OUTPUT.parent.mkdir(exist_ok=True)
 if OUTPUT.exists():raise FileExistsError(OUTPUT)
 # Independent literal alignment check: positions 1 and 2 must predict IDs 33 and 44.
 synthetic_logits=torch.randn(1,4,64,device='cuda');synthetic_ids=torch.tensor([[11,22,33,44]],device='cuda');observed,observed_count=answer_cross_entropy(synthetic_logits,synthetic_ids,2);literal=torch.nn.functional.cross_entropy(torch.stack((synthetic_logits[0,1],synthetic_logits[0,2])),torch.tensor([33,44],device='cuda'))
 if observed_count!=2 or not torch.equal(observed,literal):raise RuntimeError('independent first-answer-token CE alignment failed')
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);ds=load_dataset(c['dataset_id'],c['dataset_config'],split='train',revision=c['dataset_revision']);huginn=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();torch.manual_seed(c['predictor_initialization_seed']);wrapper=FixedPointJumpHuginn(huginn,c['bottleneck_size']).cuda();initialization_sha=tensor_state_sha(wrapper.predictor);item=encode_supervised_example(tok,ds[0]['question'],ds[0]['answer'],c['system_instruction']);ids=item.input_ids[None].cuda();schedule=materialize_h0_schedule(huginn,device=ids.device,example_id=0,base_seed=c['h0_base_seed']);h0=schedule[:,:ids.shape[1]]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):inference=wrapper(ids,h0);training=wrapper(ids,h0,compute_fixed_point=True)
 initialization={'h_star_equals_h0_bitwise':torch.equal(inference.latent_states,h0),'output_projection_zero':int(torch.count_nonzero(wrapper.predictor.out_proj.weight))==0 and int(torch.count_nonzero(wrapper.predictor.out_proj.bias))==0,'inference_core_steps':inference.core_steps_executed,'consistency_core_steps':training.core_steps_executed,'fixed_point_image_finite':bool(torch.isfinite(training.fixed_point_image).all()),'predictor_parameters':sum(p.numel() for p in wrapper.predictor.parameters())}
 if not all((initialization['h_star_equals_h0_bitwise'],initialization['output_projection_zero'],initialization['inference_core_steps']==0,initialization['consistency_core_steps']==1,initialization['fixed_point_image_finite'])):raise RuntimeError(initialization)
 before=sha(huginn);wrapper.train()
 with torch.autocast('cuda',dtype=torch.bfloat16):out=wrapper(ids,h0,compute_fixed_point=True);task,count=answer_cross_entropy(out.logits,ids,item.answer_start);fixed=fp_loss(out,item.answer_start,count,c['fixed_point_epsilon']);loss=task+c['fixed_point_weight']*fixed
 loss.backward();grads={n:p.grad is not None and bool(torch.isfinite(p.grad).all()) for n,p in wrapper.predictor.named_parameters()};output_grad=float(wrapper.predictor.out_proj.weight.grad.float().norm());frozen_grads=sum(p.grad is not None for p in huginn.parameters());opt=torch.optim.AdamW(wrapper.trainable_parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay']);opt.step();opt.zero_grad(set_to_none=True)
 gradient={'task_loss':float(task),'fixed_point_loss':float(fixed),'total_loss':float(loss),'target_tokens':count,'all_predictor_gradients_finite_when_present':all(grads.values()),'output_gradient_nonzero':output_grad>0,'frozen_huginn_gradient_tensors':frozen_grads,'huginn_unchanged':before==sha(huginn)}
 if not all((gradient['all_predictor_gradients_finite_when_present'],gradient['output_gradient_nonzero'],gradient['frozen_huginn_gradient_tensors']==0,gradient['huginn_unchanged'])):raise RuntimeError(gradient)
 wrapper.eval();checks=[]
 for example_id in (2250,2251,2252):
  encoded=encode_supervised_example(tok,ds[example_id]['question'],ds[example_id]['answer'],c['system_instruction']);prompt=encoded.input_ids[:encoded.answer_start][None].cuda();example_schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=example_id,base_seed=c['h0_base_seed']);cache=None;current=prompt;position=None;prefix=prompt;cached_tokens=[];full_tokens=[]
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for step in range(8):
    states=example_schedule[:,:current.shape[1]] if position is None else example_schedule[:,position:position+1];cp=None if position is None else torch.tensor([position],device='cuda');cached=wrapper(current,states,past_key_values=cache,use_cache=True,cache_position=cp);cache=cached.past_key_values;full=wrapper(prefix,example_schedule[:,:prefix.shape[1]]);delta=float((cached.logits[:,-1]-full.logits[:,-1]).abs().max());cached_token=int(cached.logits[:,-1].argmax());full_token=int(full.logits[:,-1].argmax());cached_tokens.append(cached_token);full_tokens.append(full_token)
    if cached_token!=full_token or delta>0.25 or cached.core_steps_executed or full.core_steps_executed:raise RuntimeError({'example_id':example_id,'step':step,'delta':delta,'cached_token':cached_token,'full_token':full_token})
    token=torch.tensor([[cached_token]],device='cuda');prefix=torch.cat((prefix,token),1);position=prefix.shape[1]-1;current=token
  checks.append({'example_id':example_id,'steps':8,'generated_ids_exact':cached_tokens==full_tokens,'generated_token_ids':cached_tokens,'max_checked_logit_abs_error_tolerance':0.25})
 commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();result={'protocol':'functional-fixed-point-jump-correctness-gate-v1','status':'pass','git_commit':commit,'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),'predictor_initialization_sha256':initialization_sha,'independent_ce_alignment':{'answer_start':2,'target_ids':[33,44],'target_count':observed_count,'exact_loss_match':True},'initialization':initialization,'gradient_and_freezing':gradient,'cache_vs_full_prefix':checks,'no_h16_or_h64_target':True,'inference_recurrent_steps':0,'full_vocabulary_logits_persisted':False};OUTPUT.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');OUTPUT.chmod(0o444);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
