import hashlib,json
from pathlib import Path
from types import MethodType
import numpy as np,torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path('/workspace/attentionsquared')
MODEL='tomg-group-umd/huginn-0125'; REV='bb6621b65e90b6a4b9b29ef88dc83866d450470c'
SYS='You are a helpful assistant that can assist users with mathematical reasoning.'
rows=[json.loads(x) for x in (ROOT/'results/004_functional/valid_end_manifest.jsonl').read_text().splitlines()[:8]]
tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,local_files_only=True)
ds=load_dataset('openai/gsm8k','main',split='train',revision='740312add88f781978c0658806c59bc2815b9866')
model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda().requires_grad_(False)

def seed_for(i): return (((3000*(1<<24)+i)*(1<<8)+0)*(1<<7)+0)
def schedule(i):
 cpu=torch.random.get_rng_state().clone(); cuda=[x.clone() for x in torch.cuda.get_rng_state_all()]
 template=torch.empty((1,2048,5280),device='cuda',dtype=torch.bfloat16)
 with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())),enabled=True):
  torch.manual_seed(seed_for(i)); out=model.initialize_state(template,scale=1.0)
 restored=torch.equal(cpu,torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
 digest=hashlib.sha256(out.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
 return out,digest,restored

def run(ids,h0):
 calls=0; captured={}; original=model.core_block_forward
 def wrapped(this,state,recurrent,*args,**kwargs):
  nonlocal calls
  calls+=1
  if calls==1: captured['x']=recurrent.detach().clone()
  result=original(state,recurrent,*args,**kwargs)
  if calls==16: captured['pre']=(result[0] if isinstance(result,tuple) else result).detach().clone()
  return result
 model.core_block_forward=MethodType(wrapped,model)
 try:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   live=model(input_ids=ids,attention_mask=torch.ones_like(ids,dtype=torch.bool),input_states=h0,num_steps=16,use_cache=False,return_dict=True,output_details={'return_logits':True,'return_latents':True,'return_head':False,'return_stats':False})
   normalized=model.transformer.ln_f(captured['pre'])
   state=normalized; bi=torch.tensor(0,device='cpu',dtype=torch.long); freqs=model.freqs_cis[:,:ids.shape[1]]
   for block in model.transformer.coda:
    bi-=1; state=block(state,freqs,bi,None,None)
   decomposed=model.lm_head(model.transformer.ln_f(state)).float()
 finally: model.core_block_forward=original
 assert calls==16 and torch.equal(normalized,live.latent_states) and torch.equal(decomposed,live.logits.float())
 return {'x':captured['x'],'h16':normalized,'logits':live.logits.float()},calls

records=[]; total_kl=0.;total_tokens=0;total_top1=0
for i,row in enumerate(rows):
 rawpath=ROOT/'results/004_functional/teacher_sequences'/row['source_relative_path']
 with np.load(rawpath,allow_pickle=False) as z: raw=z['input_ids'].copy()
 messages=[{'role':'system','content':SYS},{'role':'user','content':ds[i]['question']}]
 rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 pids=tok(rendered,return_tensors='pt',add_special_tokens=False)['input_ids'].cpu().numpy()[0].astype(np.int32)
 assert len(pids)==row['answer_start'] and np.array_equal(pids,raw[:len(pids)])
 a,ha,ra=schedule(i);b,hb,rb=schedule(i)
 assert ra and rb and ha==hb and torch.equal(a,b)
 ids=torch.from_numpy(raw.astype(np.int64))[None].cuda(); p=torch.from_numpy(pids.astype(np.int64))[None].cuda()
 full,cf=run(ids,b[:,:len(raw)]); prefix,cp=run(p,a[:,:len(pids)])
 diagnostics={}
 for key in ('x','h16'):
  d=(prefix[key].float()-full[key][:,:len(pids)].float()).abs();diagnostics[key+'_mean_abs']=float(d.mean());diagnostics[key+'_max_abs']=float(d.max())
 prefix_logits=prefix['logits'];full_logits=full['logits'][:,:len(pids)];ld=(prefix_logits-full_logits).abs()
 diagnostics['logit_mean_abs']=float(ld.mean());diagnostics['logit_max_abs']=float(ld.max())
 pl=F.log_softmax(prefix_logits,dim=-1);ql=F.log_softmax(full_logits,dim=-1)
 kls=float((pl.exp()*(pl-ql)).sum());top=int((prefix_logits.argmax(-1)==full_logits.argmax(-1)).sum())
 diagnostics['kl_per_token']=kls/len(pids);diagnostics['top1_agreement']=top/len(pids)
 total_kl+=kls;total_tokens+=len(pids);total_top1+=top
 records.append({'example_id':i,'prompt_tokens':len(pids),'sequence_tokens':len(raw),'two_schedule_hashes_equal':True,'full_schedule_sha256':ha,'all_represented_prefixes_bitwise_equal':all(torch.equal(a[:,:t],b[:,:t]) for t in range(1,len(raw)+1)),'rng_restored_both':True,'normal_d16_calls':cf,'prefix_d16_calls':cp,'normal_vs_literal_coda_logits_exact':True,'diagnostics':diagnostics})
 del a,b,ids,p,full,prefix;torch.cuda.empty_cache()
out={'protocol':'phase9r-independent-prebuild-numerical-v3','status':'pass','cache_built':False,'optimizer_steps':0,'examples':records,'global_diagnostics':{'represented_prompt_tokens':total_tokens,'kl_per_token':total_kl/total_tokens,'top1_agreement':total_top1/total_tokens},'diagnostics_only_no_threshold':True}
output=Path('/workspace/functional_protocol/phase9r_prebuild_numerical_v3.json')
if output.exists(): raise FileExistsError(output)
output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))
