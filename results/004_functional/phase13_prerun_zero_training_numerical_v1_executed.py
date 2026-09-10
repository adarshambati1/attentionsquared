import json,sys
from pathlib import Path
import torch
sys.path.insert(0,'/workspace/attentionsquared')
from scripts import run_004_phase13_evaluator_controls as p
from src.evaluation.correctness import build_chat_prompt,tokenize_prompt
from src.evaluation.functional_autoregressive import FullPrefixHuginnD16Evaluator
from src.training.functional_objective import freeze_module
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
config=json.load(open('/workspace/attentionsquared/configs/004_functional_v3_phase13.json'));p.validate_config(config)
assert not p.OUTPUT_ROOT.exists();device=torch.device('cuda')
tok=AutoTokenizer.from_pretrained(p.MODEL_ID,revision=p.MODEL_REVISION,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(p.MODEL_ID,revision=p.MODEL_REVISION,torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).cuda().eval();freeze_module(model)
cache=p.run_cache_controls(config,model,tok,device)
ds=load_dataset('openai/gsm8k','main',split='train',revision=p.DATASET_REVISION)
i=2250;text=build_chat_prompt(tok,ds[i]['question'],config['system_instruction']);ids=tokenize_prompt(tok,text)['input_ids'].cuda();ev=FullPrefixHuginnD16Evaluator(model,tok,base_seed=3000,seed_index=0)
route=p.compare_live_routes(model,tok,ev,ids,example_id=i,gold_text=ds[i]['answer'],max_new_tokens=8)
out={'protocol':'phase13-prerun-zero-training-numerical-v1','status':'pass' if cache['passed'] and route['passed'] else 'fail','optimizer_steps':0,'phase13_output_absent':not p.OUTPUT_ROOT.exists(),'cache_control':cache,'eight_token_route_control':route}
path=Path('/workspace/functional_protocol/phase13_prerun_zero_training_numerical_v1.json')
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(out['status'],cache['aggregate'],route['passed'])
