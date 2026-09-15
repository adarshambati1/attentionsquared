#!/usr/bin/env python3
"""Authorized optional h4 generation follow-up and complete final table."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
from src.evaluation.correctness import build_chat_prompt,find_repetition_onset,score_generation,tokenize_prompt
from src.models.perturbed_huginn import PerturbedHuginn
from src.training.latent_history import materialize_h0_schedule
from scripts.run_011_perturb_recover_generation import generate,summarize
from scripts.run_011_perturb_recover_latent import write_exclusive_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/011_h4_generation_followup.json'));a=p.parse_args();follow_path=ROOT/a.config;follow=json.loads(follow_path.read_text());base_path=ROOT/follow['base_config'];c=json.loads(base_path.read_text());base_sha=hashlib.sha256(base_path.read_bytes()).hexdigest();commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();base_root=Path(follow['base_output_root']);out=Path(follow['output_root']);out.mkdir(exist_ok=True);lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if base_sha!=follow['base_config_sha256']:raise RuntimeError('base config hash mismatch')
 gate=json.loads((base_root/'correctness_gate.json').read_text());latent=json.loads((base_root/'latent_summary.json').read_text());h8=json.loads((base_root/'generation_summary.json').read_text())
 if gate.get('status')!='pass' or gate.get('config_sha256')!=base_sha or gate.get('git_commit')!=follow['base_latent_commit'] or latent.get('status')!='complete' or latent.get('git_commit')!=follow['base_latent_commit'] or h8.get('status')!='complete' or h8.get('config_sha256')!=base_sha:raise RuntimeError('base gate/latent/h8 generation provenance incomplete')
 ref_path=ROOT/'results/005_latent_history_attention/comparison.json';reference=json.loads(ref_path.read_text());clean={r['example_id']:r for r in reference['records'] if r['model']=='original_huginn_d16'}
 if len(clean)!=250:raise RuntimeError('paired clean D16 records incomplete')
 summary_path=out/'generation_summary.json';record_dir=out/'records';record_dir.mkdir(exist_ok=True)
 if summary_path.exists():raise FileExistsError(summary_path)
 tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'],local_files_only=True);test=load_dataset(c['dataset_id'],c['dataset_config'],split='test',revision=c['dataset_revision']);huginn=AutoModelForCausalLM.from_pretrained(c['model_id'],revision=c['model_revision'],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda();wrapper=PerturbedHuginn(huginn).cuda().eval();conditions=[(sigma,seed,i) for sigma in follow['noise_strengths'] for seed in follow['perturbation_seed_indices'] for i in range(follow['test_ids'][0],follow['test_ids'][1]+1)];records=[];follow_sha=hashlib.sha256(follow_path.read_bytes()).hexdigest()
 for path in sorted(record_dir.glob('*.json')):
  if path.name!=f'{len(records):06d}.json':raise RuntimeError(f'noncontiguous h4 record {path}')
  r=json.loads(path.read_text())
  if (r['sigma'],r['perturbation_seed_index'],r['example_id'])!=conditions[len(records)] or r.get('followup_config_sha256')!=follow_sha:raise RuntimeError(f'h4 resume mismatch {path}')
  records.append(r)
 for sigma,seed,i in conditions[len(records):]:
  prompt=tokenize_prompt(tok,build_chat_prompt(tok,test[i]['question'],c['system_instruction']))['input_ids'].cuda();schedule=materialize_h0_schedule(huginn,device=prompt.device,example_id=i,base_seed=c['h0_base_seed']);g=generate(wrapper,tok,prompt,schedule,example_id=i,perturb_depth=follow['perturb_depth'],sigma=sigma,seed=seed,c=c);score=score_generation(g['text'],test[i]['answer'],hit_max_new_tokens=g['hit_cap']);r={'perturb_depth':4,'sigma':sigma,'perturbation_seed_index':seed,'example_id':i,'followup_config_sha256':follow_sha,'correct':score.correct,'predicted_answer':score.predicted_answer,'gold_answer':score.gold_answer,'generated_text':g['text'],'generated_token_ids':g['token_ids'],'generated_tokens':len(g['token_ids']),'generation_latency_seconds':g['latency_seconds'],'peak_memory_bytes':g['peak_memory_bytes'],'hit_max_new_tokens':g['hit_cap'],'repetition_onset':find_repetition_onset(g['token_ids'],chunk_size=c['degeneration_chunk_size'],repetitions=c['degeneration_repetitions']),'clean_d16_correct':clean[i]['correct']};records.append(r);write_exclusive_json(record_dir/f'{len(records)-1:06d}.json',r);print(json.dumps({k:v for k,v in r.items() if k not in ('generated_text','generated_token_ids')}),flush=True)
 by_seed={f'sigma{sigma}:seed{seed}':summarize([r for r in records if r['sigma']==sigma and r['perturbation_seed_index']==seed]) for sigma in follow['noise_strengths'] for seed in follow['perturbation_seed_indices']};pooled={f'sigma{sigma}':summarize([r for r in records if r['sigma']==sigma]) for sigma in follow['noise_strengths']};result={'protocol':follow['protocol'],'status':'complete','perturb_depth':4,'final_depth':16,'per_seed':by_seed,'pooled_three_seeds':pooled,'records':len(records),'followup_config_sha256':follow_sha,'base_config_sha256':base_sha,'git_commit':commit,'full_vocabulary_logits_persisted':False};write_exclusive_json(out/'generation_records.jsonl',records);write_exclusive_json(summary_path,result)
 rows=[]
 for latent_row in latent['table']:
  depth=latent_row['perturb_depth'];sigma=latent_row['sigma'];functional=(pooled if depth==4 else h8['pooled_three_seeds'])[f'sigma{sigma}'];rows.append({'perturb_depth':depth,'sigma':sigma,'E_at_D16':latent_row['at_d16']['E_mean'],'E_at_D32':latent_row['at_d32']['E_mean'],'D16_cosine':latent_row['at_d16']['cosine_mean'],'GSM8K_accuracy':functional['accuracy'],'GSM8K_correct':functional['correct'],'GSM8K_total':functional['total'],'cap_hit_rate':functional['cap_hit_rate'],'degeneration_rate':functional['degeneration_rate'],'paired_wins_vs_clean':functional['paired_wins_vs_clean'],'paired_losses_vs_clean':functional['paired_losses_vs_clean']})
 write_exclusive_json(base_root/'complete_table.json',{'protocol':c['protocol']+'-complete-table','status':'complete','rows':rows,'h4_followup_commit':commit,'base_commit':follow['base_latent_commit']});print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
