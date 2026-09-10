#!/usr/bin/env python3
"""Build only the immutable eight-item Phase 9R prefix-stable cache-v3 smoke."""
from __future__ import annotations
import argparse, ctypes, gc, hashlib, json, os, sys, uuid
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.build_004_functional_cache_v2 import _git_commit
from src.data.functional_cache_v3 import (CACHE_V3_ITEM_PROTOCOL, SMOKE_EXAMPLE_IDS,
    build_manifest, freeze_cache_directory, manifest_sha256, schedule_sha256,
    validate_item, write_item_atomic, write_manifest_atomic)
from src.data.functional_extraction_v3 import EXTRACTION_LOCATION_V3, NORMALIZED_H16_SEMANTICS, capture_scheduled_huginn_states
from src.data.valid_end_manifest import load_and_validate_manifest
from src.evaluation.correctness import SEED_PROTOCOL, per_example_seed
from src.training.functional_protocol import FIXED_H0_SCHEDULE_LENGTH, FIXED_H0_SCHEDULE_PROTOCOL, fixed_huginn_h0_schedule, schedule_prefix

CONFIG=Path("configs/004_functional_cache_v3_smoke.json"); OUTPUT=Path("/workspace/functional_cache_v3_smoke")
MODEL_ID="tomg-group-umd/huginn-0125"; MODEL_REVISION="bb6621b65e90b6a4b9b29ef88dc83866d450470c"
EXPECTED_HASHES={"split_manifest_sha256":"afdbc0c55196104e276df0d65c8df24664836874a87b623cd5f3dd8331d1a5d8","valid_end_manifest_sha256":"25188506cb7afe426e42beb5fa96455fe8ac0cff6c0ae47b72363a3fa7af187f","source_cache_v2_manifest_sha256":"32db951f6cfbf76220b1c7126e23149f087eba58934ab691b588aeb3a43535bf","source_cache_v2_freeze_sha256":"94417eab65fd04a5827bdef9aadc9a5b8b66c266700b6cfc7ab56d39f949d3f0"}
EXPECTED_RAW_HASHES={"0":"8ddef09070e5b01b4a43f474b365f95e2d361d2afbf7f5eb151f8029b220f967","1":"e5f7bcaf7b3ad0627592ff727c32a2f10df99269ab54441919289b04dff3c025","2":"a36e3a8cab93e126059c54fd59eb7ee14b41394ab1dd30fac2ad95e6b6db2e38","3":"ba6dffd0c3bea158fe57f84fc5c258ec0b9b53fa31875a5858d6cbe0c4d34a50","4":"04df30834bf1b9cf3be946aab9d4b53b30125fa308fabe31f9532cdeb4c0fe5b","5":"f76a673cfba89cc5be6f136a8aa6b98ac163f7e826fbb20fe5ddf6905a5e2430","6":"f3e7c18f2604443999a48497144765baa28955b4bb30a8e90f44bda54ea1dd0e","7":"20453212328435fcab00af8e34cd63b03ff897143c9e397dc28f473afb7da297"}
def file_sha256(path:Path)->str:
 d=hashlib.sha256();
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): d.update(b)
 return d.hexdigest()
def load_raw_source(path:Path,row:dict[str,Any])->np.ndarray:
 if path.is_symlink() or not path.is_file(): raise ValueError(f"raw source must be a regular file: {path}")
 if file_sha256(path)!=row["source_sha256"]: raise ValueError(f"raw source hash changed: {path}")
 with np.load(path,allow_pickle=False) as archive:
  ids=archive["input_ids"].copy(); answer_start=int(archive["answer_start"])
 if ids.ndim!=1 or ids.dtype!=np.int32: raise ValueError("raw input_ids must be rank-1 int32")
 if answer_start!=row["answer_start"] or len(ids)!=row["source_sequence_length"]: raise ValueError("raw source metadata differs from frozen sidecar")
 return ids

def assert_schedule_bounds(*,rendered_prompt_tokens:int,sequence_tokens:int,cap:int=1024)->None:
 if rendered_prompt_tokens+cap>FIXED_H0_SCHEDULE_LENGTH: raise ValueError("rendered prompt plus 1024-token cap exceeds 2048")
 if sequence_tokens>FIXED_H0_SCHEDULE_LENGTH: raise ValueError("teacher sequence exceeds 2048")

def validate_config(c:dict[str,Any])->None:
 fixed={"protocol":"functional-cache-v3-smoke-build-v1","model_id":MODEL_ID,"model_revision":MODEL_REVISION,"dataset_id":"openai/gsm8k","dataset_config":"main","dataset_revision":"740312add88f781978c0658806c59bc2815b9866","split_manifest":"configs/004_functional_v2_splits.json","valid_end_manifest":"results/004_functional/valid_end_manifest.jsonl","source_cache_v2_manifest":"results/004_functional/cache_v2_manifest.json","source_cache_v2_freeze":"results/004_functional/cache_v2_FROZEN.json","source_continuations":"results/004_functional/teacher_sequences","example_ids":list(SMOKE_EXAMPLE_IDS),"teacher_depth":16,"compute_dtype":"bfloat16","state_dtype":"float16","hidden_size":5280,"base_seed":3000,"seed_index":0,"schedule_length":2048,"generation_cap_tokens":1024,"maximum_context_tokens":2048,"cache_output":str(OUTPUT)}
 for k,v in {**fixed,**EXPECTED_HASHES}.items():
  if c.get(k)!=v: raise ValueError(f"config {k} must equal locked value {v!r}")
 if set(c)!={*fixed,*EXPECTED_HASHES,"raw_source_sha256"}: raise ValueError("cache-v3 config has unexpected keys")
 if c.get("raw_source_sha256")!=EXPECTED_RAW_HASHES: raise ValueError("raw hashes must equal the locked IDs 0-7 digests")
def locked_rows(c):
 for p,h in (("split_manifest","split_manifest_sha256"),("valid_end_manifest","valid_end_manifest_sha256"),("source_cache_v2_manifest","source_cache_v2_manifest_sha256"),("source_cache_v2_freeze","source_cache_v2_freeze_sha256")):
  if file_sha256(ROOT/c[p])!=c[h]: raise ValueError(f"locked source changed: {p}")
 rows=load_and_validate_manifest(ROOT/c["valid_end_manifest"])[:8]
 for i,row in enumerate(rows):
  if row["example_id"]!=i or row["phase3_role"]!="gradient_update" or row["dataset_split"]!="train": raise ValueError("requires immutable raw teacher train IDs 0-7")
  if row["source_sha256"]!=c["raw_source_sha256"][str(i)]: raise ValueError(f"raw hash lock differs for ID {i}")
  assert_schedule_bounds(rendered_prompt_tokens=row["answer_start"],sequence_tokens=row["source_sequence_length"],cap=c["generation_cap_tokens"])
 return rows
def preflight_estimate(c,rows):
 tokens=sum(r["source_sequence_length"] for r in rows); payload=tokens*(3*c["hidden_size"]*2+5)
 return {"item_count":8,"total_tokens":tokens,"estimated_uncompressed_item_bytes":payload,"largest_transient_schedule_bytes":2048*c["hidden_size"]*2,"minimum_free_bytes_with_2x_margin":payload*2}
def publish_directory_no_replace(attempt,output):
 if output.exists() or output.is_symlink(): raise FileExistsError(f"refusing to replace immutable cache: {output}")
 fn=getattr(ctypes.CDLL(None,use_errno=True),"renameat2",None)
 if fn is None: raise RuntimeError("Linux renameat2 is required for atomic no-replace publication")
 if fn(-100,os.fsencode(attempt),-100,os.fsencode(output),1)!=0:
  e=ctypes.get_errno(); raise OSError(e,os.strerror(e),str(output))
 fd=os.open(output.parent,os.O_RDONLY); os.fsync(fd); os.close(fd)
def main(config_path:Path,preflight_only=False):
 c=json.loads((ROOT/config_path).read_text()); validate_config(c); rows=locked_rows(c); estimate=preflight_estimate(c,rows)
 print(json.dumps({"protocol":"functional-cache-v3-smoke-preflight-v1",**estimate},sort_keys=True))
 if preflight_only:return
 if OUTPUT.exists() or OUTPUT.is_symlink(): raise FileExistsError(f"refusing to replace immutable cache: {OUTPUT}")
 if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
 commit=_git_commit()
 from transformers import AutoModelForCausalLM,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(c["model_id"],revision=c["model_revision"],local_files_only=True)
 if not isinstance(tok.chat_template,str) or not tok.chat_template: raise RuntimeError("pinned tokenizer has no chat template")
 manifest=build_manifest(model_revision=c["model_revision"],tokenizer_id=c["model_id"],tokenizer_revision=c["model_revision"],tokenizer_template=tok.chat_template,git_commit=commit,extraction_location=EXTRACTION_LOCATION_V3,source_continuations=c["source_continuations"],source_sha256_by_id={i:c["raw_source_sha256"][str(i)] for i in SMOKE_EXAMPLE_IDS},base_seed=3000,seed_index=0,cache_output=str(OUTPUT),split_manifest_sha256=c["split_manifest_sha256"],source_cache_v2_manifest_sha256=c["source_cache_v2_manifest_sha256"],source_cache_v2_freeze_sha256=c["source_cache_v2_freeze_sha256"],estimated_uncompressed_item_bytes=estimate["estimated_uncompressed_item_bytes"])
 model=AutoModelForCausalLM.from_pretrained(c["model_id"],revision=c["model_revision"],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda().requires_grad_(False)
 if int(model.config.n_embd)!=c["hidden_size"]: raise RuntimeError("pinned Huginn hidden size changed")
 attempt=OUTPUT.parent/f".{OUTPUT.name}.attempt-{uuid.uuid4().hex}"; attempt.mkdir(exist_ok=False); print(f"preserved staging path on failure: {attempt}")
 write_manifest_atomic(attempt/"manifest.json",manifest); digest=manifest_sha256(manifest)
 for row in rows:
  source=ROOT/c["source_continuations"]/row["source_relative_path"]
  ids_np=load_raw_source(source,row)
  ids=torch.from_numpy(ids_np).unsqueeze(0).cuda(); mask=torch.ones_like(ids,dtype=torch.bool)
  template=torch.empty((1,2048,c["hidden_size"]),device="cuda",dtype=torch.bfloat16)
  schedule,seed=fixed_huginn_h0_schedule(model,template,base_seed=3000,example_id=row["example_id"]); prefix=schedule_prefix(schedule,len(ids_np))
  states=capture_scheduled_huginn_states(model,ids,mask,prefix)
  arrays={"input_ids":ids_np,"attention_mask":np.ones(ids_np.shape,dtype=np.bool_),"answer_start":np.array(row["answer_start"],dtype=np.int32),"valid_end":np.array(row["reviewed_valid_end"],dtype=np.int32),**states,"example_id":np.array(row["example_id"],dtype=np.int64),"dataset_split":np.array("train"),"base_seed":np.array(3000,dtype=np.int64),"seed_index":np.array(0,dtype=np.int64),"derived_seed":np.array(seed,dtype=np.int64),"seed_protocol":np.array(SEED_PROTOCOL),"schedule_length":np.array(2048,dtype=np.int64),"schedule_protocol":np.array(FIXED_H0_SCHEDULE_PROTOCOL),"full_schedule_sha256":np.array(schedule_sha256(schedule)),"source_sha256":np.array(row["source_sha256"]),"h16_semantics":np.array(NORMALIZED_H16_SEMANTICS),"cache_protocol":np.array(CACHE_V3_ITEM_PROTOCOL),"manifest_sha256":np.array(digest)}
  write_item_atomic(attempt/f"{row['example_id']:05d}.npz",arrays,manifest); validate_item(attempt/f"{row['example_id']:05d}.npz",manifest)
  del template,schedule,prefix,states,arrays,ids,mask; torch.cuda.empty_cache();gc.collect()
 if {p.name for p in attempt.iterdir()}!={"manifest.json",".functional-cache-v3.lock",*(f"{i:05d}.npz" for i in SMOKE_EXAMPLE_IDS)}: raise RuntimeError("non-exact staging file set")
 os.chmod(attempt/".functional-cache-v3.lock",0o444)
 freeze_cache_directory(attempt); publish_directory_no_replace(attempt,OUTPUT)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=CONFIG);p.add_argument("--preflight-only",action="store_true");a=p.parse_args();main(a.config,a.preflight_only)
