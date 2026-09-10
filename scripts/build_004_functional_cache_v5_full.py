#!/usr/bin/env python3
"""Build the immutable full 2,500-item Phase 14 native-state cache."""
from __future__ import annotations
import argparse, ctypes, errno, gc, hashlib, json, os, sys, tempfile, uuid
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.build_004_functional_cache_v2 import _git_commit
from src.data.functional_cache_v5_full import (CACHE_V5_ITEM_PROTOCOL, FULL_EXAMPLE_IDS,
    build_manifest, freeze_cache_directory, manifest_sha256, schedule_sha256,
    validate_item, write_item_atomic, write_manifest_atomic)
from src.data.functional_extraction_v5 import EXTRACTION_LOCATION_V5, NORMALIZED_H16_SEMANTICS, capture_scheduled_huginn_states
from src.data.valid_end_manifest import load_and_validate_manifest
from src.evaluation.correctness import SEED_PROTOCOL, per_example_seed
from src.training.functional_protocol import FIXED_H0_SCHEDULE_LENGTH, FIXED_H0_SCHEDULE_PROTOCOL, fixed_huginn_h0_schedule, schedule_prefix

CONFIG=Path("configs/004_functional_cache_v5_full.json"); OUTPUT=Path("/workspace/functional_cache_v5_full")
MODEL_ID="tomg-group-umd/huginn-0125"; MODEL_REVISION="bb6621b65e90b6a4b9b29ef88dc83866d450470c"
EXPECTED_HASHES={"split_manifest_sha256":"afdbc0c55196104e276df0d65c8df24664836874a87b623cd5f3dd8331d1a5d8","valid_end_manifest_sha256":"25188506cb7afe426e42beb5fa96455fe8ac0cff6c0ae47b72363a3fa7af187f","source_cache_v2_manifest_sha256":"32db951f6cfbf76220b1c7126e23149f087eba58934ab691b588aeb3a43535bf","source_cache_v2_freeze_sha256":"94417eab65fd04a5827bdef9aadc9a5b8b66c266700b6cfc7ab56d39f949d3f0"}
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
 fixed={"protocol":"functional-cache-v5-full-build-v1","model_id":MODEL_ID,"model_revision":MODEL_REVISION,"dataset_id":"openai/gsm8k","dataset_config":"main","dataset_revision":"740312add88f781978c0658806c59bc2815b9866","split_manifest":"configs/004_functional_v2_splits.json","valid_end_manifest":"results/004_functional/valid_end_manifest.jsonl","source_cache_v2_manifest":"results/004_functional/cache_v2_manifest.json","source_cache_v2_freeze":"results/004_functional/cache_v2_FROZEN.json","source_continuations":"results/004_functional/teacher_sequences","example_id_start":0,"example_id_stop":2500,"teacher_depth":16,"compute_dtype":"bfloat16","state_dtype":"h0=bfloat16-bits-uint16;x=float32;h16=float32","hidden_size":5280,"base_seed":3000,"seed_index":0,"schedule_length":2048,"generation_cap_tokens":1024,"maximum_context_tokens":2048,"cache_output":str(OUTPUT),"validation_replay_selection_seed":1414,"validation_replay_ids":[0,13,261,342,383,491,569,842,951,1068,1425,1487,1512,1824,1922,1957,2249,2250,2289,2499]}
 for k,v in {**fixed,**EXPECTED_HASHES}.items():
  if c.get(k)!=v: raise ValueError(f"config {k} must equal locked value {v!r}")
 if set(c)!={*fixed,*EXPECTED_HASHES}: raise ValueError("full cache-v5 config has unexpected keys")
def locked_rows(c):
 for p,h in (("split_manifest","split_manifest_sha256"),("valid_end_manifest","valid_end_manifest_sha256"),("source_cache_v2_manifest","source_cache_v2_manifest_sha256"),("source_cache_v2_freeze","source_cache_v2_freeze_sha256")):
  if file_sha256(ROOT/c[p])!=c[h]: raise ValueError(f"locked source changed: {p}")
 all_rows={row["example_id"]:row for row in load_and_validate_manifest(ROOT/c["valid_end_manifest"])}
 rows=[all_rows[i] for i in FULL_EXAMPLE_IDS]
 for row in rows:
  expected_role="gradient_update" if row["example_id"] < 2250 else "checkpoint_selection_validation"
  if row["phase3_role"]!=expected_role or row["dataset_split"]!="train": raise ValueError("full-cache ID role differs from locked protocol")
  example_id=row["example_id"]
  assert_schedule_bounds(rendered_prompt_tokens=row["answer_start"],sequence_tokens=row["source_sequence_length"],cap=c["generation_cap_tokens"])
 return rows
def preflight_estimate(c,rows):
 tokens=sum(r["source_sequence_length"] for r in rows); payload=tokens*(c["hidden_size"]*10+5)
 return {"item_count":len(FULL_EXAMPLE_IDS),"total_tokens":tokens,"estimated_uncompressed_item_bytes":payload,"largest_transient_schedule_bytes":2048*c["hidden_size"]*2,"minimum_free_bytes_for_cache_and_replica":payload*3}

def storage_preflight(output:Path,minimum_free_bytes:int)->dict[str,Any]:
 """Fail closed on durable writes and Linux no-replace rename in output's filesystem."""
 parent=output.parent
 if parent.is_symlink() or not parent.is_dir(): raise RuntimeError(f"output parent must be an existing real directory: {parent}")
 stats=os.statvfs(parent);free=int(stats.f_bavail)*int(stats.f_frsize)
 if free<minimum_free_bytes: raise OSError(errno.ENOSPC,f"free bytes {free} below full-cache plus replica requirement {minimum_free_bytes}",str(parent))
 probe=Path(tempfile.mkdtemp(prefix=f".{output.name}.preflight-",dir=parent));source=probe/"source";target=probe/"target"
 try:
  with source.open("xb") as stream:
   stream.write(b"phase9r-source");stream.flush();os.fsync(stream.fileno())
  with target.open("xb") as stream:
   stream.write(b"phase9r-preserve");stream.flush();os.fsync(stream.fileno())
  if source.stat().st_dev!=parent.stat().st_dev or target.stat().st_dev!=parent.stat().st_dev: raise RuntimeError("storage probes are not on output filesystem")
  fd=os.open(probe,os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
  fn=getattr(ctypes.CDLL(None,use_errno=True),"renameat2",None)
  if fn is None: raise RuntimeError("Linux renameat2 is required")
  ctypes.set_errno(0);rc=fn(-100,os.fsencode(source),-100,os.fsencode(target),1);observed=ctypes.get_errno()
  if rc==0 or observed!=errno.EEXIST: raise RuntimeError(f"RENAME_NOREPLACE existing-target probe failed: rc={rc} errno={observed}")
  if source.read_bytes()!=b"phase9r-source" or target.read_bytes()!=b"phase9r-preserve": raise RuntimeError("RENAME_NOREPLACE did not preserve existing target and source")
  fd=os.open(probe,os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
  return {"free_bytes":free,"write_flush_fsync":True,"directory_fsync":True,"same_filesystem":True,"renameat2_noreplace_existing_preserved":True}
 finally:
  for path in (source,target):
   try:path.unlink()
   except FileNotFoundError:pass
  probe.rmdir()
  fd=os.open(parent,os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)

def publish_directory_no_replace(attempt,output):
 if output.exists() or output.is_symlink(): raise FileExistsError(f"refusing to replace immutable cache: {output}")
 fn=getattr(ctypes.CDLL(None,use_errno=True),"renameat2",None)
 if fn is None: raise RuntimeError("Linux renameat2 is required for atomic no-replace publication")
 if fn(-100,os.fsencode(attempt),-100,os.fsencode(output),1)!=0:
  e=ctypes.get_errno(); raise OSError(e,os.strerror(e),str(output))
 fd=os.open(output.parent,os.O_RDONLY); os.fsync(fd); os.close(fd)
def main(config_path:Path,preflight_only=False):
 c=json.loads((ROOT/config_path).read_text()); validate_config(c); rows=locked_rows(c); estimate=preflight_estimate(c,rows)
 storage=storage_preflight(OUTPUT,estimate["minimum_free_bytes_for_cache_and_replica"])
 print(json.dumps({"protocol":"functional-cache-v5-full-preflight-v1",**estimate,"storage":storage},sort_keys=True))
 if preflight_only:return
 if OUTPUT.exists() or OUTPUT.is_symlink(): raise FileExistsError(f"refusing to replace immutable cache: {OUTPUT}")
 if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
 commit=_git_commit()
 from transformers import AutoModelForCausalLM,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(c["model_id"],revision=c["model_revision"],local_files_only=True)
 if not isinstance(tok.chat_template,str) or not tok.chat_template: raise RuntimeError("pinned tokenizer has no chat template")
 manifest=build_manifest(model_revision=c["model_revision"],tokenizer_id=c["model_id"],tokenizer_revision=c["model_revision"],tokenizer_template=tok.chat_template,git_commit=commit,extraction_location=EXTRACTION_LOCATION_V5,source_continuations=c["source_continuations"],source_sha256_by_id={row["example_id"]:row["source_sha256"] for row in rows},base_seed=3000,seed_index=0,cache_output=str(OUTPUT),split_manifest_sha256=c["split_manifest_sha256"],source_cache_v2_manifest_sha256=c["source_cache_v2_manifest_sha256"],source_cache_v2_freeze_sha256=c["source_cache_v2_freeze_sha256"],estimated_uncompressed_item_bytes=estimate["estimated_uncompressed_item_bytes"])
 model=AutoModelForCausalLM.from_pretrained(c["model_id"],revision=c["model_revision"],torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).eval().cuda().requires_grad_(False)
 if int(model.config.n_embd)!=c["hidden_size"]: raise RuntimeError("pinned Huginn hidden size changed")
 attempt=OUTPUT.parent/f".{OUTPUT.name}.building"; attempt.mkdir(exist_ok=True); print(f"resumable staging path: {attempt}")
 write_manifest_atomic(attempt/"manifest.json",manifest); digest=manifest_sha256(manifest)
 for item_index,row in enumerate(rows,1):
  item_path=attempt/f"{row['example_id']:05d}.npz"
  if item_path.exists():
   try:
    validate_item(item_path,manifest)
   except ValueError:
    pass
   else:
    print(f"item={item_index}/2500 id={row['example_id']} status=validated_existing",flush=True)
    continue
  source=ROOT/c["source_continuations"]/row["source_relative_path"]
  ids_np=load_raw_source(source,row)
  ids=torch.from_numpy(ids_np).unsqueeze(0).cuda(); mask=torch.ones_like(ids,dtype=torch.bool)
  template=torch.empty((1,2048,c["hidden_size"]),device="cuda",dtype=torch.bfloat16)
  schedule,seed=fixed_huginn_h0_schedule(model,template,base_seed=3000,example_id=row["example_id"]); prefix=schedule_prefix(schedule,len(ids_np))
  states=capture_scheduled_huginn_states(model,ids,mask,prefix)
  arrays={"input_ids":ids_np,"attention_mask":np.ones(ids_np.shape,dtype=np.bool_),"answer_start":np.array(row["answer_start"],dtype=np.int32),"valid_end":np.array(row["reviewed_valid_end"],dtype=np.int32),**states,"example_id":np.array(row["example_id"],dtype=np.int64),"dataset_split":np.array("train"),"base_seed":np.array(3000,dtype=np.int64),"seed_index":np.array(0,dtype=np.int64),"derived_seed":np.array(seed,dtype=np.int64),"seed_protocol":np.array(SEED_PROTOCOL),"schedule_length":np.array(2048,dtype=np.int64),"schedule_protocol":np.array(FIXED_H0_SCHEDULE_PROTOCOL),"full_schedule_sha256":np.array(schedule_sha256(schedule)),"source_sha256":np.array(row["source_sha256"]),"h16_semantics":np.array(NORMALIZED_H16_SEMANTICS),"cache_protocol":np.array(CACHE_V5_ITEM_PROTOCOL),"manifest_sha256":np.array(digest)}
  write_item_atomic(item_path,arrays,manifest); validate_item(item_path,manifest)
  print(f"item={item_index}/2500 id={row['example_id']} status=validated",flush=True)
  del template,schedule,prefix,states,arrays,ids,mask; torch.cuda.empty_cache();gc.collect()
 quarantine=attempt/"quarantine"
 if quarantine.exists():
  evidence_root=OUTPUT.parent/f"{OUTPUT.name}_quarantine_records"
  evidence_root.mkdir(exist_ok=True)
  evidence=evidence_root/f"build-{uuid.uuid4().hex}"
  os.replace(quarantine,evidence)
  for directory in (evidence_root,attempt):
   fd=os.open(directory,os.O_RDONLY);os.fsync(fd);os.close(fd)
  print(f"quarantine evidence preserved outside cache: {evidence}",flush=True)
 if {p.name for p in attempt.iterdir()}!={"manifest.json",".functional-cache-v5.lock",*(f"{i:05d}.npz" for i in FULL_EXAMPLE_IDS)}: raise RuntimeError("non-exact staging file set")
 os.chmod(attempt/".functional-cache-v5.lock",0o444)
 freeze_cache_directory(attempt); publish_directory_no_replace(attempt,OUTPUT)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=CONFIG);p.add_argument("--preflight-only",action="store_true");a=p.parse_args();main(a.config,a.preflight_only)
