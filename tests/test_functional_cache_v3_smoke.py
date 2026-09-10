import copy, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scripts import build_004_functional_cache_v3_smoke as builder
from scripts import validate_004_functional_cache_v3_smoke as validator
from src.data.functional_cache_v3 import CACHE_V3_ITEM_KEYS, build_manifest, freeze_cache_directory, manifest_sha256, schedule_sha256, validate_item, write_item_atomic, write_manifest_atomic
from src.data.functional_extraction_v3 import NORMALIZED_H16_SEMANTICS, capture_scheduled_huginn_states
from src.evaluation.correctness import SEED_PROTOCOL, per_example_seed
from src.training.functional_protocol import FIXED_H0_SCHEDULE_LENGTH, FIXED_H0_SCHEDULE_PROTOCOL, fixed_huginn_h0_schedule, schedule_prefix
ROOT=Path(__file__).resolve().parents[1]
RAW={"0":"8ddef09070e5b01b4a43f474b365f95e2d361d2afbf7f5eb151f8029b220f967","1":"e5f7bcaf7b3ad0627592ff727c32a2f10df99269ab54441919289b04dff3c025","2":"a36e3a8cab93e126059c54fd59eb7ee14b41394ab1dd30fac2ad95e6b6db2e38","3":"ba6dffd0c3bea158fe57f84fc5c258ec0b9b53fa31875a5858d6cbe0c4d34a50","4":"04df30834bf1b9cf3be946aab9d4b53b30125fa308fabe31f9532cdeb4c0fe5b","5":"f76a673cfba89cc5be6f136a8aa6b98ac163f7e826fbb20fe5ddf6905a5e2430","6":"f3e7c18f2604443999a48497144765baa28955b4bb30a8e90f44bda54ea1dd0e","7":"20453212328435fcab00af8e34cd63b03ff897143c9e397dc28f473afb7da297"}
def manifest():
 return build_manifest(model_revision="b"*40,tokenizer_id="tomg-group-umd/huginn-0125",tokenizer_revision="b"*40,tokenizer_template="template",git_commit="a"*40,extraction_location="explicit input_states; normalized pre-coda",source_continuations="results/004_functional/teacher_sequences",source_sha256_by_id={int(k):v for k,v in RAW.items()},base_seed=3000,seed_index=0,cache_output="/workspace/functional_cache_v3_smoke",split_manifest_sha256=builder.EXPECTED_HASHES["split_manifest_sha256"],source_cache_v2_manifest_sha256=builder.EXPECTED_HASHES["source_cache_v2_manifest_sha256"],source_cache_v2_freeze_sha256=builder.EXPECTED_HASHES["source_cache_v2_freeze_sha256"],estimated_uncompressed_item_bytes=123)
def arrays(m,example_id=0,t=3,h=5280):
 return {"input_ids":np.arange(t,dtype=np.int32),"attention_mask":np.ones(t,dtype=np.bool_),"answer_start":np.array(1,dtype=np.int32),"valid_end":np.array(t,dtype=np.int32),"h0":np.zeros((t,h),dtype=np.float16),"x":np.ones((t,h),dtype=np.float16),"h16":np.full((t,h),2,dtype=np.float16),"example_id":np.array(example_id,dtype=np.int64),"dataset_split":np.array("train"),"base_seed":np.array(3000,dtype=np.int64),"seed_index":np.array(0,dtype=np.int64),"derived_seed":np.array(per_example_seed(3000,example_id,seed_index=0),dtype=np.int64),"seed_protocol":np.array(SEED_PROTOCOL),"schedule_length":np.array(2048,dtype=np.int64),"schedule_protocol":np.array(FIXED_H0_SCHEDULE_PROTOCOL),"full_schedule_sha256":np.array("f"*64),"source_sha256":np.array(RAW[str(example_id)]),"h16_semantics":np.array(NORMALIZED_H16_SEMANTICS),"cache_protocol":np.array("functional-cache-v3-smoke-item-v1"),"manifest_sha256":np.array(manifest_sha256(m))}
def test_config_locks_v2_manifest_freeze_sidecar_split_and_raw_ids():
 c=json.loads((ROOT/"configs/004_functional_cache_v3_smoke.json").read_text());builder.validate_config(c)
 assert c["example_ids"]==list(range(8)) and c["cache_output"]=="/workspace/functional_cache_v3_smoke"
 changed=copy.deepcopy(c);changed["raw_source_sha256"]["7"]="0"*64
 with pytest.raises(ValueError,match="raw hashes must equal"):builder.validate_config(changed)
def test_preflight_estimates_only_eight_compact_items():
 c=json.loads((ROOT/"configs/004_functional_cache_v3_smoke.json").read_text());rows=[{"source_sequence_length":n} for n in (155,118,270,434,147,300,194,311)]
 p=builder.preflight_estimate(c,rows);assert p["item_count"]==8 and p["largest_transient_schedule_bytes"]==2048*5280*2 and p["minimum_free_bytes_with_2x_margin"]==2*p["estimated_uncompressed_item_bytes"]
def test_schema_has_exact_replay_coda_semantics_and_no_logits_or_full_schedule(tmp_path):
 m=manifest();a=arrays(m);path=tmp_path/"00000.npz";assert write_item_atomic(path,a,m)=="created";meta=validate_item(path,m)
 assert meta["sequence_length"]<FIXED_H0_SCHEDULE_LENGTH
 assert (path.stat().st_mode&0o777)==0o444
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)==CACHE_V3_ITEM_KEYS;assert not any("logit" in k.lower() for k in z.files);assert "schedule" not in z.files;assert str(z["h16_semantics"])==NORMALIZED_H16_SEMANTICS
def test_item_crash_safe_and_existing_content_is_never_silently_overwritten(tmp_path):
 m=manifest();a=arrays(m);path=tmp_path/"00000.npz";write_item_atomic(path,a,m);before=path.read_bytes();assert write_item_atomic(path,a,m)=="validated_existing";assert path.read_bytes()==before
 wrong=arrays(m);wrong["x"]=np.full_like(wrong["x"],9);assert write_item_atomic(path,wrong,m)=="created_after_quarantine";assert any((tmp_path/"quarantine").glob("*.quarantine"))
def test_complete_cache_freeze_makes_files_and_directory_read_only(tmp_path):
 m=manifest();root=tmp_path/"cache";root.mkdir();write_manifest_atomic(root/"manifest.json",m)
 for example_id in range(8):write_item_atomic(root/f"{example_id:05d}.npz",arrays(m,example_id=example_id),m)
 freeze_cache_directory(root)
 try:
  assert (root.stat().st_mode&0o777)==0o555
  assert all((path.stat().st_mode&0o777)==0o444 for path in root.iterdir())
 finally:root.chmod(0o755)

def test_final_directory_publication_refuses_replacement_before_rename(tmp_path):
 attempt=tmp_path/"attempt";attempt.mkdir();output=tmp_path/"final";output.mkdir();marker=output/"preserved";marker.write_text("raw")
 with pytest.raises(FileExistsError):builder.publish_directory_no_replace(attempt,output)
 assert marker.read_text()=="raw" and attempt.exists()
def test_validator_rejects_k_directories_and_logits(tmp_path):
 root=tmp_path/"cache";root.mkdir();(root/"manifest.json").touch()
 for i in range(8):(root/f"{i:05d}.npz").touch()
 (root/"K4").mkdir()
 with pytest.raises(ValueError,match="no K directories"):validator.validate_file_set(root)

class FakeHuginn:
 def parameters(self):return iter(())
 def core_block_forward(self,state,input_embeds,*args,**kwargs):return state+input_embeds,torch.tensor(0)
 def __call__(self,**kwargs):
  assert kwargs["input_states"].shape[1]==kwargs["input_ids"].shape[1];assert kwargs["output_details"]["return_logits"] is False
  state=kwargs["input_states"];x=torch.ones_like(state)
  for _ in range(kwargs["num_steps"]):state,_=self.core_block_forward(state,x)
  return SimpleNamespace(latent_states=state)
def test_builder_has_no_dynamic_prefix_initializer_call():
 source=(ROOT/"scripts/build_004_functional_cache_v3_smoke.py").read_text()
 assert ".initialize_state(" not in source
 assert "fixed_huginn_h0_schedule(model,template" in source

def test_extraction_injects_explicit_slice_and_never_persists_logits():
 model=FakeHuginn();h0=torch.zeros((1,3,4),dtype=torch.bfloat16);ids=torch.arange(3)[None];states=capture_scheduled_huginn_states(model,ids,torch.ones_like(ids,dtype=torch.bool),h0)
 assert set(states)=={"h0","x","h16"};assert np.all(states["h16"]==16)

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_one_fixed_schedule_call_restores_rng_and_every_prefix_is_bit_exact():
 class Initializer:
  def __init__(self):self.calls=0
  def initialize_state(self,template,scale=1.0):self.calls+=1;return torch.randn_like(template)*scale
 model=Initializer();template=torch.empty((1,2048,4),device="cuda",dtype=torch.bfloat16);torch.manual_seed(77);cpu=torch.random.get_rng_state().clone();cuda=[s.clone() for s in torch.cuda.get_rng_state_all()]
 schedule,seed=fixed_huginn_h0_schedule(model,template,base_seed=3000,example_id=7)
 assert model.calls==1 and seed==per_example_seed(3000,7,seed_index=0)
 assert len(schedule_sha256(schedule))==64
 assert torch.equal(cpu,torch.random.get_rng_state());assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
 for t in range(1,2049):assert torch.equal(schedule_prefix(schedule,t),schedule[:,:t])
