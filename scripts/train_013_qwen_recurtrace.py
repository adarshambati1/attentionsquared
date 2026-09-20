#!/usr/bin/env python3
"""Step-3D launch interlock; Stage-1 training is intentionally not authorized yet."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.training.recurtrace_protocol import validate_training_authorization
CONFIG=ROOT/'configs/013_step3_qwen_recurtrace.json'
def main():
 config=json.loads(CONFIG.read_text());config['_config_path']=str(CONFIG)
 if config.get('training_launch_allowed') is not True:
  raise RuntimeError({'step3d_training_blocked':True,'reason':'The approved execution boundary permits implementation and correctness gates only. A new committed config plus user-approved authoritative asset manifest and sufficient compute are required before Stage-1 training.','blockers':config.get('training_blockers',[])})
 authorization=validate_training_authorization(config)
 raise RuntimeError({'step3d_training_not_implemented_for_authorized_assets':True,'authorization':authorization,'reason':'Implement the exact repository-specific data packing and distributed Stage-1 runner only after the authoritative assets are frozen.'})
if __name__=='__main__':main()
