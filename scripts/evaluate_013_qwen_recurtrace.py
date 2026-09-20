#!/usr/bin/env python3
"""Step-3D evaluation interlock pending authorized Stage-1 checkpoints."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CONFIG=ROOT/'configs/013_step3_qwen_recurtrace.json'
def main():
 config=json.loads(CONFIG.read_text())
 raise RuntimeError({'step3d_evaluation_blocked':True,'reason':'No exact or explicitly approved paper-guided Stage-1 checkpoints exist. Evaluation at primary T=2 and secondary T=4 must remain blocked until training provenance is authorized and complete.','training_launch_allowed':config['training_launch_allowed'],'required_variants':config['evaluation_variants']})
if __name__=='__main__':main()
