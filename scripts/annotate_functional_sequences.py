#!/usr/bin/env python3
import glob,json
from pathlib import Path
import numpy as np

def onset(gen,n=12,reps=3):
    for i in range(len(gen)-n*reps+1):
        chunk=gen[i:i+n]
        if all(np.array_equal(chunk,gen[i+j*n:i+(j+1)*n]) for j in range(1,reps)):
            return i
    return None

root=Path('results/004_functional/teacher_sequences'); paths=glob.glob(str(root/'*'/'*.npz')); pathological=[]
for path in paths:
    with np.load(path) as a:
        seq=a['input_ids']; start=int(a['answer_start']); trunc=bool(a['truncated']) if 'truncated' in a.files else int(a['generated_tokens'])>=1024; end=len(seq)
        if trunc:
            o=onset(seq[start:])
            if o is not None and o>=16:
                end=start+o; pathological.append({'path':path,'generated_tokens':len(seq)-start,'valid_generated_tokens':o})
        fields={k:a[k] for k in a.files if k not in {'truncated','valid_end','pathological'}}
        fields.update(truncated=np.array(trunc,dtype=np.bool_), pathological=np.array(any(x['path']==path for x in pathological),dtype=np.bool_), valid_end=np.array(end,dtype=np.int32))
    np.savez(path,**fields)
meta=json.loads((root/'manifest.json').read_text()); meta.update(pathological_count=len(pathological),pathological_rate=len(pathological)/len(paths),pathological_examples=pathological,functional_training_rule='use answer positions [answer_start:valid_end]; exclude post-degeneration tails'); (root/'manifest.json').write_text(json.dumps(meta,indent=2)); print(json.dumps({'examples':len(paths),'pathological_count':len(pathological),'examples_preview':pathological[:10]},indent=2))
