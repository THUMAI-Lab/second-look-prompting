#!/usr/bin/env python3
"""Relocate the exact native latency configuration; do not start any workload."""
import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
for k in ['template','model','python','output','cache','write']:p.add_argument('--'+k,required=True)
a=p.parse_args();repo=Path(__file__).resolve().parents[1]
d=json.loads(Path(a.template).read_text());mapping={'@REPO@':str(repo),'@MODEL@':a.model,'@PYTHON@':a.python,'@OUTPUT@':a.output,'@CACHE@':a.cache,'@LMUDATA@':os.environ['LMUData']}
def relocate(x):
 if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
 if isinstance(x,list):return [relocate(v) for v in x]
 if isinstance(x,str):
  for k,v in mapping.items():x=x.replace(k,v)
 return x
out=Path(a.write)
with out.open('x') as f:json.dump(relocate(d),f,indent=2)
print(out)
