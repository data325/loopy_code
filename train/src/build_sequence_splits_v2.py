#!/usr/bin/env python3
"""Build exact 80/10/10 temporal-group splits for the original UrbanTwin 10K."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--group-size", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    ids = sorted(p.stem for p in (a.source / "points").glob("*.npy") if int(p.stem) < 10000)
    if len(ids) != 10000: raise ValueError(f"expected 10000 original frames, got {len(ids)}")
    gids = {x: f"UCF-DT-V2X-Real-Seq-1:block-{int(x)//a.group_size:03d}" for x in ids}
    ordered = sorted(set(gids.values()), key=lambda x: hashlib.sha1(f"{a.seed}:{x}".encode()).digest())
    n = len(ordered); ns = max(1, round(n*.1)); nv = max(1, round(n*.1))
    assignment = {g: ("sensor_stress_val" if i < ns else "scenario_val" if i < ns+nv else "train") for i,g in enumerate(ordered)}
    splits = {k: [] for k in ("train","scenario_val","sensor_stress_val")}
    groups = {}
    for x in ids:
        g=gids[x]; s=assignment[g]; splits[s].append(x); groups.setdefault(g,{"split":s,"ids":[]})["ids"].append(x)
    a.output.mkdir(parents=True, exist_ok=True)
    for k,v in splits.items(): (a.output/f"{k}.txt").write_text("\n".join(v)+"\n")
    (a.output/"val.txt").write_text("\n".join(splits["scenario_val"])+"\n")
    m={"method":"stable-hash ordered contiguous temporal blocks; exact group allocation","group_size":a.group_size,"seed":a.seed,
       "ellipsoid_augmented_ids_excluded":"010000-011999","counts":{k:len(v) for k,v in splits.items()},"groups":groups}
    (a.output/"manifest.json").write_text(json.dumps(m,indent=2))
    print(json.dumps({**m,"groups":len(groups)},indent=2))
if __name__=="__main__": main()
