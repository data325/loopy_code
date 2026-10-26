#!/usr/bin/env python3
"""Extract an unlabeled sensor profile from allowed V2X-Real train/-1 frames."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
RE=np.arange(0,45,5,dtype=np.float32); AE=np.linspace(-np.pi,np.pi,73,dtype=np.float32)
def load(p):
    x=np.fromfile(p,np.float32); w=4 if x.size%4==0 else 3; x=x.reshape(-1,w)[:,:3]
    r=np.linalg.norm(x,axis=1); return x[np.isfinite(x).all(1)&np.any(x!=0,1)&(r>.5)&(r<80)]
def rings(e,n):
    c=np.quantile(e,(np.arange(n)+.5)/n)
    for _ in range(20):
        a=np.abs(e[:,None]-c).argmin(1); u=np.array([np.median(e[a==i]) if np.any(a==i) else c[i] for i in range(n)])
        if np.max(np.abs(u-c))<1e-7: break
        c=u
    return np.sort(c).astype(np.float32)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--real-root",type=Path,required=True); ap.add_argument("--forbidden",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True); ap.add_argument("--rings",type=int,default=128); ap.add_argument("--max-frames",type=int,default=160)
    ap.add_argument("--points-per-frame",type=int,default=10000); ap.add_argument("--seed",type=int,default=20260723); a=ap.parse_args()
    bad={x.split("_")[-1].strip() for x in a.forbidden.read_text().splitlines() if x.strip() and not x.startswith("#")}
    ps=[p for p in sorted(a.real_root.glob("train*/**/-1/*.bin")) if p.stem not in bad]
    if len(ps)>a.max_frames: ps=[ps[i] for i in np.linspace(0,len(ps)-1,a.max_frames,dtype=int)]
    rng=np.random.default_rng(a.seed); cs=[]; counts=[]
    for p in ps:
        x=load(p); counts.append(len(x))
        if len(x)>a.points_per_frame:x=x[rng.choice(len(x),a.points_per_frame,False)]
        cs.append(x)
    x=np.concatenate(cs); r=np.linalg.norm(x,axis=1); e=np.arctan2(x[:,2],np.linalg.norm(x[:,:2],axis=1)); az=np.arctan2(x[:,1],x[:,0])
    fit=e if len(e)<=200000 else e[rng.choice(len(e),200000,False)]; ra=rings(fit,a.rings); ri=np.abs(e[:,None]-ra).argmin(1)
    rc=np.bincount(ri,minlength=a.rings); rv=rc/max(rc.max(),1); rr=np.array([np.histogram(r[ri==i],RE)[0] for i in range(a.rings)],float); rr/=np.maximum(rr.max(1,keepdims=True),1)
    ac=np.histogram(az,AE)[0].astype(float); ac/=max(ac.max(),1)
    occ=x if len(x)<=250000 else x[rng.choice(len(x),250000,False)]
    a.output.mkdir(parents=True,exist_ok=True); np.save(a.output/"ring_angles.npy",ra); np.save(a.output/"ring_valid_rate.npy",rv.astype(np.float32))
    np.save(a.output/"range_ring_keep_prob.npy",rr.astype(np.float32)); np.save(a.output/"azimuth_keep_prob.npy",ac.astype(np.float32))
    np.savez_compressed(a.output/"multiscale_occupancy.npz",**{f"voxel_{s:g}":np.unique(np.floor(occ/s).astype(np.int16),axis=0) for s in (.25,.5,1.,2.)})
    q=np.percentile(counts,(0,10,50,90,100)); m={"policy":"unlabeled train -1 only; every forbidden basename suffix excluded","frames":len(ps),"pooled_points":len(x),"rings":a.rings,
      "ring_angle_degrees":np.rad2deg(ra).tolist(),"range_edges_m":RE.tolist(),"azimuth_edges_rad":AE.tolist(),
      "point_count_distribution":dict(zip(("min","p10","median","p90","max"),map(float,q))),"source_paths":[str(p.relative_to(a.real_root)) for p in ps]}
    (a.output/"sensor_profile.json").write_text(json.dumps(m,indent=2)); print(json.dumps({k:v for k,v in m.items() if k!="source_paths"},indent=2))
if __name__=="__main__":main()
