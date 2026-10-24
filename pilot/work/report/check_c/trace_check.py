"""numpy only: off-road runs of the final model's trace, and key-press statistics of the 21 training laps."""
import json, glob, os
import numpy as np
R='/home/sukyun02/workspace/aiad/pilot/work/report/'
A='/home/sukyun02/workspace/aiad/assignment1/data/'
out={}
for name in ('final_model','stage1_model'):
    z=np.load(R+name+'/trace.npz')
    rows=z['rows']
    off=rows[:,2]>0.5; cls=rows[:,3].astype(int); speed=rows[:,4]; along=rows[:,5]; err=rows[:,6]
    runs=[]; start=None
    for i,f in enumerate(np.r_[off,False]):
        if f and start is None: start=i
        elif (not f) and start is not None:
            runs.append((int(start), int(i-start), round(float(along[start]),1), round(float(along[i-1]),1), round(float(err[start:i].max()),2)))
            start=None
    # steering class runs
    steer=np.isin(cls,(3,4,5,6,7,8))
    out[name]=dict(frames=len(rows), off=int(off.sum()), pct=round(100*off.mean(),2), runs=runs,
                   longest=max([r[1] for r in runs]) if runs else 0,
                   class_counts=np.bincount(cls,minlength=9).tolist(),
                   first_gas=int(np.argmax(np.isin(cls,(1,5,6)))),
                   class0_after_first_gas=int((cls[int(np.argmax(np.isin(cls,(1,5,6)))):]==0).sum()),
                   half_width=float(z['half_width']), centre_len=float(np.linalg.norm(np.roll(z['centre'],-1,axis=0)-z['centre'],axis=1).sum()),
                   max_err=round(float(err.max()),2), mean_err=round(float(err.mean()),3),
                   speed_max=round(float(speed.max()),2), first_move_frame=int(np.argmax(speed>0.5)),
                   along_scored_off=round(float(sum(r[3]-r[2] for r in runs)),1))
print(json.dumps(out, indent=1))

def classes(a):
    steer,gas,brake=a[:,0],a[:,1],a[:,2]
    L=steer<-0.1; Rr=steer>0.1; G=gas>0.1; B=brake>0.1
    c=np.zeros(len(a),int)
    c[G&~L&~Rr]=1; c[B&~G&~L&~Rr]=2; c[L&~G&~B]=3; c[Rr&~G&~B]=4; c[L&G]=5; c[Rr&G]=6; c[L&B&~G]=7; c[Rr&B&~G]=8
    return c
tot=np.zeros(9,int); press_len=[]; hold=[]; nlap=0; presses_per_lap=[]
c0_after=0
for sub in ('stage2','stage2_over41'):
    for lap in sorted(os.listdir(A+sub)):
        files=sorted(glob.glob(A+sub+'/'+lap+'/action_*.npy'))
        a=np.stack([np.load(f) for f in files]); c=classes(a); nlap+=1
        tot+=np.bincount(c,minlength=9)
        first=int(np.argmax(c!=0)); hold.append(first); c0_after+=int((c[first:]==0).sum())
        s=np.isin(c,(3,4,5,6,7,8)); n=0; k=0
        for v in np.r_[s,False]:
            if v: n+=1
            elif n: press_len.append(n); n=0; k+=1
        presses_per_lap.append(k)
press_len=np.array(press_len)
print('laps',nlap,'frames',int(tot.sum()),'class counts',tot.tolist(), 'pct',np.round(100*tot/tot.sum(),2).tolist())
print('start hold per lap',hold,'sum',sum(hold),'class0 after start',c0_after)
print('steering presses: n',len(press_len),'per lap mean',np.mean(presses_per_lap),'len median',float(np.median(press_len)),'mean',round(float(press_len.mean()),2),'p90',float(np.percentile(press_len,90)),'max',int(press_len.max()), 'share<=10 frames', round(float((press_len<=10).mean()),3))
# provided 7 laps
tot=np.zeros(9,int); press_len=[]; hold=[]
for lap in sorted(os.listdir(A+'collected')):
    files=sorted(glob.glob(A+'collected/'+lap+'/action_*.npy'))
    if not files: continue
    a=np.stack([np.load(f) for f in files]); c=classes(a); tot+=np.bincount(c,minlength=9)
    first=int(np.argmax(c!=0)); hold.append(first)
    s=np.isin(c,(3,4,5,6,7,8)); n=0
    for v in np.r_[s,False]:
        if v: n+=1
        elif n: press_len.append(n); n=0
press_len=np.array(press_len)
print('provided: frames',int(tot.sum()),'class counts',tot.tolist(),'pct',np.round(100*tot/tot.sum(),2).tolist(),'hold',hold)
print('provided steering presses: n',len(press_len),'median',float(np.median(press_len)),'mean',round(float(press_len.mean()),2))
