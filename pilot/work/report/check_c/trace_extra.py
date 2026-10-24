"""numpy only: off-road runs of the two extra models and of the final model for comparison."""
import json
import numpy as np
R='/home/sukyun02/workspace/aiad/pilot/work/report/'
CORNERS = [('C1', 45.5, 77.0), ('C2', 248.5, 273.0), ('C3', 381.5, 399.0), ('C4', 518.0, 549.5), ('C5', 609.0, 633.5),
           ('C6', 728.0, 738.5), ('C7', 749.0, 766.5), ('C8', 879.0, 913.5)]
for name in ('final_model','m_20261006_232759','m_20261006_233500'):
    z=np.load(R+name+'/trace.npz'); rows=z['rows']
    off=rows[:,2]>0.5; cls=rows[:,3].astype(int); speed=rows[:,4]; along=rows[:,5]; err=rows[:,6]
    runs=[]; start=None
    for i,f in enumerate(np.r_[off,False]):
        if f and start is None: start=i
        elif (not f) and start is not None:
            runs.append((int(start), int(i-start), round(float(along[start]),1), round(float(along[i-1]),1), round(float(err[start:i].max()),2)))
            start=None
    print('==',name,'frames',len(rows),'off',int(off.sum()),'runs (start, len, along from, to, max err):',runs)
    print('   class counts',np.bincount(cls,minlength=9).tolist(),'first gas',int(np.argmax(np.isin(cls,(1,5,6)))))
    print('   last frame: along',round(float(along[-1]),1),'err',round(float(err[-1]),2),'speed',round(float(speed[-1]),2),'cls last 60',np.bincount(cls[-60:],minlength=9).tolist())
    last=runs[-1]
    s=last[0]
    print('   last run: speed at entry',round(float(speed[s]),2),'min speed in run',round(float(speed[s:s+last[1]].min()),2),'speed at end',round(float(speed[s+last[1]-1]),2), 'classes in run',np.bincount(cls[s:s+last[1]],minlength=9).tolist())
    # position where the final model enters its last off-road run vs this one
