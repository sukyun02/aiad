import json, os, statistics as st
A='/home/sukyun02/workspace/aiad/assignment1/data/'
d=json.load(open('/home/sukyun02/workspace/aiad/pilot/work/report/laps_metrics.json'))
laps=d['laps']
s2=sorted(os.listdir(A+'stage2')); s2o=sorted(os.listdir(A+'stage2_over41'))
print(len(s2), len(s2o), 'overlap', set(s2)&set(s2o))
kept=set(s2)|set(s2o)
print('union', len(kept), 'all in metrics', all(k in laps for k in kept))
K=[laps[k] for k in sorted(kept)]
D=[(k,laps[k]) for k in sorted(laps) if k not in kept]
print('dropped (6?)', len(D))
for k,v in D: print('  ', k, v['frames'], round(v['lap_time'],2), v['center'], v['progress'], v['off_road_frames'], v['start_hold'])
def s(key, rs):
    xs=[r[key] for r in rs]; return (round(min(xs),2), round(max(xs),2), round(st.mean(xs),3), round(sum(xs),2))
for key in ['frames','lap_time','effective_lap','center','progress','start_hold','off_road_frames','brake_frames','no_gas_after_start','max_abs_offset']:
    print(key, s(key,K))
print('off-road fraction of training frames', sum(r['off_road_frames'] for r in K)/sum(r['frames'] for r in K))
print('sum start_hold', sum(r['start_hold'] for r in K))
print('kept centers sorted', sorted(r['center'] for r in K))
print('n kept with center<30', sum(r['center']<30 for r in K), ' center<=28', sum(r['center']<=28 for r in K))
print('n kept with lap>=42', sum(r['lap_time']>=42 for r in K))
print('n kept with lap<=42 and center>=30 (meet both targets)', sum(r['lap_time']<=42 and r['center']>=30 for r in K))
print('kept progress sorted', sorted(r['progress'] for r in K))
print('kept offroad sorted', sorted(r['off_road_frames'] for r in K))
print('kept laptime sorted', sorted(round(r['lap_time'],2) for r in K))
# frames actual on disk
import glob
tot=0
for sub in ['stage2','stage2_over41']:
    for f in sorted(os.listdir(A+sub)):
        n=len(glob.glob(A+sub+'/'+f+'/observation_*.npy')); na=len(glob.glob(A+sub+'/'+f+'/action_*.npy'))
        tot+=n
        if n!=laps[f]['frames'] or n!=na: print('MISMATCH', f, n, na, laps[f]['frames'])
print('disk frames total', tot)
# provided
tot=0
for f in sorted(os.listdir(A+'collected')):
    n=len(glob.glob(A+'collected/'+f+'/observation_*.npy')); tot+=n; print(f, n, n/50)
print('provided total', tot)
# correlation lap vs center/progress among 27 and among kept
def corr(a,b):
    ma,mb=st.mean(a),st.mean(b)
    num=sum((x-ma)*(y-mb) for x,y in zip(a,b)); den=(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))**.5
    return num/den
allv=list(laps.values())
for nm,rs in [('27',allv),('21',K)]:
    print(nm,'corr lap-center',round(corr([r['lap_time'] for r in rs],[r['center'] for r in rs]),3),'lap-progress',round(corr([r['lap_time'] for r in rs],[r['progress'] for r in rs]),3), 'center-progress', round(corr([r['center'] for r in rs],[r['progress'] for r in rs]),3))
# sections with off_road among kept
from collections import Counter
c=Counter(); cf=Counter()
for r in K:
    for sct in r['sections']:
        if sct['off_road']>0: c[sct['name']]+=1; cf[sct['name']]+=sct['off_road']
print('sections with off-road (laps of 21):', dict(c)); print('offroad frames by section', dict(cf))
