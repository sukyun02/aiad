import json, os
d=json.load(open('/home/sukyun02/workspace/aiad/pilot/work/report/laps_metrics.json'))
laps=d['laps']
print('coverage keys', type(d['coverage']), (list(d['coverage'].keys())[:20] if isinstance(d['coverage'],dict) else d['coverage'][:3]))
stage2=sorted(os.listdir('/home/sukyun02/workspace/aiad/assignment1/data/stage2'))
print('stage2 folders', len(stage2))
print('laps in metrics', len(laps))
rows=[]
for k in sorted(laps):
    v=laps[k]
    rows.append((k, v['frames'], v['lap_time'], v.get('effective_lap'), v['center'], v['progress'], v['start_hold'], v['off_road_frames'], v['brake_frames'], v['no_gas_after_start'], v['max_abs_offset'], k in stage2))
for r in rows: print(r)
kept=[r for r in rows if r[-1]]
drop=[r for r in rows if not r[-1]]
print('kept', len(kept), 'dropped', len(drop))
import statistics as st
def rng(i, rs): 
    xs=[r[i] for r in rs]; return min(xs), max(xs), round(st.mean(xs),3), sum(xs)
for name,i in [('frames',1),('lap',2),('center',4),('progress',5),('start_hold',6),('off_road',7),('brake',8),('nogas',9)]:
    print(name, 'kept', rng(i,kept), 'dropped', rng(i,drop) if drop else None)
print('missing in stage2 from metrics:', [s for s in stage2 if s not in laps])
print('dropped center values', sorted(r[4] for r in drop))
print('kept center values', sorted(r[4] for r in kept))
print('kept lap', sorted(r[2] for r in kept))
print('kept progress', sorted(r[5] for r in kept))
print('kept offroad', sorted(r[7] for r in kept))
print('kept start_hold', sorted(r[6] for r in kept), sum(r[6] for r in kept))
print('all 27 lap', sorted(r[2] for r in rows))
# off-road fraction of training data
print('off-road fraction kept', sum(r[7] for r in kept)/sum(r[1] for r in kept))
print(json.dumps(d['coverage'], ensure_ascii=False)[:3000])
