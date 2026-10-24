"""Table of the stage-2 pilot grid from results/s2_cnnA.jsonl (stdlib only).  usage: tab.py [TAG ...]"""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

rows = [json.loads(line) for line in Path('/home/sukyun02/workspace/aiad/pilot/results/s2_cnnA.jsonl').read_text().splitlines() if line.strip()]
groups = defaultdict(list)
for r in rows:
    groups[(r['tag'], r['epoch'])].append(r)
want = sys.argv[1:]
clean = lambda d: d.get('status') == 'finished' and d['progress'] >= 100.0 and d['offroad_frames'] == 0
print(f"{'tag':8s} {'ep':>4s} {'n':>2s} | {'nominal clean':>13s} {'lap min-med-max (clean nominal)':>32s} {'centre':>11s} {'maxoff':>9s} | {'perturbed clean':>15s} {'not fin':>7s} {'pert lap max':>12s} | {'loss':>11s} {'acc':>5s}")
for (tag, epoch), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    if want and tag not in want:
        continue
    nominal = [r['drive'] for r in g]
    ok = [d for d in nominal if clean(d)]
    laps = [d['lap_time'] for d in ok]
    pert = [r[k] for r in g for k in r if k.startswith('drive_')]
    pert_ok = [d for d in pert if clean(d)]
    pert_laps = [d['lap_time'] for d in pert if d.get('status') == 'finished' and d.get('lap_time')]
    lap_text = f"{min(laps):.2f} {st.median(laps):.2f} {max(laps):.2f}" if laps else '-'
    centre = f"{min(d['center'] for d in ok):.0f}-{max(d['center'] for d in ok):.0f}" if ok else '-'
    maxoff = f"{min(d['max_offset'] for d in nominal):.1f}-{max(d['max_offset'] for d in nominal):.1f}"
    loss = f"{min(r['train_loss_sum'] for r in g):.1f}-{max(r['train_loss_sum'] for r in g):.1f}"
    acc = f"{st.mean(r['train_acc'] for r in g):.3f}"
    print(f"{tag:8s} {epoch:4d} {len(g):2d} | {len(ok):>6d}/{len(g):<6d} {lap_text:>32s} {centre:>11s} {maxoff:>9s} | {len(pert_ok):>7d}/{len(pert):<7d} "
          f"{sum(d.get('status') != 'finished' for d in pert):>7d} {(f'{max(pert_laps):.2f}' if pert_laps else '-'):>12s} | {loss:>11s} {acc:>5s}")
    bad = [(r['seed'], d['status'], round(d['progress'], 2), d['offroad_frames'], d.get('lap_time')) for r, d in zip(g, nominal) if not clean(d)]
    if bad:
        print(f"{'':16s}unclean nominal (seed, status, progress, offroad frames, lap): {bad}")
