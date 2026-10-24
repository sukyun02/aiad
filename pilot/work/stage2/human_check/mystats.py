"""Independent key statistics straight from the saved action files (numpy only, no env, no torch).

usage: mystats.py OUT.json NAME=FOLDER [NAME=FOLDER ...]
Written by the checker; shares no code with ../human/keystats.py.
"""
import json
import sys
from pathlib import Path

import numpy as np


def rle(seq):
    """Run-length encoding -> list of [value, start, length]."""
    out = []
    for i, v in enumerate(seq):
        if out and out[-1][0] == v:
            out[-1][2] += 1
        else:
            out.append([v, i, 1])
    return out


def quart(x):
    if not x:
        return None
    a = np.asarray(x, dtype=float)
    return [float(a.min()), float(np.percentile(a, 25)), float(np.percentile(a, 50)), float(np.percentile(a, 75)), float(a.max())]


def stats(name, folder):
    folder = Path(folder)
    files = sorted(folder.glob('action_*.npy'))
    nobs = len(list(folder.glob('observation_*.npy')))
    acts = np.stack([np.load(f) for f in files])
    # contiguous numbering?
    idx = [int(f.stem.split('_')[1]) for f in files]
    contiguous = idx == list(range(len(idx)))
    steer_raw, gas_raw, brake_raw = acts[:, 0], acts[:, 1], acts[:, 2]
    steer = [(-1 if s < -0.1 else 1 if s > 0.1 else 0) for s in steer_raw]
    gas = [int(g > 0.1) for g in gas_raw]
    brake = [int(b > 0.1) for b in brake_raw]
    n = len(steer)
    hold = next((i for i, g in enumerate(gas) if g), n)
    # also: first frame with ANY key
    first_any = next((i for i in range(n) if steer[i] or gas[i] or brake[i]), n)
    runs = rle(steer)
    presses = [r for r in runs if r[0] != 0]
    plen = [r[2] for r in presses]
    gaps = [b[1] - (a[1] + a[2]) for a, b in zip(presses[:-1], presses[1:])]
    # all steer-state runs (left / right / none), with first and last run separated
    all_len = [r[2] for r in runs]
    interior = all_len[1:-1]
    # after the hold only (the hold itself is a 'none' run that merges with the first none run)
    key_tuple = list(zip(steer, gas, brake))
    key_changes = sum(1 for a, b in zip(key_tuple[:-1], key_tuple[1:]) if a != b)
    steer_changes = sum(1 for a, b in zip(steer[:-1], steer[1:]) if a != b)
    coast_after_hold = sum(1 for i in range(hold, n) if not gas[i] and not brake[i])
    short_interior = [(r[0], r[1], r[2]) for r in runs[1:-1] if r[2] < 5]
    return dict(
        name=name, folder=str(folder), frames=n, n_observation_files=nobs, contiguous=contiguous,
        action_dtype=str(acts.dtype), action_shape=list(acts.shape[1:]),
        unique_steer=sorted(set(float(v) for v in steer_raw)), unique_gas=sorted(set(float(v) for v in gas_raw)),
        unique_brake=sorted(set(float(v) for v in brake_raw)),
        start_hold=hold, first_any_key=first_any, steer_presses=len(presses),
        left_presses=sum(1 for r in presses if r[0] < 0), right_presses=sum(1 for r in presses if r[0] > 0),
        press_len=quart(plen), gap_len=quart(gaps),
        press_le2=sum(1 for v in plen if v <= 2), press_le4=sum(1 for v in plen if v <= 4), press_ge6=sum(1 for v in plen if v >= 6),
        gap_le2=sum(1 for v in gaps if v <= 2), direct_reversals=sum(1 for v in gaps if v == 0),
        key_changes=key_changes, steer_changes=steer_changes,
        coast_frames_after_hold=coast_after_hold, brake_frames=sum(brake),
        steer_share=round(sum(1 for s in steer if s) / n, 4),
        first_run=runs[0], last_run=runs[-1],
        min_interior_run=min(interior) if interior else None,
        min_interior_press=min((r[2] for r in runs[1:-1] if r[0] != 0), default=None),
        min_interior_none=min((r[2] for r in runs[1:-1] if r[0] == 0), default=None),
        max_interior_run_needed=None,
        interior_runs_shorter_than_5=short_interior[:20], n_interior_runs_shorter_than_5=len(short_interior),
        interior_run_hist={str(k): int(v) for k, v in zip(*np.unique(interior, return_counts=True))} if interior else {},
        press_lengths=plen, gaps=gaps)


if __name__ == '__main__':
    out = Path(sys.argv[1])
    res = [stats(*item.split('=', 1)) for item in sys.argv[2:]]
    out.write_text(json.dumps(res, indent=1))
    print(f"{'name':10s} {'frm':>5s} {'obs':>5s} {'hold':>4s} {'press':>5s} {'L':>3s} {'R':>3s} {'press min/p25/med/p75/max':>26s} {'gap min/p25/med/p75/max':>26s} "
          f"{'chg':>5s} {'schg':>5s} {'coast':>5s} {'brk':>4s} {'share':>6s} {'minIntRun':>9s} {'minIntPress':>11s} {'minIntNone':>10s} first/last run")
    for r in res:
        f = lambda q: '/'.join(f'{v:g}' for v in q) if q else '-'
        print(f"{r['name']:10s} {r['frames']:5d} {r['n_observation_files']:5d} {r['start_hold']:4d} {r['steer_presses']:5d} {r['left_presses']:3d} {r['right_presses']:3d} "
              f"{f(r['press_len']):>26s} {f(r['gap_len']):>26s} {r['key_changes']:5d} {r['steer_changes']:5d} {r['coast_frames_after_hold']:5d} {r['brake_frames']:4d} "
              f"{r['steer_share']:6.3f} {str(r['min_interior_run']):>9s} {str(r['min_interior_press']):>11s} {str(r['min_interior_none']):>10s} {r['first_run']} {r['last_run']}")
