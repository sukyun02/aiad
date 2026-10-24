"""Trimmed cache for partial laps: drop the frames inside the bends that a lap drove on the centre line (numpy only).

usage: mktrim.py SET PREFIX     e.g. mktrim.py P10 P_lap   -> cache/s2_P10trim.npz from cache/s2_P10.npz,
       out/<PREFIX><n>.npz (trace, rows[:,0] = along) and lines/<PREFIX><n>.npz (off_windows)
"""
import sys
from pathlib import Path
import numpy as np

SP = Path(__file__).resolve().parents[2]
name, prefix = sys.argv[1], sys.argv[2]
data = np.load(SP / 'cache' / f's2_{name}.npz')
folders = sorted(p.name for p in (SP / 'data_s2' / name).iterdir() if p.is_dir())      # same order as `kit cache`
keep = np.ones(len(data['lap']), dtype=bool)
here = Path(__file__).resolve().parent
for index, folder in enumerate(folders):
    suffix = folder[3:]                                       # lap7 -> 7, lap07 -> 07
    along = np.load(here / 'out' / f'{prefix}{suffix}.npz')['rows'][:, 0]
    windows = np.load(here / 'lines' / f'{prefix}{suffix}.npz')['off_windows']
    mask = data['lap'] == index
    frames = data['frame'][mask]
    assert len(along) == mask.sum() and frames.max() == len(along) - 1, (folder, len(along), mask.sum())
    drop = np.zeros(len(along), dtype=bool)
    for a, b in windows:
        drop |= (along >= a) & (along < b)
    keep[np.flatnonzero(mask)] = ~drop[frames]
    print(folder, 'frames', len(along), 'dropped', int(drop.sum()), 'windows', np.round(windows).astype(int).tolist())
out = {key: data[key][keep] for key in data.files}
np.savez(SP / 'cache' / f's2_{name}trim.npz', **out)
print('cached', f's2_{name}trim', out['obs'].shape, 'of', len(keep), 'class counts', np.bincount(out['labels'], minlength=9).tolist())
