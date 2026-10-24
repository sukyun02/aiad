"""Build composite kit caches from existing ones (numpy only, no torch).

usage: mkcache.py OUT SRC[:lap,lap,...] [SRC[:laps] ...]
  OUT and SRC are cache names under pilot/cache (without .npz). Laps are 0-based indexes inside SRC; no list = all laps.
example: mkcache.py s2_mixA data s2_scriptA      -> provided 7 laps + scripted set A
"""
import sys
from pathlib import Path
import numpy as np

CACHE = Path(__file__).resolve().parents[2] / 'cache'


def main(out, specs):
    parts = {key: [] for key in ('obs', 'act', 'lap', 'frame', 'labels')}
    offset = 0
    for spec in specs:
        name, _, chosen = spec.partition(':')
        data = np.load(CACHE / f'{name}.npz')
        laps = sorted(set(data['lap'].tolist()))
        keep_laps = [int(v) for v in chosen.split(',')] if chosen else laps
        assert set(keep_laps) <= set(laps), (name, keep_laps, laps)
        for new_index, lap in enumerate(keep_laps):
            mask = data['lap'] == lap
            for key in ('obs', 'act', 'frame', 'labels'):
                parts[key].append(data[key][mask])
            parts['lap'].append(np.full(int(mask.sum()), offset + new_index))
        offset += len(keep_laps)
    merged = {key: np.concatenate(value) for key, value in parts.items()}
    np.savez(CACHE / f'{out}.npz', **merged)
    print('cached', out, merged['obs'].shape, 'laps', offset, 'class counts', np.bincount(merged['labels'], minlength=9).tolist())


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
