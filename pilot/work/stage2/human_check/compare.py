"""Compare the checker's re-runs with the stored results of the other agent (numpy only)."""
import json
from pathlib import Path

import numpy as np

HERE = Path('/home/sukyun02/workspace/aiad/pilot/work/stage2/human_check/out')
S2 = Path('/home/sukyun02/workspace/aiad/pilot/work/stage2')
HU = S2 / 'human/out'
pairs = [
    ('eq_orig_A_lap1', S2 / 'out/A_lap1'),          # original driver re-run vs stored original
    ('eq_h_A_lap1', S2 / 'out/A_lap1'),             # driver_h (min-press 1, react 0) vs stored original
    ('eq_h_A_lap1', HU / 'eq_A_lap1'),              # driver_h re-run vs their stored driver_h run
    ('re_m6_r6_E4_la15', HU / 'follow/f_m6_r6_E4_la15'),
    ('re_m6_r6_E6.0_la15', HU / 'elimit/x_m6_r6_E6.0_la15'),
    ('re_m6_r6_E6.5_la15', HU / 'elimit/x_m6_r6_E6.5_la15'),
    ('re_m10_r0_E3_la9', HU / 'grid/g_m10_r0_E3'),
    ('re_m10_r3_E4_la12', HU / 'follow/f_m10_r3_E4_la12'),
    ('re_h8_b6', HU / 'h8/b6'),
]
report = []
for mine, ref in pairs:
    a = json.loads((HERE / f'{mine}.json').read_text())
    b = json.loads(Path(str(ref) + '.json').read_text())
    common = [k for k in a if k in b]
    diff = [k for k in common if a[k] != b[k]]
    ra, rb = np.load(HERE / f'{mine}.npz')['rows'], np.load(str(ref) + '.npz')['rows']
    same_rows = bool(ra.shape == rb.shape and np.array_equal(ra, rb))
    keys_same = None
    za, zb = np.load(HERE / f'{mine}.npz'), np.load(str(ref) + '.npz')
    if 'keys' in za.files and 'keys' in zb.files:
        keys_same = bool(np.array_equal(za['keys'], zb['keys']))
    item = dict(mine=mine, ref=str(ref), n_common_fields=len(common), differing_fields=diff, rows_shape=[list(ra.shape), list(rb.shape)],
                rows_bit_identical=same_rows, keys_identical=keys_same,
                mine_summary={k: a.get(k) for k in ('status', 'lap_time', 'progress', 'center', 'frames', 'max_abs_offset', 'key_changes',
                                                    'steer_presses', 'offroad_frames', 'first_offroad_along', 'clean', 'action_counts')},
                ref_summary={k: b.get(k) for k in ('status', 'lap_time', 'progress', 'center', 'frames', 'max_abs_offset', 'key_changes',
                                                   'steer_presses', 'offroad_frames', 'first_offroad_along', 'clean', 'action_counts')})
    report.append(item)
    print(json.dumps(item))
(HERE / 'compare.json').write_text(json.dumps(report, indent=1))

# saved-data check: H8/lapN vs H8_raw/bX (actions identical? observations identical for a sample?) and format vs a collect.py folder
SP = Path('/home/sukyun02/workspace/aiad/pilot/data_s2')
mapping = dict(line.split() for line in (HU / 'h8/H8_mapping.txt').read_text().split('\n') if line.strip())
fmt = []
human = Path('/home/sukyun02/workspace/aiad/assignment1/data/collected/20260923_223710_800132')
ho, ha = np.load(human / 'observation_00100.npy'), np.load(human / 'action_00100.npy')
for lap, raw in mapping.items():
    fa = sorted((SP / 'H8' / lap).glob('action_*.npy'))
    fb = sorted((SP / 'H8_raw' / raw).glob('action_*.npy'))
    acts_equal = len(fa) == len(fb) and all(np.array_equal(np.load(x), np.load(y)) for x, y in zip(fa, fb))
    sample = list(range(0, len(fa), 97))
    obs_equal = all(np.array_equal(np.load(SP / 'H8' / lap / f'observation_{i:05d}.npy'), np.load(SP / 'H8_raw' / raw / f'observation_{i:05d}.npy'))
                    for i in sample)
    o = np.load(SP / 'H8' / lap / 'observation_00100.npy')
    a = np.load(SP / 'H8' / lap / 'action_00100.npy')
    # the trace 'keys' of the candidate must equal the saved steer column
    keys = np.load(HU / f'h8/{raw}.npz')['keys']
    saved = np.array([np.load(x)[0] for x in fa], dtype=np.float32)
    fmt.append(dict(lap=lap, raw=raw, n_actions=len(fa), actions_equal_to_raw=bool(acts_equal), obs_sample_equal_to_raw=bool(obs_equal),
                    n_obs_sampled=len(sample), obs_dtype=str(o.dtype), obs_shape=list(o.shape), act_dtype=str(a.dtype), act_shape=list(a.shape),
                    human_obs_dtype=str(ho.dtype), human_obs_shape=list(ho.shape), human_act_dtype=str(ha.dtype), human_act_shape=list(ha.shape),
                    saved_steer_equals_trace_keys=bool(np.array_equal(keys, saved))))
    print(json.dumps(fmt[-1]))
(HERE / 'format.json').write_text(json.dumps(fmt, indent=1))
