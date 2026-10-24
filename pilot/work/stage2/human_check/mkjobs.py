"""Checker job lists (stdlib only). Run through the kit like everything else."""
from pathlib import Path

HERE = '/home/sukyun02/workspace/aiad/pilot/work/stage2/human_check'
SP = '/home/sukyun02/workspace/aiad/pilot'
COL = '/home/sukyun02/workspace/aiad/assignment1/data/collected'
HUM = ['20260923_223710_800132', '20260923_223755_475813', '20260923_223842_371887', '20260923_223934_463535',
       '20260923_224019_106700', '20260923_224103_364888', '20260923_224147_693729']
NEW = sorted(p.name for p in Path(COL).glob('20261006_*'))

# 1) key statistics from the raw action files
items = [f'H{i + 1}={COL}/{d}' for i, d in enumerate(HUM)]
items += [f'A_lap{i}={SP}/data_s2/scriptA/lap{i}' for i in range(1, 9)]
items += [f'H8_lap{i}={SP}/data_s2/H8/lap{i}' for i in range(1, 9)]
raw = sorted(p.name for p in Path(f'{SP}/data_s2/H8_raw').iterdir())
items += [f'raw_{d}={SP}/data_s2/H8_raw/{d}' for d in raw]
items += [f'N{i + 1}={COL}/{d}' for i, d in enumerate(NEW)]
Path(f'{HERE}/jobs_stats.txt').write_text(f'{HERE}/mystats.py {HERE}/out/mystats.json ' + ' '.join(items) + '\n')

# 3) laps.py replays: H8 laps in 4 jobs of 2, new human laps in 4 jobs of 2
lines = []
for k in range(4):
    a, b = 2 * k + 1, 2 * k + 2
    lines.append(f'{SP}/work/stage2/laps.py {SP}/data_s2/H8/lap{a} {SP}/data_s2/H8/lap{b} --json {HERE}/out/laps_H8_{a}_{b}.json')
Path(f'{HERE}/jobs_laps_h8.txt').write_text('\n'.join(lines) + '\n')
lines = []
for k in range(0, len(NEW), 2):
    pair = NEW[k:k + 2]
    lines.append(f'{SP}/work/stage2/laps.py ' + ' '.join(f'{COL}/{d}' for d in pair) + f' --json {HERE}/out/laps_new_{k // 2}.json')
Path(f'{HERE}/jobs_laps_new.txt').write_text('\n'.join(lines) + '\n')
Path(f'{HERE}/new_folders.txt').write_text('\n'.join(NEW) + '\n')

# 2/4) driver re-runs: equivalence (A_lap1 arguments) with the ORIGINAL driver and with driver_h, plus sweep rows
D0 = f'{SP}/work/stage2/driver.py'
DH = f'{SP}/work/stage2/human/driver_h.py'
common = '--W 28 --k0 0.03 --react-steer current --decide commit'
runs = [
    (D0, '--E 4 --W 28 --k0 0.03 --lookahead 9 --gain 1.0', 'eq_orig_A_lap1'),
    (DH, '--E 4 --W 28 --k0 0.03 --lookahead 9 --gain 1.0 --min-press 1 --react 0', 'eq_h_A_lap1'),
    (DH, f'--E 4 --lookahead 15 --gain 1.0 --min-press 6 --react 6 {common}', 're_m6_r6_E4_la15'),
    (DH, f'--E 6.0 --lookahead 15 --gain 1.0 --min-press 6 --react 6 {common}', 're_m6_r6_E6.0_la15'),
    (DH, f'--E 6.5 --lookahead 15 --gain 1.0 --min-press 6 --react 6 {common}', 're_m6_r6_E6.5_la15'),
    (DH, f'--E 3 --lookahead 9 --gain 1.0 --min-press 10 --react 0 {common}', 're_m10_r0_E3_la9'),
    (DH, f'--E 4 --lookahead 12 --gain 1.0 --min-press 10 --react 3 {common}', 're_m10_r3_E4_la12'),
    (DH, f'--E 4.5 --lookahead 16 --gain 1.05 --min-press 5 --press-jitter 2 --noise-seed 6 --W 28 --k0 0.03 --react 6 --react-steer current --decide commit', 're_h8_b6'),
]
Path(f'{HERE}/jobs_rerun.txt').write_text('\n'.join(
    f'{script} {args} --out {HERE}/out/{name}.json --trace {HERE}/out/{name}.npz' for script, args, name in runs) + '\n')
print('ok', len(items), len(NEW))
