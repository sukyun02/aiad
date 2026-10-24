"""Second batch of checker jobs (stdlib only): own replays and robustness runs of the chosen 'human-like' setting."""
from pathlib import Path

HERE = '/home/sukyun02/workspace/aiad/pilot/work/stage2/human_check'
SP = '/home/sukyun02/workspace/aiad/pilot'
COL = '/home/sukyun02/workspace/aiad/assignment1/data/collected'
NEW = [l for l in Path(f'{HERE}/new_folders.txt').read_text().split() if l]
Path(f'{HERE}/out/rob').mkdir(parents=True, exist_ok=True)

# own replays: 2 provided human laps (H1, H6), 2 set-A laps, all new keyboard laps (snapshot in new_folders.txt)
groups = [
    ['H1=' + f'{COL}/20260923_223710_800132', 'H6=' + f'{COL}/20260923_224103_364888',
     'A_lap1=' + f'{SP}/data_s2/scriptA/lap1', 'A_lap7=' + f'{SP}/data_s2/scriptA/lap7'],
    [f'N{i + 1}={COL}/{d}' for i, d in enumerate(NEW) if i % 2 == 0],
    [f'N{i + 1}={COL}/{d}' for i, d in enumerate(NEW) if i % 2 == 1],
]
Path(f'{HERE}/jobs_myreplay.txt').write_text('\n'.join(
    f'{HERE}/myreplay.py {HERE}/out/myreplay_{k}.json ' + ' '.join(g) for k, g in enumerate(groups)) + '\n')

DH = f'{SP}/work/stage2/human/driver_h.py'
base = '--W 28 --k0 0.03 --decide commit'
lines = []
def add(name, args):
    lines.append(f'{DH} {args} --out {HERE}/out/rob/{name}.json --trace {HERE}/out/rob/{name}.npz')
# R1 / R2: the H8 recipe at ONE fixed configuration, 8 fresh seeds each
for seed in range(101, 109):
    add(f'r1_E4.0_la15_s{seed}', f'--E 4.0 --lookahead 15 --gain 1.0 --min-press 5 --press-jitter 2 --noise-seed {seed} --react 6 --react-steer current {base}')
    add(f'r2_E4.5_la15_s{seed}', f'--E 4.5 --lookahead 15 --gain 1.0 --min-press 5 --press-jitter 2 --noise-seed {seed} --react 6 --react-steer current {base}')
# R3: longer reaction delay at the chosen setting (min-press 6, E 4)
for react, la in ((9, 15), (9, 20), (12, 15), (12, 20), (12, 25)):
    add(f'r3_m6_r{react}_E4_la{la}', f'--E 4 --lookahead {la} --gain 1.0 --min-press 6 --react {react} --react-steer current {base}')
# R4: literal delay (steering state also old) with the tuned lookahead
for la in (15, 20):
    add(f'r4_m6_r6_E4_la{la}_delayed', f'--E 4 --lookahead {la} --gain 1.0 --min-press 6 --react 6 --react-steer delayed {base}')
Path(f'{HERE}/jobs_rob.txt').write_text('\n'.join(lines) + '\n')
print(len(lines), [len(g) for g in groups])
