"""Job list for set H5: the 8 parameter combinations of set A (../jobsA.txt) driven with --min-hold 5 (no data saved)."""
from pathlib import Path
G = Path('/home/sukyun02/workspace/aiad/pilot/work/stage2/guide')
lines = []
for index, line in enumerate(Path(G.parent / 'jobsA.txt').read_text().splitlines(), 1):
    args = line.split(' --save-dir')[0]
    lines.append(f"{G}/driver_hold.py {args} --min-hold 5 --out {G}/hold/H5_lap{index}.json --trace {G}/hold/H5_lap{index}.npz")
(G / 'jobs_hold2.txt').write_text('\n'.join(lines) + '\n')
print(len(lines)); print(lines[0])
