"""Write jobs5.txt: three processes that re-drive the human-rate traces of guide/hold from their key sequences."""
from pathlib import Path

HERE = Path('/home/sukyun02/workspace/aiad/pilot/work/stage2/guide_check')
HOLD = Path('/home/sukyun02/workspace/aiad/pilot/work/stage2/guide/hold')
groups = [['H5_lap1', 'H5_lap2', 'H5_lap3', 'H5_lap4'], ['H5_lap5', 'H5_lap6', 'H5_lap7', 'H5_lap8'],
          ['N5', 'N6', 'N8', 'E3_N5', 'E5_N5', 'E0_N5']]
lines = []
for i, g in enumerate(groups, 1):
    lines.append(f"{HERE / 'check5_trace_replay.py'} {HERE / f'trace_replay_{i}.jsonl'} " + ' '.join(f'{n}={HOLD / (n + ".npz")}' for n in g))
(HERE / 'jobs5.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
