"""Compare driver_h.py (--min-press 1 --react 0) with the original driver's stored results: json fields and trace rows."""
import json
import sys
from pathlib import Path
import numpy as np
REF = Path('/home/sukyun02/workspace/aiad/pilot/work/stage2/out')
OUT = Path('/home/sukyun02/workspace/aiad/pilot/work/stage2/human/out')
report = []
for name in sys.argv[1:]:
    ref = json.loads((REF / f'{name}.json').read_text())
    new = json.loads((OUT / f'eq_{name}.json').read_text())
    same_fields = {k: ref[k] == new[k] for k in ref if k in new}
    a, b = np.load(REF / f'{name}.npz')['rows'], np.load(OUT / f'eq_{name}.npz')['rows']
    rows_equal = bool(a.shape == b.shape and np.array_equal(a, b))
    item = dict(name=name, ref_lap=ref['lap_time'], new_lap=new['lap_time'], ref_counts=ref['action_counts'], new_counts=new['action_counts'],
                all_common_json_fields_equal=all(same_fields.values()), differing_fields=[k for k, v in same_fields.items() if not v],
                trace_shape=[list(a.shape), list(b.shape)], trace_rows_bit_identical=rows_equal)
    report.append(item)
    print(json.dumps(item))
(OUT / 'eqcheck.json').write_text(json.dumps(report, indent=1))
