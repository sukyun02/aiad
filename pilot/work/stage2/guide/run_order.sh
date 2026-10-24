#!/bin/sh
# Order in which the guide was produced (every python process through the kit). Not meant to be run blindly:
# the laps under hold/ and centre_lap/ already exist, and driver.py --save-dir refuses to overwrite a folder.
K=/home/sukyun02/workspace/aiad/pilot/kit/k
PY=/home/sukyun02/workspace/aiad/assignment1/.venv/bin/python
G=/home/sukyun02/workspace/aiad/pilot/work/stage2/guide
xargs -P 3 -L 1 $K job -- $PY < $G/jobs_prep.txt        # replay the 7 human laps (human/), centre-line lap with data (centre_lap/)
$K job -- $PY $G/analyze.py > $G/analysis_output.txt     # corner_numbers.json, frames_A1.json, frames_centre.json
xargs -P 2 -L 1 $K job -- $PY < $G/jobs_shots.txt       # window replay + screenshots (shots/), replay_A1.*, replay_centre.*
$K job -- $PY $G/trackmap.py                            # track_map.png, kerbs.json
$K job -- $PY $G/mk_hold_jobs.py                        # jobs_hold2.txt
xargs -P 4 -L 1 $K job -- $PY < $G/jobs_hold1.txt       # human-rate key scan on the lap-1 line (hold/N*.json)
xargs -P 4 -L 1 $K job -- $PY < $G/jobs_hold2.txt       # set H5 (hold/H5_lap*.json)
xargs -P 4 -L 1 $K job -- $PY < $G/jobs_hold3.txt       # depth scan with human-rate keys (hold/E*_N*.json)
$K job -- $PY $G/analyze.py > $G/analysis_output.txt     # again: now with kerb distances and set H5
$K job -- $PY $G/sheets.py                              # sheet_*.png
$K job -- $PY $G/explore2.py > $G/track_profile.txt      # off-centre regions per lap, turn angle per 10 units
