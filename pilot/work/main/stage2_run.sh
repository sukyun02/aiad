#!/bin/sh
# stage2_run.sh N  -> one round of assignment1/docs/02_inside_line_data_plan.md, steps 5.5-5.7, for run N.
# Run from assignment1/. Same flow as stage1_run.sh; the config and the lap limit differ.
# Everything goes through the kit and runs one command after another.
set -u
N=$1
CONFIG=${CONFIG:-configs/stage2.yaml}
LAP_MAX=${LAP_MAX:-42.00}
K=../pilot/kit/k
LOG=docs/logs

echo "=== s2 run $N: training ($CONFIG) ==="
free -h | sed -n 2p
$K job -- .venv/bin/python scripts/train.py "$CONFIG" > "$LOG/s2_run$N.log" 2>&1
echo "train exit=$?"
grep -E "^설정:|새 모델로|기존 모델에서|전체 학습 대상|모델 저장 완료|제출할 구조 파일" "$LOG/s2_run$N.log" | cut -c1-210
grep -E "^Epoch +(1|30|50|100|150|200)[^0-9]" "$LOG/s2_run$N.log" | cut -c1-48
SAVED=$(grep "모델 저장 완료" "$LOG/s2_run$N.log" | sed 's/.*: //')
[ -n "$SAVED" ] || { echo "NO MODEL SAVED"; tail -5 "$LOG/s2_run$N.log"; exit 1; }
NAME=$(basename "$SAVED" .pth)
M=models/$NAME

echo "=== s2 run $N: kit drive $M ==="
$K eval "$M.pth" --slot > "$LOG/eval_$NAME.json" 2> "$LOG/eval_$NAME.err"
python3 - "$LOG/eval_$NAME.json" "$LAP_MAX" <<'E'
import json, sys
d = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
lap_max = float(sys.argv[2])
print({k: d[k] for k in ('status', 'lap_time', 'progress', 'center', 'max_offset', 'offroad_frames')})
clean = d['status'] == 'finished' and d['progress'] >= 100.0 and d['offroad_frames'] == 0 and d['max_offset'] < 6.67
fast = clean and round(d['lap_time'], 2) <= lap_max
ok = fast and d['center'] >= 30
print('clean lap:', 'yes' if clean else 'no', '| lap <=', lap_max, ':', 'yes' if fast else 'no',
      '| centre >= 30:', 'yes' if d['center'] >= 30 else 'no', '| drive verdict:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
E
PASS=$?

echo "=== s2 run $N: official evaluation code ==="
$K official "$M.pth" > "$LOG/official_$NAME.txt" 2>&1
grep -E "주행 결과|실행 오류|주행 종료" "$LOG/official_$NAME.txt"

if [ "$PASS" -eq 0 ]; then
    echo "=== s2 run $N: gate ==="
    $K gate "$M.pth" --json "$LOG/gate_$NAME.json" | tee "$LOG/gate_$NAME.txt"
    # lap times of the 8 gate laps against the limit (the gate itself only judges clean / not clean)
    python3 - "$LOG/gate_$NAME.json" "$LAP_MAX" <<'E' | tee -a "$LOG/gate_$NAME.txt"
import json, sys
results = json.load(open(sys.argv[1]))
lap_max = float(sys.argv[2])
laps = {kind: r.get('lap_time') for kind, r in results.items() if r.get('status') == 'finished' and r.get('lap_time') is not None}
slow = {kind: round(lap, 2) for kind, lap in laps.items() if round(lap, 2) > lap_max}
print(f"gate lap times: max {max(laps.values()):.2f} over {len(laps)} finished laps | over {lap_max}: {slow if slow else 'none'}"
      f" | lap check {'PASS' if not slow else 'FAIL'}")
E
else
    echo "=== s2 run $N: gate skipped (drive verdict not PASS) ==="
fi
echo "=== s2 run $N done: $M ==="
