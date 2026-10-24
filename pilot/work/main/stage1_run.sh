#!/bin/sh
# stage1_run.sh N  -> one round of assignment1/docs/01_network_structure_plan.md, steps 5.3-5.5, for run N.
# Run from assignment1/. Everything goes through the kit and runs one command after another.
set -u
N=$1
K=../pilot/kit/k
LOG=docs/logs

echo "=== run $N: training (5.3) ==="
free -h | sed -n 2p
$K job -- .venv/bin/python scripts/train.py configs/stage1.yaml > "$LOG/run$N.log" 2>&1
echo "train exit=$?"
grep -E "^설정:|새 모델로|기존 모델에서|전체 학습 대상|모델 저장 완료|제출할 구조 파일" "$LOG/run$N.log" | cut -c1-210
grep -E "^Epoch +(1|30|50|100)[^0-9]" "$LOG/run$N.log" | cut -c1-48
SAVED=$(grep "모델 저장 완료" "$LOG/run$N.log" | sed 's/.*: //')
[ -n "$SAVED" ] || { echo "NO MODEL SAVED"; tail -5 "$LOG/run$N.log"; exit 1; }
NAME=$(basename "$SAVED" .pth)
M=models/$NAME

echo "=== run $N: kit drive (5.4) $M ==="
$K eval "$M.pth" --slot > "$LOG/eval_$NAME.json" 2> "$LOG/eval_$NAME.err"
python3 - "$LOG/eval_$NAME.json" <<'E'
import json, sys
d = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
print({k: d[k] for k in ('status', 'lap_time', 'progress', 'center', 'max_offset', 'offroad_frames')})
ok = (d['status'] == 'finished' and d['progress'] >= 100.0 and d['offroad_frames'] == 0
      and round(d['lap_time'], 2) <= 43.6 and d['center'] >= 30 and d['max_offset'] < 6.67)
print('5.4 verdict:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
E
PASS=$?

echo "=== run $N: official evaluation code (5.4) ==="
$K official "$M.pth" > "$LOG/official_$NAME.txt" 2>&1
grep -E "주행 결과|실행 오류|주행 종료" "$LOG/official_$NAME.txt"

if [ "$PASS" -eq 0 ]; then
    echo "=== run $N: gate (5.5) ==="
    $K gate "$M.pth" --json "$LOG/gate_$NAME.json" | tee "$LOG/gate_$NAME.txt"
else
    echo "=== run $N: gate skipped (5.4 not passed) ==="
fi
echo "=== run $N done: $M ==="
