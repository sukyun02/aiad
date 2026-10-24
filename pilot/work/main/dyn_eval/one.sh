#!/bin/sh
# one.sh CHECKPOINT KIND   -> out/<name>__<kind>.json  (one headless lap through the fixed kit; skips finished ones)
SP=/home/sukyun02/workspace/aiad/pilot
OUT=$SP/work/main/dyn_eval/out
name=$(basename "$1" .pth)
target=$OUT/${name}__$2.json
[ -s "$target" ] && exit 0
case "$2" in
    cuda) flags="" ;;
    cpu) flags="--device cpu" ;;
    hud_zero) flags="--hud zero" ;;
esac
"$SP/kit/k" eval "$1" --slot $flags > "$target.tmp" 2> "$OUT/${name}__$2.err" && mv "$target.tmp" "$target"
