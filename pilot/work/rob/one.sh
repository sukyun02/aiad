#!/bin/sh
# args: id path tool opts
S=/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad
W=$S/work/rob
id="$1"; path="$2"; tool="$3"; opts="$4"
[ -s "$W/out/$id.json" ] && exit 0
if [ "$tool" = k ]; then
  $S/kit/k eval "$path" $opts --slot --trace "$W/trace/$id.json" > "$W/out/$id.tmp" 2> "$W/out/$id.err"
else
  /home/sukyun02/workspace/aiad/assignment1/.venv/bin/python $W/rdrive.py "$path" $opts --trace "$W/trace/$id.json" > "$W/out/$id.tmp" 2> "$W/out/$id.err"
fi
[ $? -eq 0 ] && tail -n 1 "$W/out/$id.tmp" > "$W/out/$id.json"
rm -f "$W/out/$id.tmp"
[ -s "$W/out/$id.err" ] || rm -f "$W/out/$id.err"
exit 0
