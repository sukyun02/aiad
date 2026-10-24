#!/bin/bash
W=/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad/work/rob
while [ "$(ps -eo args | grep -E 'work/rob/one.sh .*__cuda' | grep -v grep | wc -l)" -gt 1 ]; do sleep 3; done
exec $W/runall.sh $W/jobs2.tsv 5
