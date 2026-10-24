#!/bin/bash
# usage: runall.sh JOBFILE PAR
W=/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad/work/rob
tr '\t' '\n' < $1 | xargs -d '\n' -n 4 -P ${2:-6} $W/one.sh
echo DONE > $1.done
