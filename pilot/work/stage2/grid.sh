#!/bin/sh
# grid.sh TAG CACHE EPOCHS SAVE_EPOCHS SEED_START  -> one 4-seed kit run of the stage-2 pilot grid (cand s2_cnnA)
# SAVE_EPOCHS: comma list or "-" for none. Everything goes through the kit (slots, RAM guard, memory-capped slice).
cd /home/sukyun02/workspace/aiad/pilot || exit 1
TAG=$1; CACHE=$2; EPOCHS=$3; SAVE=$4; START=$5
[ "$SAVE" = "-" ] && SAVE_ARGS="" || SAVE_ARGS="--save-epochs $SAVE"
KIT_CACHE=$CACHE kit/k run s2_cnnA --tag "$TAG" --batch-size 256 --epochs "$EPOCHS" $SAVE_ARGS --seeds 4 --seed-start "$START" \
    --probes hud_zero,noise2 > "work/stage2/out/grid_$TAG.log" 2>&1
echo "grid $TAG exit=$? $(date +%H:%M:%S)"
