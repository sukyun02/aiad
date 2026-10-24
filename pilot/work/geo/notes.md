# geo lens notes (running log)
Started 2026-10-06. Specs in work/geo/*.spec, applied with work/geo/mk.py onto `K new geo_X --from cnnA`.
Plan (6 K runs max, all --batch-size 256 --epochs 100 --save-epochs 25,50 --seeds 4 --probes hud_zero, to be directly
comparable with cnnA/b256_e100 rows at e25/e50/e100):
 R1 geo_map8 (5/2,3/2,3/2,3/1; 32ch -> 32x8x8 flatten)      R2 geo_gap (cnnA + AvgPool2d(4) -> 64 feats; no-geometry control)
 R3 geo_dqn  (8/4,4/2,3/1 -> 32x8x8, coarse first stride)    R4 geo_wide (strides 2,2,(2,1),(2,1) -> 32x4x18)
 R5/R6: decided after R1-R4 (dilation variant and/or seeds 4-7 confirmation of the best).
All four are PRIMARY (only the two Sequentials changed; K check: helper_code_before_class 0, changes_elsewhere 0).
