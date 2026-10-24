# LENS hud notes (running log)
- [start] Plan: 6 K runs. Existing cnnA/cnnA_crop b256_e100 (seeds 0-3, ckpts e25/e50/e100) reused; I add seeds 4-7 with the identical config
  (--batch-size 256 --epochs 100 --save-epochs 25,50 --probes hud_zero,hud_shift,noise2) under hud_a (= byte copy of cnnA) and hud_crop (= byte copy of cnnA_crop).
  Remaining 4 runs: EXTENDED clean-sensor variant without crop (8 seeds) and with crop (8 seeds).
- run1: hud_a  tag b256_e100_s4  seeds 4-7
- run2: hud_crop tag b256_e100_s4 seeds 4-7
- [pixel analysis, work/hud/feat.py pix3.py verify.py] MEASURED on all 15067 cached frames:
  * digits rows 87-91, cols 11-27, white (R=G=B); bars rows 92-93 only (speed cols 12-14, ABS cols 16-26). Row 93 is exactly constant (182) at cruise in all 5 read columns;
    row 92 cols 22,24 varies with the WIDTH of the digit text box (text background hides the bar top) -> rows 92:94 is not fully clean, row 93 alone is.
  * original features at cruise: speed 0.227-0.700 std 0.108 (true bar would be constant) ; abs std 0.108-0.147; corr with digit column sums 0.998-1.000. steer/gyro unaffected.
  * digits change on ~85 % of frames (1812-1857 of ~2150 per lap), 7866 unique patterns in 15067 frames, only 310 of lap0's 1752 patterns occur in lap1 -> digits depend on time as well as tiles (per-frame fingerprint).
  * cleaned (row 93) features: 0 change under hud_zero / hud_shift / digit swap; constant 0.714 at cruise (0 of 14017 cruise frames deviate); ramp 0 -> 0.714 within frames ~20-36.
    All 152 label-0 frames have clean speed == 0 (159 frames with speed==0: 152 label 0, 7 label 1) -> possible "speed==0 -> do nothing" shortcut (standstill deadlock risk) for the EXTENDED clean-speed variant.
  * KEY: cnnA's conv (k5s2,k3s2,k3s2,k3s2, no padding) only covers input rows 0..80 and cols 0..80 -> the conv NEVER sees the HUD (rows 84-95). cnnA_crop is therefore the same function as cnnA
    (random-init logits: digits blanked change identical 0.012766 for both; hud_cs with un-cropped conv: 0.000000). cnnA vs cnnA_crop is an A/A comparison = noise floor.
- revised plan for remaining 4 runs: hud_sg (PRIMARY: cnnA + CircularPad1d((2,0)) + ConstantPad1d((0,-7)) in fc = keep steer,gyro, drop the 5 contaminated) seeds 0-7 (2 runs);
  hud_full (padded conv that really sees the HUD) seeds 0-3 with hud probes; hud_cs (EXTENDED row-93 sensors) seeds 0-3.
