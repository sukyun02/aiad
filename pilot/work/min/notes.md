# LENS min notes (running log)
Budget: 6 K runs. Plan: bisect ladder between starter (fails) and cnnA (works).
All runs: --batch-size 256 --epochs 100 --save-epochs 25,50 --seeds 4 --probes hud_zero (matches cnnA/b256_e100 rows).
Candidates created via work/min/mk.py (edits only self.conv / self.fc of my own cands):
- min_c3: Conv(3,8,5,s4) R Conv(8,16,3,s2) R Conv(16,32,3,s2) R -> 32x5x5 ; fc Linear(807,64) R Linear(64,9) ; 58,713 params 0.59 ms
- min_c1: starter conv Conv(3,8,5,s4) R + MPSCompatibleAdaptiveAvgPool2d((12,12)) -> 8x12x12 ; fc Linear(1159,64) R Linear(64,9) ; 75,433 params 0.53 ms
## Run 1+2 launched (tags b256_e100): min_c3, min_c1
Next: if min_c1 works -> single-ingredient ablations from the starter (pool12 only / hidden only); if not -> min_c2, min_c4h.
