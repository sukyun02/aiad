# LENS dyn notes (batch/epochs/lr for fixed cnnA) 
Candidate: cands/dyn_cnnA (copy of cnnA, unchanged). Budget: 6 K runs, <=2 in flight.
Steps/epoch: bs64=236, bs128=118, bs256=59, bs512=30.
Prior (cnnA/b256_e100 seeds 0-3, lr1e-3): e25 4/4 44.61 (hud_zero 1/4), e50 4/4 43.13, e100 4/4 43.18.

## Plan (6 runs, 4 seeds each, probes hud_zero,noise2)
A  tag b256_lr1e3_e300  bs256 lr1e-3 e300 seeds 4-7 save 30,40,50,60,80,100,150,200,250   (long training + fresh seeds for e50/e100)
B  tag b512_lr1e3_e300  bs512 lr1e-3 e300 seeds 0-3 save 50,75,100,125,150,200,250
C  tag b128_lr1e3_e100  bs128 lr1e-3 e100 save 10,15,20,25,30,40,50,75
D  tag b64_lr1e3_e60    bs64  lr1e-3 e60  save 5,8,10,13,16,20,30,40
E  tag b256_lr3e3_e100  bs256 lr3e-3 e100 save 10,15,20,25,30,40,50,75
F  tag b256_lr3e4_e100  bs256 lr3e-4 e100 save 30,40,50,60,75,90
Logs: work/dyn/run<X>.log ; results: results/dyn_cnnA.jsonl ; read with `K summary dyn_cnnA`.

## Log
