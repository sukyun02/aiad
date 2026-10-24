# rob lens notes (robustness of existing checkpoints to perturbation)
Scripts: work/rob/mkjobs.py -> jobsA.tsv (270 evals: cuda,cpu,n2s0,hudzero,n1s0,n4s0,hudshift,huddim,thr4,ln3) and jobsB.tsv (162: noise seeds 1,2).
runall.sh JOBFILE 6 runs them (idempotent: skips ids with out/ID.json). Results: out/ID.json, traces trace/ID.json. ID = arch-eEP-sSEED__probe.
27 checkpoints: cnnA / cnnA_crop x e25,e50,e100 x s0-3 ; baseline ; base4096 s0,s1.
Custom probes via rdrive.py (wraps kit.drive, kit untouched): thr4 = CPU 4 threads; ln3 = gaussian logit noise sigma 1e-3; later: single-frame flips.
All noise/hud probes run on CPU (1 thread). No K run used so far (0/2).
- 17:15 launched jobsA in background (P=6).
- 17:18 machine saturated by other agents' trainings (load 29, GPU 100 %): a CUDA eval takes 60-90 s. Killed the jobsA runner (in-flight cuda evals for e50 x8, e100 x6 finish on their own).
  New priority-ordered list jobs2.tsv (mkjobs2.py), CPU only: cpu nominal x27 -> n2s0,n1s0 on 16 main (e50,e100) -> hudzero,n4s0,n2s1 main -> n2s0,n1s0 on e25+base -> thr4,ln3 -> ... 
  CUDA nominal for the other checkpoints is taken from results/*.jsonl (K run drive) and compared by frames/path_len/action_counts.
  Runner: runall.sh jobs2.tsv 5 (lane of 1 kept free for flip probes).
