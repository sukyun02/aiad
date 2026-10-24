# rel lens notes (running log)
Plan: 6 PRIMARY variants of cnnA (rel_bn, rel_drop p0.3, rel_w2, rel_w05, rel_pool, rel_5conv), created via work/rel/gen.py.
All runs: --batch-size 256 --epochs 100 --save-epochs 25,50 --seeds 4 --probes cpu,noise2 (same grid as cnnA/b256_e100 control), tag b256_e100.
Control probes on existing cnnA checkpoints via work/rel/probe.py -> work/rel/cnnA_probes.jsonl
K check (machine loaded, ms noisy): bn 194889p 1.25ms; drop 194537p 1.16ms; w2 506697p 2.23ms; w05 83385p 0.82ms; pool 194537p 2.70ms; 5conv 231465p 1.68ms
## Round 1 launched: rel_bn, rel_drop (K run #1, #2)
