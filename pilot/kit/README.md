# Pilot kit (read-only for agents — do not edit files in kit/)

SP=/home/sukyun02/workspace/aiad/pilot
K=$SP/kit/k            # wrapper: runs kit.py with the repo's venv python inside a memory-capped scope

The kit mirrors the official loops of /home/sukyun02/workspace/aiad/assignment1 (src/training.py, src/evaluation.py)
but loads the data from an in-memory cache and drives headlessly. The headless drive reproduces the GUI result of the
provided baseline exactly (45.54 s / center 51.03), and replaying the 7 expert laps reproduces every recorded frame.

## Commands
$K new NAME [--from OTHER]        -> $SP/cands/NAME/network.py  (copy of the repo's src/network.py, or of cands/OTHER)
                                     edit `self.conv = ...` / `self.fc = ...` in that file
$K check NAME                     -> conv output shape, expected fc input, params, CPU ms/frame, and which regions differ
                                     from the original (changes_inside_todo_region / helper_code_before_class / changes_elsewhere)
$K run NAME --tag TAG [--batch-size 256] [--epochs 60] [--lr 0.001] [--seeds 4] [--seed-start 0]
            [--save-epochs 30] [--probes cpu,hud_zero,noise2] [--holdout-lap 6]
     trains each seed independently (official semantics: Adam, CrossEntropyLoss, clip_grad_norm 1.0, shuffle each epoch,
     model.train() mode), saves checkpoints at --save-epochs and at the end, then drives the fixed track once per checkpoint
     (+ once per probe). One line per checkpoint is printed and appended to $SP/results/NAME.jsonl.
     A TAG can be used once per candidate; network.py is frozen into cands/NAME/runs/TAG/ at launch, so you may keep
     editing cands/NAME/network.py afterwards (but prefer a new NAME per architecture so results stay attributable).
     Checkpoints: cands/NAME/runs/TAG/s<seed>.pth (final) and s<seed>_e<epoch>.pth, each with a .network.py twin.
$K eval MODEL.pth [--device cpu] [--hud zero|shift|dim] [--noise K --noise-seed S] [--trace out.json] [--slot]
     one headless lap. --trace writes rows [along_track(0..959), lateral_offset, class, logit_margin] per frame.
     Always pass --slot when more than one eval may be running.
$K gate MODEL.pth [--json out.json]
     pre-submission check of one checkpoint: 8 laps = nominal on cuda and cpu, hud_zero, +-2 pixel noise with seeds 0-4,
     one table row each. "clean" = finished, progress 100.00, 0 off-road frames.
     PASS (exit code 0) = cuda and cpu clean and identical, every perturbed lap finished, at most 1 perturbed lap unclean.
     A nominally clean cnnA checkpoint was unclean in about 8 % of such perturbed laps, so an ordinary good checkpoint
     passes most of the time; the gate screens out fragile ones, it does not certify anything.
     Works on official checkpoints too (models/<date>.pth with its .network.py twin next to it).
$K official MODEL.pth
     one headless lap through the OFFICIAL evaluation code (src/evaluation.evaluate = scripts/infer.py without the
     window): official loader, hash / equivalence check against the repo's current src/network.py, official drive loop.
     Prints the official result line (진행률 / 중앙선 유지 / 랩타임). Use it to confirm that a pair loads and drives
     the same way in the official code as in `$K eval`.
$K job [--kind job|eval] -- COMMAND ...
     runs any other command (your own analysis script, a trace tool that imports torch) under the shared limits.
$K summary [NAME ...]             -> aggregated table (finish count, median/min/max lap of finished runs, probes)

## Reading the output
status finished|aborted|timeout ; lap_time (s, only when finished) ; progress (0-100; <100 with status finished means
part of the lap was driven off-road) ; center (0-100, 100 = on the centre line) ; path_len (centre line = 959) ;
max_offset (road half-width = 6.67) ; steer_frac (share of frames with a steering class) ; switches (class changes).
Probes: cpu = same model on CPU ; hud_zero = leaked score digits blanked before the model sees the frame ;
hud_shift = digits shifted 1 px ; noiseK = uniform integer noise in [-K, K] on every pixel.

## Capacity rules (host RAM is the limit, not the GPU)
On 2026-10-06 this machine (30.7 GiB RAM, 8 GiB swap, 12 GB GPU) crashed twice because 100-170 python processes were
alive at once: every queued job had already imported torch (0.26 GiB each) and each finished training started all of
its evaluations together. The kit now enforces:
- A torch process exists only while it holds a slot: 4 training slots (batch_size <= 512), 1 for larger batches,
  6 eval slots, 4 `job` slots. Whatever waits is a small stdlib-only process or thread.
- No new work starts while the host has less than 6 GiB of available RAM (it waits, and gives up after 30 minutes).
- Everything started through $K lives in aiad-pilot.slice, capped at 18 GiB RAM + 2 GiB swap. If the cap is hit the
  kernel kills a process of that slice only; the run reports `exit code -9`.
  Stop everything the kit started with:  systemctl --user stop aiad-pilot.slice

Rules for whoever uses the kit (people and agents alike):
- Every process that imports torch must be started through $K (run / eval --slot / check / job). Never start such
  processes with a bare `xargs -P`, `&` loops, `nohup python ...` or a thread pool of subprocesses: wrap each one as
  `$K job -- python script.py ...` (then `xargs -P 8 $K job -- ...` is fine, the surplus just queues cheaply).
- Keep --seeds <= 4 per run and at most 2 `run` commands in flight per agent; more only queue.
- Before launching anything, look at `free -h` (Mem available AND Swap used), not only at nvidia-smi.
- A 4-seed run with 60 epochs typically needs 3-8 minutes of wall clock: call Bash with timeout 600000. If a command
  times out it keeps running in the background; read its results later with `$K summary NAME`.
- Never use pkill/killall. Pilot experiments never write outside $SP and treat /home/sukyun02/workspace/aiad/assignment1
  as read-only. The only exception is executing a step of a plan under assignment1/docs/ (official training through
  `$K job -- .venv/bin/python scripts/train.py ...` writes models/ and docs/logs/ there).
- Own scripts go to $SP/work/<your-lens>/ ; they may `import kit` after sys.path.insert(0, '$SP/kit').
