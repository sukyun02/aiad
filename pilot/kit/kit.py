#!/usr/bin/env python
"""Pilot kit: fast, isolated experiments mirroring assignment1's official train / infer loops.

All outputs go under the scratchpad. The repository is only read (src/, data/).

Subcommands
  cache [--data-dir D --out N]  build the in-memory dataset cache (once); KIT_CACHE=N selects a non-default cache
  new   NAME [--from OTHER]  create cands/NAME/network.py (copy of the repo's src/network.py or of another candidate)
  check NAME                 shapes, parameter count, CPU latency, and which parts of network.py differ from the original
  run   NAME --tag T ...     train K independent seeds with the official loop semantics, then drive the fixed track headlessly
  eval  MODEL.pth ...        drive the fixed track with one saved checkpoint
  gate  MODEL.pth            nominal lap on both devices + 6 perturbed laps of one checkpoint (pre-submission check)
  official MODEL.pth         one headless lap through the OFFICIAL evaluation code (loader, hash check, drive loop)
  summary [NAME ...]         aggregate results/*.jsonl
  job [--kind K] -- CMD ...  run any other command under the shared limits

Resource rule (after the 2026-10-06 OOM crashes): a torch process exists only while it holds a slot.
Waiting is done by small stdlib-only processes or threads, and no new work starts while the host is short of RAM.
"""
import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

SP = Path(__file__).resolve().parents[1]
REPO = Path('/home/sukyun02/workspace/aiad/assignment1')
SRC = REPO / 'src'
PY = REPO / '.venv/bin/python'
CANDS = SP / 'cands'
CACHE = SP / 'cache' / (os.environ.get('KIT_CACHE', 'data') + '.npz')    # KIT_CACHE=NAME trains on cache/NAME.npz
RESULTS = SP / 'results'
LOCKS = SP / 'locks'
SLOT_COUNTS = {
    'light': int(os.environ.get('KIT_TRAIN_SLOTS', 4)),     # trainings with batch_size <= 512
    'heavy': int(os.environ.get('KIT_HEAVY_SLOTS', 1)),     # larger batches need GBs of activations
    'eval': int(os.environ.get('KIT_EVAL_SLOTS', 6)),       # headless drives (about 1 GiB of host RAM each)
    'job': int(os.environ.get('KIT_JOB_SLOTS', 4)),         # anything else started through `job`
}
LIGHT_SLOTS, HEAVY_SLOTS, EVAL_SLOTS = SLOT_COUNTS['light'], SLOT_COUNTS['heavy'], SLOT_COUNTS['eval']
MIN_AVAILABLE_GB = float(os.environ.get('KIT_MIN_AVAILABLE_GB', 6))            # no new work below this much free host RAM
MEMORY_WAIT_SECONDS = float(os.environ.get('KIT_MEMORY_WAIT_SECONDS', 1800))   # then give up instead of piling on
CUDA_OOM_EXIT = 75           # exit code of `_train` when the GPU ran out of memory (the orchestrator retries)
GRADIENT_CLIP = 1.0          # same as src/training.py
HUD_BLOB = (slice(86, 92), slice(11, 30))   # leaked score digits in the 96x96 observation
PROBE_FLAGS = {'cuda': [], 'cpu': ['--device', 'cpu'], 'hud_zero': ['--hud', 'zero'],
               'hud_shift': ['--hud', 'shift'], 'hud_dim': ['--hud', 'dim'],
               'noise1': ['--noise', '1'], 'noise2': ['--noise', '2'], 'noise4': ['--noise', '4'],
               'noise2b': ['--noise', '2', '--noise-seed', '1']}
# Gate = the nominal lap on both devices plus perturbations whose base rate was measured on cnnA
# (hud_zero and +-2 pixel noise: about 8 % of perturbed laps of a nominally clean checkpoint were unclean).
GATE_NOMINAL = {'cuda': [], 'cpu': ['--device', 'cpu']}
GATE_PERTURBED = {'hud_zero': ['--hud', 'zero'],
                  **{f'noise2_s{seed}': ['--noise', '2', '--noise-seed', str(seed)] for seed in range(5)}}


def available_gb():
    with open('/proc/meminfo') as stream:
        for line in stream:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1048576
    return float('inf')


class Slot:
    """Cross-process semaphore built from flock'ed files.

    Take the slot first and start the torch process afterwards: whoever waits here must be stdlib-only.
    New work is also held back while the host has less than MIN_AVAILABLE_GB of free RAM.
    """
    jitter = random.Random()     # private, so waiting never disturbs the seeded global generator

    def __init__(self, kind, count=None):
        self.kind, self.count = kind, count or SLOT_COUNTS[kind]

    def __enter__(self):
        LOCKS.mkdir(parents=True, exist_ok=True)
        low_since = None
        while True:
            free = available_gb()
            if free >= MIN_AVAILABLE_GB:
                low_since = None
                for index in range(self.count):
                    handle = open(LOCKS / f'{self.kind}{index}', 'w')
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        self.handle = handle
                        return self
                    except BlockingIOError:
                        handle.close()
            elif low_since is None:
                low_since = time.time()
                print(f'kit: waiting for host RAM ({free:.1f} GiB available, need {MIN_AVAILABLE_GB:g})',
                      file=sys.stderr, flush=True)
            elif time.time() - low_since > MEMORY_WAIT_SECONDS:
                raise SystemExit(f'kit: host RAM stayed below {MIN_AVAILABLE_GB:g} GiB for {MEMORY_WAIT_SECONDS:.0f} s '
                                 f'({free:.1f} GiB available); not starting {self.kind} work')
            time.sleep(0.3 + self.jitter.random() * 0.7)

    def __exit__(self, *exc):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


def load_network_class(path, name='cand_network'):
    source = Path(path).read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module.ClassificationNetwork, source


def labels_from_actions(act):
    import numpy as np
    act = act.astype(np.float64)
    steer, gas, brake = act[:, 0], act[:, 1], act[:, 2]
    left, right, on_gas, on_brake = steer < -0.1, steer > 0.1, gas > 0.1, brake > 0.1
    return np.select([left & on_gas, right & on_gas, left & on_brake, right & on_brake, left, right, on_gas, on_brake],
                     [5, 6, 7, 8, 3, 4, 1, 2], default=0).astype(np.int64)


def cmd_cache(args):
    import numpy as np
    import torch
    root = Path(args.data_dir).resolve() if args.data_dir else REPO / 'data/collected'
    target = SP / 'cache' / (args.out + '.npz') if args.out else CACHE
    paths = sorted(root.rglob('observation_*.npy'))
    if not paths:
        raise SystemExit(f'no observation_*.npy under {root}')
    laps = sorted({p.parent for p in paths})
    obs, act, lap, frame = [], [], [], []
    for path in paths:
        obs.append(np.load(path))
        act.append(np.load(path.with_name(path.name.replace('observation_', 'action_', 1))))
        lap.append(laps.index(path.parent))
        frame.append(int(path.stem.rsplit('_', 1)[1]))
    obs, act = np.stack(obs), np.stack(act)
    assert obs.dtype == np.uint8 and obs.shape[1:] == (96, 96, 3), (obs.dtype, obs.shape)
    labels = labels_from_actions(act)
    Net, _ = load_network_class(SRC / 'network.py', 'orig_network')
    official = torch.cat(Net(device='cpu').actions_to_classes(list(torch.as_tensor(act)))).numpy()
    assert (official == labels).all(), 'vectorised labels differ from actions_to_classes'
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, obs=obs, act=act, lap=np.array(lap), frame=np.array(frame), labels=labels)
    print('cached', target, obs.shape, 'laps', len(laps), 'class counts', np.bincount(labels, minlength=9).tolist())


def cand_dir(name):
    return CANDS / name


def cmd_new(args):
    target = cand_dir(args.name)
    if target.exists():
        raise SystemExit(f'already exists: {target}')
    target.mkdir(parents=True)
    source = cand_dir(args.source) / 'network.py' if args.source else SRC / 'network.py'
    shutil.copyfile(source, target / 'network.py')
    print(target / 'network.py')


def region_report(candidate_text):
    """Classify which parts of the candidate differ from the repo's original network.py."""
    import difflib
    original = (SRC / 'network.py').read_text().splitlines()
    candidate = candidate_text.splitlines()
    start = next(i for i, line in enumerate(original) if '===== TODO' in line)
    end = next(i for i, line in enumerate(original) if '(구현 영역) 끝' in line)
    class_line = next(i for i, line in enumerate(original) if line.startswith('class ClassificationNetwork'))
    inside = helper = other = 0
    other_lines = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, original, candidate, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        if start <= i1 and i2 <= end + 1:
            inside += 1
        elif i2 <= class_line:
            helper += 1
        else:
            other += 1
            other_lines.append(f'orig L{i1 + 1}-{i2}: ' + ' | '.join(l.strip() for l in candidate[j1:j2])[:160])
    return dict(changes_inside_todo_region=inside, helper_code_before_class=helper, changes_elsewhere=other,
                elsewhere_detail=other_lines)


def cmd_check(args):
    with Slot('eval'):
        check(args)


def check(args):
    import torch
    path = cand_dir(args.name) / 'network.py'
    Net, source = load_network_class(path)
    report = region_report(source.decode())
    model = Net(device='cuda')
    x = torch.randint(0, 256, (4, 96, 96, 3), device='cuda').float()
    conv_out = model.conv(x.permute(0, 3, 1, 2).contiguous() / 255.0)
    out = model(x)
    assert out.shape == (4, 9), out.shape
    out.sum().backward()
    params = sum(p.numel() for p in model.parameters())
    cpu = Net(device='cpu').eval()
    torch.set_num_threads(1)
    one = torch.zeros(1, 96, 96, 3)
    with torch.inference_mode():
        for _ in range(20):
            cpu(one)
        begin = time.perf_counter()
        for _ in range(200):
            cpu(one)
        latency = (time.perf_counter() - begin) / 200 * 1000
    print(json.dumps(dict(cand=args.name, conv_out=list(conv_out.shape[1:]), flat=int(conv_out[0].numel()),
                          fc_in_expected=int(conv_out[0].numel()) + 7, out=list(out.shape), params=params,
                          cpu_ms_per_frame_1thread=round(latency, 3), **report), ensure_ascii=False, indent=1))


# ----------------------------------------------------------------------------- training

def train_one(Net, source, run_dir, args, seed, log):
    import numpy as np
    import torch
    import torch.nn as nn
    device = torch.device('cuda')
    data = np.load(CACHE)
    keep = np.ones(len(data['labels']), dtype=bool)
    if args.holdout_lap is not None:
        keep = data['lap'] != args.holdout_lap
    obs = torch.as_tensor(data['obs'][keep]).to(device)
    labels = torch.as_tensor(data['labels'][keep]).to(device)
    count = len(labels)
    model = Net(device=device)
    model.device = device
    model.requires_grad_(True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()
    save_epochs = sorted(set(args.save_epochs + [args.epochs]))
    saved, history = [], {}
    begin = time.time()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(count, device=device)
        total, batches = 0.0, 0
        for start in range(0, count, args.batch_size):
            index = order[start:start + args.batch_size]
            batch_out = model(obs[index].float())
            loss = loss_function(batch_out, labels[index])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
            optimizer.step()
            total += loss.item()
            batches += 1
        history[epoch] = (total, total / batches)
        if epoch in save_epochs:
            target = run_dir / (f's{seed}.pth' if epoch == args.epochs else f's{seed}_e{epoch}.pth')
            target.with_suffix('.network.py').write_bytes(source)
            torch.save(dict(format_version=1, network_sha256=hashlib.sha256(source).hexdigest(), epoch=epoch,
                            state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            optimizer_state=optimizer.state_dict()), target)
            model.eval()
            with torch.inference_mode():
                predicted = torch.cat([model(obs[i:i + 2048].float()).argmax(1) for i in range(0, count, 2048)])
            model.train()
            steering = predicted >= 3
            stats = dict(epoch=epoch, train_loss_sum=round(total, 5), train_loss_mean=round(total / batches, 5),
                         train_acc=round((predicted == labels).float().mean().item(), 4),
                         pred_steer_rate=round(steering.float().mean().item(), 4),
                         data_steer_rate=round((labels >= 3).float().mean().item(), 4))
            if args.holdout_lap is not None:
                val_obs = torch.as_tensor(data['obs'][~keep]).to(device)
                val_labels = torch.as_tensor(data['labels'][~keep]).to(device)
                model.eval()
                with torch.inference_mode():
                    val_pred = torch.cat([model(val_obs[i:i + 2048].float()).argmax(1) for i in range(0, len(val_labels), 2048)])
                model.train()
                stats['val_acc'] = round((val_pred == val_labels).float().mean().item(), 4)
                del val_obs, val_labels
            saved.append((target, stats))
            log(f'seed {seed} epoch {epoch}: loss_sum {total:.4f} mean {total / batches:.4f} acc {stats["train_acc"]}')
    seconds = time.time() - begin
    del obs, labels, optimizer
    torch.cuda.empty_cache()
    return saved, seconds, sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------- driving

def drive(model_path, device='cuda', hud='none', max_seconds=150.0, trace=False, noise=0, noise_seed=0):
    """Headless replica of evaluation.drive_until_return on the fixed track."""
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import numpy as np
    import torch
    from gui import make_driving_env
    from sdc_wrapper import SDC_Wrapper
    from track import TRACK_SEED
    from driving import DrivingController, DEFAULT_DRIVING
    model_path = Path(model_path)
    Net, _ = load_network_class(model_path.with_suffix('.network.py'), 'eval_network')
    device = torch.device(device)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model = Net(device=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.device = device
    model.eval()
    env = SDC_Wrapper(make_driving_env(display=False, unlimited=True))
    controller = DrivingController(DEFAULT_DRIVING, env.metadata['render_fps'])
    counts = [0] * 9
    rng = np.random.default_rng(noise_seed)
    offsets, speeds, rows = [], [], []
    path = 0.0
    previous = None
    last_class, switches, offroad = None, 0, 0
    info = {}
    status = 'running'
    try:
        observation, _ = env.reset(seed=TRACK_SEED)
        controller.reset()
        metrics = env.unwrapped.metrics
        with torch.inference_mode():
            while True:
                seen = observation
                if hud != 'none':
                    seen = observation.copy()
                    if hud == 'zero':
                        seen[HUD_BLOB] = 0
                    elif hud == 'shift':
                        seen[HUD_BLOB] = np.roll(seen[HUD_BLOB], 1, axis=1)
                    elif hud == 'dim':
                        seen[HUD_BLOB] = (seen[HUD_BLOB].astype(np.float32) * 0.9).astype(np.uint8)
                if noise:
                    seen = np.clip(seen.astype(np.int16) + rng.integers(-noise, noise + 1, size=seen.shape), 0, 255).astype(np.uint8)
                image = torch.as_tensor(np.ascontiguousarray(seen[None]), dtype=torch.float32, device=device)
                scores = model(image)
                cls = int(torch.argmax(scores, dim=1).item())
                action = model.scores_to_action(scores)
                counts[cls] += 1
                if last_class is not None and cls != last_class:
                    switches += 1
                last_class = cls
                speed = np.linalg.norm(env.unwrapped.car.hull.linearVelocity)
                position = np.array(env.unwrapped.car.hull.position)
                path += 0.0 if previous is None else float(np.linalg.norm(position - previous))
                previous = position
                along, error = metrics.project(position)
                offsets.append(error)
                speeds.append(float(speed))
                if trace:
                    top = torch.topk(scores[0], 2).values
                    rows.append((round(along, 2), round(error, 3), cls, round(float(top[0] - top[1]), 3)))
                observation, _, done, truncated, info = env.step(controller.apply(action, speed))
                offroad += int(info.get('lane_departure', False))
                if truncated or (done and not info.get('lap_finished', False)):
                    status = 'aborted'
                    break
                if info.get('lap_finished', False):
                    status = 'finished'
                    break
                if info['elapsed'] >= max_seconds:
                    status = 'timeout'
                    break
    finally:
        env.close()
    offsets, speeds = np.array(offsets), np.array(speeds)
    frames = sum(counts)
    result = dict(status=status, lap_time=info.get('lap_time'), progress=round(info['progress'], 2),
                  center=round(info['center'], 2), frames=frames, path_len=round(path, 1),
                  mean_offset=round(float(offsets.mean()), 3), max_offset=round(float(offsets.max()), 2),
                  mean_speed=round(float(speeds[100:].mean()) if frames > 100 else float(speeds.mean()), 2),
                  steer_frac=round(sum(counts[3:]) / frames, 4), switches=switches, offroad_frames=offroad,
                  action_counts=counts)
    if trace:
        result['trace'] = rows
    return result


def cmd_train(args):
    """The torch half of one seed. Started by worker() only while it holds a training slot."""
    import torch
    torch.set_num_threads(2)
    seed = args.seed
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    run_dir = Path(args.run_dir)
    Net, source = load_network_class(run_dir / 'network.py')

    def log(message):
        print(message, flush=True)

    try:
        saved, seconds, params = train_one(Net, source, run_dir, args, seed, log)
    except RuntimeError as error:      # torch.OutOfMemoryError and torch.AcceleratorError both derive from it
        if 'out of memory' not in str(error).lower():
            raise
        raise SystemExit(CUDA_OOM_EXIT)
    Path(args.manifest).write_text(json.dumps(dict(
        saved=[dict(target=str(target), stats=stats) for target, stats in saved], params=params,
        seconds=round(seconds, 1), network_sha=hashlib.sha256(source).hexdigest()[:12])))


def worker(args):
    """Stdlib-only orchestrator of one seed: it queues for slots and starts a torch process only inside one."""
    seed = args.seed
    run_dir = Path(args.run_dir)
    manifest = run_dir / f's{seed}.train.json'
    kit = str(Path(__file__).resolve())

    def log(message):
        print(message, flush=True)

    train = [str(PY), kit, '_train'] + sys.argv[2:] + ['--manifest', str(manifest)]
    for attempt in range(6):
        with Slot('heavy' if args.batch_size > 512 else 'light'):
            code = subprocess.run(train).returncode      # writes to this seed's log
        if code != CUDA_OOM_EXIT:
            break
        log(f'seed {seed}: CUDA OOM, retry {attempt + 1}')
        time.sleep(15 + 15 * attempt)
    if code != 0:
        hint = ' (SIGKILL: probably the memory cap, check `journalctl -k | grep -i "out of memory"`)' if code == -9 else ''
        raise SystemExit(f'seed {seed}: training failed with exit code {code}{hint}')
    trained = json.loads(manifest.read_text())
    saved = [(Path(item['target']), item['stats']) for item in trained['saved']]
    base = dict(cand=args.name, tag=args.tag, seed=seed, batch_size=args.batch_size, lr=args.lr,
                epochs_total=args.epochs, holdout_lap=args.holdout_lap, params=trained['params'],
                train_seconds=trained['seconds'], network_sha=trained['network_sha'])
    from concurrent.futures import ThreadPoolExecutor
    kinds = ['cuda'] + list(args.probes)
    jobs = [(target, kind) for target, _ in saved for kind in kinds]

    def launch(job):
        target, kind = job
        command = [str(PY), kit, 'eval', str(target)] + PROBE_FLAGS[kind]
        for attempt in range(6):
            with Slot('eval'):       # the thread waits, the eval process starts only with the slot in hand
                done = subprocess.run(command, capture_output=True, text=True)
            if done.returncode == 0 or 'out of memory' not in done.stderr.lower():
                break
            time.sleep(10 + 10 * attempt)
        if done.returncode != 0:
            raise RuntimeError(f'eval failed for {target} {kind}:\n{done.stderr[-2000:]}')
        return json.loads(done.stdout.strip().splitlines()[-1])

    with ThreadPoolExecutor(max_workers=min(len(jobs), SLOT_COUNTS['eval'])) as pool:
        outcomes = dict(zip(jobs, pool.map(launch, jobs)))
    for target, stats in saved:
        record = dict(base, model=str(target), **stats)
        record['drive'] = outcomes[(target, 'cuda')]
        for kind in args.probes:
            record['drive_' + kind] = outcomes[(target, kind)]
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(RESULTS / f'{args.name}.jsonl', 'a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        log('RESULT ' + brief(record))


def brief(record):
    def one(d):
        if d is None:
            return '-'
        lap = f"{d['lap_time']:.2f}s" if d.get('lap_time') is not None else d['status']
        return f"{lap} prog {d['progress']:.1f} ctr {d['center']:.1f} path {d['path_len']:.0f} steer {d['steer_frac']:.2f} maxoff {d['max_offset']:.1f}"
    text = (f"{record['cand']}/{record['tag']} s{record['seed']} e{record['epoch']} loss_sum {record['train_loss_sum']:.3f} "
            f"acc {record['train_acc']:.3f} | cuda: {one(record['drive'])}")
    if 'val_acc' in record:
        text += f" | val_acc {record['val_acc']:.3f}"
    for key in sorted(k for k in record if k.startswith('drive_')):
        if key in record:
            d = record[key]
            text += f" | {key[6:]}: " + (f"{d['lap_time']:.2f}s" if d.get('lap_time') is not None else f"{d['status']}@{d['progress']:.0f}")
    return text


def cmd_run(args):
    if not CACHE.exists():
        raise SystemExit(f'cache not found: {CACHE} (run `kit.py cache` first)')
    source = cand_dir(args.name) / 'network.py'
    run_dir = cand_dir(args.name) / 'runs' / args.tag
    if run_dir.exists():
        raise SystemExit(f'tag already used: {run_dir} (choose a new --tag)')
    run_dir.mkdir(parents=True)
    shutil.copyfile(source, run_dir / 'network.py')     # frozen for this run
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    procs = []
    for seed in seeds:
        command = [str(PY), str(Path(__file__).resolve()), '_worker', args.name, '--tag', args.tag,
                   '--run-dir', str(run_dir), '--seed', str(seed), '--batch-size', str(args.batch_size),
                   '--epochs', str(args.epochs), '--lr', str(args.lr),
                   '--save-epochs', ','.join(map(str, args.save_epochs))]
        if args.holdout_lap is not None:
            command += ['--holdout-lap', str(args.holdout_lap)]
        if args.probes:
            command += ['--probes', ','.join(args.probes)]
        log = open(run_dir / f's{seed}.log', 'w')
        procs.append((seed, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
    failed = []
    for seed, proc, log in procs:
        code = proc.wait()
        log.close()
        text = (run_dir / f's{seed}.log').read_text()
        lines = [line for line in text.splitlines() if line.startswith('RESULT ')]
        for line in lines:
            print(line[7:])
        if code != 0:
            failed.append(seed)
            print(f'seed {seed} FAILED (exit {code}); last log lines:\n' + '\n'.join(text.splitlines()[-25:]))
    print(f'results file: {RESULTS / (args.name + ".jsonl")}')
    if failed:
        raise SystemExit(1)


def cmd_eval(args):
    def run():
        import torch
        torch.set_num_threads(1)
        return drive(args.model, args.device, hud=args.hud, trace=bool(args.trace), noise=args.noise, noise_seed=args.noise_seed)

    if args.slot:
        with Slot('eval'):       # torch is imported only after the slot is taken
            result = run()
    else:
        result = run()
    if args.trace:
        Path(args.trace).write_text(json.dumps(result.pop('trace')))
    print(json.dumps(result, ensure_ascii=False))


def cmd_official(args):
    """One headless lap through src/evaluation.evaluate, i.e. scripts/infer.py without the window.

    It builds the model from the repo's current src/network.py exactly as infer.py does, so it checks the
    official loader and hash path that the kit's own eval bypasses.
    """
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')     # same as scripts/infer.py
    model = str(Path(args.model).resolve())
    with Slot('eval'):
        sys.path.insert(0, str(SRC))
        from evaluation import evaluate
        from driving import DEFAULT_DRIVING
        try:
            evaluate(dict(model_path=model, display=False), score=False, driving=dict(DEFAULT_DRIVING))
        except (ValueError, OSError, RuntimeError) as error:        # scripts/infer.py reports these the same way
            raise SystemExit(f'실행 오류: {error}')


def is_clean(result):
    """A lap that counts: finished, the whole distance scored, never off the road."""
    return result.get('status') == 'finished' and result['progress'] >= 100.0 and result['offroad_frames'] == 0


def cmd_gate(args):
    """Pre-submission check of one checkpoint: the nominal lap plus small perturbations of the input."""
    from concurrent.futures import ThreadPoolExecutor
    kit = str(Path(__file__).resolve())
    model = str(Path(args.model).resolve())

    probes = {**GATE_NOMINAL, **GATE_PERTURBED}

    def launch(kind):
        with Slot('eval'):
            done = subprocess.run([str(PY), kit, 'eval', model] + probes[kind], capture_output=True, text=True)
        if done.returncode != 0:
            return dict(status='error', detail=done.stderr.strip()[-300:])
        return json.loads(done.stdout.strip().splitlines()[-1])

    with ThreadPoolExecutor(max_workers=min(len(probes), SLOT_COUNTS['eval'])) as pool:
        results = dict(zip(probes, pool.map(launch, probes)))
    print(f"{'probe':10s} {'clean':5s} {'status':9s} {'lap':>6s} {'progress':>8s} {'center':>6s} {'maxoff':>6s} {'offroad':>7s}")
    for kind, result in results.items():
        if result['status'] == 'error':
            print(f"{kind:10s} {'no':5s} error     {result['detail']}")
            continue
        lap = f"{result['lap_time']:.2f}" if result.get('lap_time') is not None else '-'
        print(f"{kind:10s} {'yes' if is_clean(result) else 'no':5s} {result['status']:9s} {lap:>6s} {result['progress']:8.2f} "
              f"{result['center']:6.2f} {result['max_offset']:6.2f} {result['offroad_frames']:7d}")
    same = ('status', 'lap_time', 'progress', 'center', 'frames')
    nominal = (all(is_clean(results[kind]) for kind in GATE_NOMINAL)
               and all(results['cuda'][key] == results['cpu'][key] for key in same))
    unclean = [kind for kind in GATE_PERTURBED if not is_clean(results[kind])]
    not_finished = [kind for kind in unclean if results[kind].get('status') != 'finished']
    worst = max((result['max_offset'] for result in results.values() if 'max_offset' in result), default=float('nan'))
    passed = nominal and not not_finished and len(unclean) <= 1
    print(f"nominal (cuda, cpu clean and identical): {'yes' if nominal else 'NO'} | perturbed clean "
          f"{len(GATE_PERTURBED) - len(unclean)}/{len(GATE_PERTURBED)}, not finished {len(not_finished)} | "
          f"worst max_offset {worst:.2f} | gate {'PASS' if passed else 'FAIL'}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    raise SystemExit(0 if passed else 1)


def cmd_job(args):
    """Run any command under the shared limits, e.g. `k job -- python my_analysis.py`."""
    command = args.cmd[1:] if args.cmd[:1] == ['--'] else args.cmd
    if not command:
        raise SystemExit('usage: k job [--kind job|eval|light|heavy] -- COMMAND ...')
    with Slot(args.kind):
        code = subprocess.run(command).returncode
    raise SystemExit(code if code >= 0 else 128 - code)


def cmd_summary(args):
    import statistics
    files = [RESULTS / f'{name}.jsonl' for name in args.names] if args.names else sorted(RESULTS.glob('*.jsonl'))
    groups = {}
    for path in files:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            record = json.loads(line)
            key = (record['cand'], record['tag'], record['epoch'], record['batch_size'], record['lr'])
            groups.setdefault(key, []).append(record)

    def med(values):
        return statistics.median(values) if values else float('nan')
    print(f"{'cand/tag':44s} {'ep':>4s} {'bs':>5s} {'lr':>7s} {'fin/n':>5s} {'lap med':>7s} {'min':>6s} {'max':>6s} {'ctr':>5s} {'path':>6s} "
          f"{'maxoff':>6s} {'steer':>5s} {'acc':>5s} {'loss':>7s}  probes (finished/n, median lap of finished)")
    for key in sorted(groups):
        records = groups[key]
        finished = [r for r in records if r['drive']['status'] == 'finished']
        laps = [r['drive']['lap_time'] for r in finished]
        probes = []
        for kind in sorted({k for r in records for k in r if k.startswith('drive_')}):
            have = [r[kind] for r in records if kind in r]
            ok = [d['lap_time'] for d in have if d['status'] == 'finished']
            extra = ''
            if kind == 'drive_cpu':
                same = sum(1 for r in records if kind in r and r[kind]['lap_time'] == r['drive']['lap_time'] and r[kind]['status'] == r['drive']['status'])
                extra = f' same-as-cuda {same}/{len(have)}'
            probes.append(f"{kind[6:]} {len(ok)}/{len(have)} ({med(ok):.2f}){extra}")
        print(f"{(key[0] + '/' + key[1])[:44]:44s} {key[2]:4d} {key[3]:5d} {key[4]:7.0e} {str(len(finished)) + '/' + str(len(records)):>5s} "
              f"{med(laps):7.2f} {min(laps, default=float('nan')):6.2f} {max(laps, default=float('nan')):6.2f} "
              f"{med([r['drive']['center'] for r in finished]):5.1f} {med([r['drive']['path_len'] for r in finished]):6.0f} "
              f"{med([r['drive']['max_offset'] for r in finished]):6.1f} "
              f"{med([r['drive']['steer_frac'] for r in records]):5.2f} {med([r['train_acc'] for r in records]):5.3f} "
              f"{med([r['train_loss_mean'] for r in records]):7.4f}  " + '; '.join(probes))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('cache')
    p.add_argument('--data-dir', help='folder with observation_/action_ files (default: the repo data/collected)')
    p.add_argument('--out', help='cache name; train on it with KIT_CACHE=NAME (default: data)')
    p = sub.add_parser('new')
    p.add_argument('name')
    p.add_argument('--from', dest='source')
    p = sub.add_parser('check')
    p.add_argument('name')
    p = sub.add_parser('official')
    p.add_argument('model')
    p = sub.add_parser('gate')
    p.add_argument('model')
    p.add_argument('--json', help='also write the raw results to this file')
    p = sub.add_parser('job')
    p.add_argument('--kind', default='job', choices=sorted(SLOT_COUNTS))
    p.add_argument('cmd', nargs=argparse.REMAINDER)
    for name in ('run', '_worker', '_train'):
        p = sub.add_parser(name)
        p.add_argument('name')
        p.add_argument('--tag', required=True)
        p.add_argument('--batch-size', type=int, default=4096)
        p.add_argument('--epochs', type=int, default=100)
        p.add_argument('--lr', type=float, default=0.001)
        p.add_argument('--seeds', type=int, default=3)
        p.add_argument('--seed-start', type=int, default=0)
        p.add_argument('--save-epochs', type=lambda s: [int(v) for v in s.split(',') if v], default=[])
        p.add_argument('--holdout-lap', type=int)
        p.add_argument('--probes', type=lambda v: [x for x in v.split(',') if x], default=[],
                       help='comma list of extra drives: cpu,hud_zero,hud_shift,hud_dim,noise1,noise2,noise2b,noise4')
        if name != 'run':
            p.add_argument('--run-dir', required=True)
            p.add_argument('--seed', type=int, required=True)
        if name == '_train':
            p.add_argument('--manifest', required=True)
    p = sub.add_parser('eval')
    p.add_argument('model')
    p.add_argument('--device', default='cuda')
    p.add_argument('--hud', default='none', choices=['none', 'zero', 'shift', 'dim'])
    p.add_argument('--trace')
    p.add_argument('--slot', action='store_true')
    p.add_argument('--noise', type=int, default=0)
    p.add_argument('--noise-seed', type=int, default=0)
    p = sub.add_parser('summary')
    p.add_argument('names', nargs='*')
    args = parser.parse_args()
    {'cache': cmd_cache, 'new': cmd_new, 'check': cmd_check, 'run': cmd_run, '_worker': worker, '_train': cmd_train,
     'eval': cmd_eval, 'gate': cmd_gate, 'official': cmd_official, 'summary': cmd_summary, 'job': cmd_job}[args.command](args)


if __name__ == '__main__':
    main()
