"""Inspect recorded laps section by section and build the stage-2 training folder from their good sections (no torch).

    curate.py inspect [LAP_DIR ...] [--new] [--target SECONDS] [--json FILE]
    curate.py build --out DIR [LAP_DIR ...] [--new] [--target SECONDS] [--whole-laps [--lap-max S]] [--min-laps 4] [--dry-run]
    curate.py refs                      (re-measure the reference section times from the scripted laps; done once)

A lap dir holds observation_XXXXX.npy / action_XXXXX.npy as saved by scripts/collect.py. --new = every lap folder under
assignment1/data/collected that is not one of the 7 provided laps (20260923_*) plus everything under assignment1/data/stage2_raw.

Each lap is replayed on the fixed track (deterministic), then the track is cut into 14 sections (7 bends with 30 units of
approach and exit, 7 straights). A section of a lap is GOOD when it was driven on the road, with the gas held, in at most
`ref + tol` frames, where ref is the time of the scripted inside-line reference laps (set A). `build` copies the frames of the
good sections to DIR/<lap>/ (the originals are never changed) and writes DIR/curation.json. Frames before the first gas
(start hold) and frames without gas are left out.
Start it through the kit:  k job -- python curate.py ...
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
HERE = Path(__file__).resolve().parent
SP = HERE.parents[1]
REPO = Path('/home/sukyun02/workspace/aiad/assignment1')
sys.path.insert(0, str(REPO / 'src'))

import numpy as np

PROVIDED_PREFIX = '20260923_'
# The bends of the fixed track along the centre line (0..959). C6 (left) runs straight into C7 (right): one unit.
BENDS = [('C1', 45.5, 77.0), ('C2', 248.5, 273.0), ('C3', 381.5, 399.0), ('C4', 518.0, 549.5), ('C5', 609.0, 633.5),
         ('C67', 728.0, 766.5), ('C8', 879.0, 913.5)]
PAD = 30.0
TRACK_LENGTH = 959.0
GAS_CLASSES = (1, 5, 6)
REFS = HERE / 'refs.json'
REPLAYS = HERE / 'replays'


def sections():
    """[(name, start, end)] covering 0..TRACK_LENGTH: bends with PAD on both sides, straights in between."""
    out, cursor, index = [], 0.0, 0
    windows = [(name, a - PAD, b + PAD) for name, a, b in BENDS]
    for position, (name, a, b) in enumerate(windows):
        a = max(a, cursor)                                  # C4 and C5 windows touch
        if position + 1 < len(windows) and windows[position + 1][1] < b:
            b = (b + windows[position + 1][1]) / 2
        if a > cursor + 1e-9:
            out.append((f'S{index}', cursor, a))
            index += 1
        out.append((name, a, b))
        cursor = b
    if cursor < TRACK_LENGTH:
        out.append((f'S{index}', cursor, TRACK_LENGTH))
    return out


SECTIONS = sections()


def classes_from_actions(actions):
    steer, gas, brake = actions[:, 0], actions[:, 1], actions[:, 2]
    left, right, on_gas, on_brake = steer < -0.1, steer > 0.1, gas > 0.1, brake > 0.1
    return np.select([left & on_gas, right & on_gas, left & on_brake, right & on_brake, left, right, on_gas, on_brake],
                     [5, 6, 7, 8, 3, 4, 1, 2], default=0).astype(np.int64)


def replay(folder):
    """Per-frame arrays of one recorded lap. Cached in replays/ by folder name and frame count."""
    folder = Path(folder).resolve()
    obs_paths = sorted(folder.glob('observation_*.npy'))
    if not obs_paths:
        raise SystemExit(f'no observation_*.npy in {folder}')
    REPLAYS.mkdir(exist_ok=True)
    cache = REPLAYS / f'{folder.parent.name}__{folder.name}__{len(obs_paths)}.npz'
    if cache.exists():
        data = np.load(cache)
        return {key: data[key] for key in data.files}
    from gui import make_driving_env
    from sdc_wrapper import SDC_Wrapper
    from track import TRACK_SEED
    from driving import DrivingController, DEFAULT_DRIVING

    actions = np.array([np.load(p.with_name(p.name.replace('observation_', 'action_', 1))) for p in obs_paths])
    numbers = np.array([int(p.stem.rsplit('_', 1)[1]) for p in obs_paths])
    env = SDC_Wrapper(make_driving_env(display=False, collect=True))
    controller = DrivingController(DEFAULT_DRIVING, env.metadata['render_fps'])
    along, signed, speed_log, off_road, mismatch = [], [], [], [], 0
    info = {}
    try:
        observation, _ = env.reset(seed=TRACK_SEED)
        controller.reset()
        metrics = env.unwrapped.metrics
        car = env.unwrapped.car
        for obs_path, action in zip(obs_paths, actions):
            mismatch += int(not np.array_equal(observation, np.load(obs_path)))
            position = np.array(car.hull.position)
            speed = float(np.linalg.norm(car.hull.linearVelocity))
            relative = position - metrics.points
            fraction = np.clip(np.sum(relative * metrics.vectors, axis=1) / metrics.lengths ** 2, 0, 1)
            errors = np.linalg.norm(relative - fraction[:, None] * metrics.vectors, axis=1)
            j = int(np.argmin(errors))
            v = metrics.vectors[j]
            along.append(float(metrics.starts[j] + fraction[j] * metrics.lengths[j]))
            signed.append(float(np.sign(v[0] * relative[j][1] - v[1] * relative[j][0]) * errors[j]))      # + = left
            speed_log.append(speed)
            observation, _, done, truncated, info = env.step(controller.apply(action, speed))
            off_road.append(bool(metrics.off_road))
            if done or truncated:
                break
    finally:
        env.close()
    count = len(along)
    result = dict(along=np.array(along), offset=np.array(signed), speed=np.array(speed_log), off_road=np.array(off_road),
                  cls=classes_from_actions(actions[:count]), number=numbers[:count], frames_on_disk=np.array(len(obs_paths)),
                  mismatch=np.array(mismatch), finished=np.array(bool(info.get('lap_finished', False))),
                  lap_time=np.array(info.get('lap_time') or np.nan), progress=np.array(info.get('progress', 0.0)),
                  center=np.array(info.get('center', 0.0)))
    np.savez(cache, **result)
    return result


def lap_sections(lap):
    """Per section: frame mask, number of frames, off-road frames, frames without gas, largest inside offset."""
    cls, along = lap['cls'], lap['along']
    gas = np.isin(cls, GAS_CLASSES)
    hold = int(np.argmax(gas)) if gas.any() else len(cls)
    index = np.arange(len(cls))
    # `along` jumps back to ~0 only at the very end of the lap; treat the last frames near the finish as the last section
    wrapped = (index > len(cls) * 0.8) & (along < 100)
    position = np.where(wrapped, TRACK_LENGTH - 1e-6, along)
    rows = []
    for name, a, b in SECTIONS:
        mask = (position >= a) & (position < b) & (index >= hold)
        rows.append(dict(name=name, mask=mask, frames=int(mask.sum()), off_road=int(lap['off_road'][mask].sum()),
                         no_gas=int((~gas[mask]).sum()), left_max=round(float(lap['offset'][mask].max()), 2) if mask.any() else 0.0,
                         right_max=round(float(-lap['offset'][mask].min()), 2) if mask.any() else 0.0))
    return hold, rows


def load_refs():
    if not REFS.exists():
        raise SystemExit(f'{REFS} is missing: run `curate.py refs` once')
    return json.loads(REFS.read_text())


def judge(lap, refs, whole_laps=False, lap_max=None):
    """Decide per section whether its frames are used. Returns a dict that is also what curation.json stores."""
    hold, rows = lap_sections(lap)
    lap_time = None if np.isnan(lap['lap_time']) else float(lap['lap_time'])
    effective = None if lap_time is None else round(lap_time - hold / 50, 2)
    problems = []
    if int(lap['mismatch']):
        problems.append(f"replay differs from the recording in {int(lap['mismatch'])} frames")
    if not bool(lap['finished']):
        problems.append('lap not finished')
    out = []
    for row in rows:
        ref = refs['sections'][row['name']]
        limit = ref['good'] + ref['tol']
        reason = ''
        if problems:
            reason = 'lap unusable'
        elif row['off_road']:
            reason = f"off road {row['off_road']}f"
        elif row['no_gas'] > 5:
            reason = f"no gas {row['no_gas']}f"
        elif row['frames'] > limit:
            reason = f"slow +{row['frames'] - ref['good']}f"
        out.append(dict(name=row['name'], frames=row['frames'], ref=ref['good'], limit=limit, off_road=row['off_road'],
                        no_gas=row['no_gas'], left_max=row['left_max'], right_max=row['right_max'], good=not reason, reason=reason))
    whole_ok = (not problems and float(lap['progress']) >= 100.0 and not any(r['off_road'] for r in out)
                and (lap_max is None or (effective is not None and effective <= lap_max)))
    if whole_laps:
        for row in out:
            row['good'] = whole_ok
            row['reason'] = '' if whole_ok else 'whole lap rejected'
    return dict(frames=int(len(lap['cls'])), finished=bool(lap['finished']), lap_time=lap_time, effective_lap=effective,
                progress=round(float(lap['progress']), 2), center=round(float(lap['center']), 2), start_hold=hold,
                no_gas_after_start=int((~np.isin(lap['cls'][hold:], GAS_CLASSES)).sum()),
                brake_frames=int(np.isin(lap['cls'], (2, 7, 8)).sum()), off_road_frames=int(lap['off_road'].sum()),
                max_abs_offset=round(float(np.abs(lap['offset']).max()), 2), problems=problems, whole_lap_ok=whole_ok, sections=out)


def tighten(verdicts, target, min_laps):
    """Drop the slowest used sections (largest excess over the reference) until the predicted lap time reaches `target`,
    never leaving a section with fewer than min_laps laps. Returns the number of sections dropped."""
    dropped = 0
    while True:
        counts = [sum(v['sections'][i]['good'] for v in verdicts) for i in range(len(SECTIONS))]
        if 0 in counts:
            return dropped
        means = [np.mean([v['sections'][i]['frames'] for v in verdicts if v['sections'][i]['good']]) for i in range(len(SECTIONS))]
        if sum(means) / 50 <= target + 1e-9:
            return dropped
        candidates = [(row['frames'] - row['ref'], lap_index, i) for lap_index, v in enumerate(verdicts)
                      for i, row in enumerate(v['sections']) if row['good'] and counts[i] > min_laps and row['frames'] > means[i]]
        if not candidates:
            return dropped
        _, lap_index, i = max(candidates)
        verdicts[lap_index]['sections'][i].update(good=False, reason='tightened')
        dropped += 1


def discover(folders, new):
    found = [Path(f) for f in folders]
    if new:
        for root in (REPO / 'data/collected', REPO / 'data/stage2_raw'):
            if root.is_dir():
                found += sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(PROVIDED_PREFIX)
                                and any(p.glob('observation_*.npy')))
    seen, unique = set(), []
    for folder in found:
        key = folder.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(folder)
    return unique


def print_table(names, verdicts, refs):
    section_names = [name for name, _, _ in SECTIONS]
    print('times are frames (50 per second); a section marked * is not used for training (off road, no gas, or slower than ref + tol)')
    print(f"{'lap':24s} {'lap':>6s} {'-hold':>6s} {'prog':>6s} {'ctr':>5s} {'hold':>4s} {'nogas':>5s} {'offrd':>5s} | "
          + ' '.join(f'{n:>5s}' for n in section_names) + ' | used')
    print(f"{'  reference (set A)':24s} {refs['lap_good']:6.2f} {'':6s} {'':6s} {'':5s} {'':4s} {'':5s} {'':5s} | "
          + ' '.join(f"{refs['sections'][n]['good']:5d}" for n in section_names))
    print(f"{'  centre line (E0)':24s} {refs['lap_centre']:6.2f} {'':6s} {'':6s} {'':5s} {'':4s} {'':5s} {'':5s} | "
          + ' '.join(f"{refs['sections'][n]['centre']:5d}" for n in section_names))
    print(f"{'  limit (ref + tol)':24s} {'':6s} {'':6s} {'':6s} {'':5s} {'':4s} {'':5s} {'':5s} | "
          + ' '.join(f"{refs['sections'][n]['good'] + refs['sections'][n]['tol']:5d}" for n in section_names))
    for name, v in zip(names, verdicts):
        lap = f"{v['lap_time']:.2f}" if v['lap_time'] is not None else '-'
        eff = f"{v['effective_lap']:.2f}" if v['effective_lap'] is not None else '-'
        cells = ' '.join(f"{row['frames']:>4d}{' ' if row['good'] else '*'}" for row in v['sections'])
        used = sum(row['good'] for row in v['sections'])
        note = ('  ' + '; '.join(v['problems'])) if v['problems'] else ''
        print(f"{name[:24]:24s} {lap:>6s} {eff:>6s} {v['progress']:6.2f} {v['center']:5.1f} {v['start_hold']:4d} "
              f"{v['no_gas_after_start']:5d} {v['off_road_frames']:5d} | {cells} | {used}/{len(v['sections'])}{note}")


def coverage(verdicts, refs, min_laps):
    """Per section: how many laps supply it and the mean time of those; the sum predicts the lap time of the trained model."""
    section_names = [name for name, _, _ in SECTIONS]
    counts, means, weak = [], [], []
    for position, name in enumerate(section_names):
        used = [v['sections'][position]['frames'] for v in verdicts if v['sections'][position]['good']]
        counts.append(len(used))
        means.append(float(np.mean(used)) if used else float('nan'))
        if len(used) < min_laps:
            weak.append(name)
    predicted = None if any(np.isnan(m) for m in means) else round(sum(means) / 50, 2)
    print(f"{'  laps used per section':24s} {'':41s} | " + ' '.join(f'{c:5d}' for c in counts))
    print(f"{'  mean frames of used':24s} {'':41s} | " + ' '.join((f'{m:5.0f}' if not np.isnan(m) else '    -') for m in means))
    print(f"predicted lap time of a model trained on the used sections: {predicted if predicted is not None else 'n/a (a section has no lap)'} s"
          f" (reference laps {refs['lap_good']:.2f}, centre line {refs['lap_centre']:.2f})")
    if weak:
        print(f"sections with fewer than {min_laps} usable laps: {', '.join(weak)}  -> record more laps that drive these well")
    return dict(counts=dict(zip(section_names, counts)), mean_frames=dict(zip(section_names, [None if np.isnan(m) else round(m, 1) for m in means])),
                predicted_lap=predicted, weak=weak)


def cmd_inspect(args):
    refs = load_refs()
    folders = discover(args.folders, args.new)
    if not folders:
        print('no lap folders found (nothing new under data/collected or data/stage2_raw)')
        return
    verdicts = [judge(replay(f), refs, args.whole_laps, args.lap_max) for f in folders]
    names = [f.name for f in folders]
    if args.target and not args.whole_laps:
        print(f'--target {args.target}: {tighten(verdicts, args.target, args.min_laps)} slower sections left out to reach it')
    print_table(names, verdicts, refs)
    summary = coverage(verdicts, refs, args.min_laps)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(laps=dict(zip(names, verdicts)), coverage=summary), ensure_ascii=False, indent=1))


def cmd_build(args):
    refs = load_refs()
    folders = discover(args.folders, args.new)
    if not folders:
        raise SystemExit('no lap folders found')
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        if not (out / 'curation.json').exists():
            raise SystemExit(f'{out} exists, is not empty and was not made by this tool: choose another --out')
        if not args.dry_run:
            shutil.rmtree(out)                      # derived data: rebuilt from the untouched originals every time
    laps = [replay(f) for f in folders]
    verdicts = [judge(lap, refs, args.whole_laps, args.lap_max) for lap in laps]
    names = [f.name for f in folders]
    if len(set(names)) != len(names):
        raise SystemExit('two lap folders have the same name')
    if args.target and not args.whole_laps:
        print(f'--target {args.target}: {tighten(verdicts, args.target, args.min_laps)} slower sections left out to reach it')
    print_table(names, verdicts, refs)
    summary = coverage(verdicts, refs, args.min_laps)
    total = 0
    for folder, lap, verdict in zip(folders, laps, verdicts):
        hold, rows = lap_sections(lap)
        keep = np.zeros(len(lap['cls']), dtype=bool)
        for row, decided in zip(rows, verdict['sections']):
            if decided['good']:
                keep |= row['mask']
        keep &= np.isin(lap['cls'], GAS_CLASSES)        # single frames without gas are left out as well
        verdict['frames_used'] = int(keep.sum())
        total += int(keep.sum())
        if args.dry_run or not keep.any():
            continue
        target = out / folder.name
        target.mkdir(parents=True)
        for number in lap['number'][keep]:
            for kind in ('observation', 'action'):
                shutil.copy2(folder / f'{kind}_{number:05d}.npy', target / f'{kind}_{number:05d}.npy')
    print(f"frames used: {total} from {sum(v['frames_used'] > 0 for v in verdicts)} laps"
          + (' (dry run, nothing written)' if args.dry_run else f' -> {out}'))
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'curation.json').write_text(json.dumps(dict(
            source={name: str(Path(f).resolve()) for name, f in zip(names, folders)}, whole_laps=args.whole_laps,
            lap_max=args.lap_max, min_laps=args.min_laps, target=args.target, laps=dict(zip(names, verdicts)), coverage=summary, frames_used=total),
            ensure_ascii=False, indent=1))


def cmd_refs(args):
    """Reference section times: median over the scripted inside-line laps (set A) and the centre-line lap E0."""
    good_dirs = sorted((SP / 'data_s2/scriptA').iterdir())
    centre_dir = SP / 'data_s2/E0/lap1'
    good = [lap_sections(replay(d))[1] for d in good_dirs]
    centre = lap_sections(replay(centre_dir))[1]
    table = {}
    for position, (name, a, b) in enumerate(SECTIONS):
        frames = sorted(lap[position]['frames'] for lap in good)
        ref = int(round(float(np.median(frames))))
        gain = centre[position]['frames'] - ref
        # a bend counts as driven well when at least half of the reference gain is there; straights get 2 frames of slack
        table[name] = dict(start=a, end=b, good=ref, good_min=frames[0], good_max=frames[-1], centre=centre[position]['frames'],
                           tol=max(2, int(gain // 2)))
    refs = dict(sections=table, lap_good=round(float(np.median([float(replay(d)['lap_time']) for d in good_dirs])), 2),
                lap_centre=round(float(replay(centre_dir)['lap_time']), 2), pad=PAD)
    REFS.write_text(json.dumps(refs, indent=1))
    for name, row in table.items():
        print(f"{name:4s} {row['start']:6.1f}-{row['end']:6.1f}  good {row['good']:4d} ({row['good_min']}-{row['good_max']})  centre {row['centre']:4d}  tol {row['tol']}")
    print('lap good', refs['lap_good'], 'lap centre', refs['lap_centre'], '->', REFS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'build'):
        p = sub.add_parser(name)
        p.add_argument('folders', nargs='*')
        p.add_argument('--new', action='store_true', help='all new lap folders under data/collected and data/stage2_raw')
        p.add_argument('--whole-laps', action='store_true', help='use a lap completely or not at all (clean, progress 100, --lap-max)')
        p.add_argument('--lap-max', type=float, help='with --whole-laps: largest lap time after removing the start hold')
        p.add_argument('--min-laps', type=int, default=4, help='warn when a section is supplied by fewer laps')
        p.add_argument('--target', type=float, help='leave out the slowest used sections until the predicted lap time is at most this '
                                                    '(each section keeps at least --min-laps laps)')
        if name == 'inspect':
            p.add_argument('--json')
        else:
            p.add_argument('--out', required=True)
            p.add_argument('--dry-run', action='store_true')
    sub.add_parser('refs')
    args = parser.parse_args()
    dict(inspect=cmd_inspect, build=cmd_build, refs=cmd_refs)[args.command](args)


if __name__ == '__main__':
    main()
