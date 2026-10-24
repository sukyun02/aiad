"""Fact-check harness: runs the OFFICIAL src/demonstrations.record_demonstrations() loop with a scripted keyboard.

Nothing of the official loop is re-implemented.  Only these are replaced:
  - pygame.event.get / pygame.key.get_pressed  -> scripted key events and key states
  - demonstrations.prepare_gui                 -> no-op, so SDL stays on the 'dummy' video driver (no window on the desktop)
  - frame_pacing.FramePacer.tick               -> no sleeping (the lap then runs much faster than 50 fps wall clock)
  - gui.make_driving_env                       -> wrapped only to keep a reference to the env
  - sys.modules['torch']                       -> empty stub: src/evaluation.py does `import torch` at module level and
                                                  record_demonstrations imports it when a lap completes; print_metrics needs no torch
What could NOT be tested this way: the real SDL keyboard state of a real window (here get_pressed is scripted).

usage: harness.py SCENARIO   (S1 | S2 | S3)   -> prints the official terminal lines + '[H]' harness lines, writes out/harness_<S>.json
"""
import json
import os
import re
import shutil
import sys
import time
import types
from pathlib import Path

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
REPO = Path('/home/sukyun02/workspace/aiad/assignment1')
SP = Path('/home/sukyun02/workspace/aiad/pilot')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'src'))
sys.modules['torch'] = types.ModuleType('torch')      # see docstring

import numpy as np
import pygame

import demonstrations
import frame_pacing
import gui

demonstrations.prepare_gui = lambda: None
frame_pacing.FramePacer.tick = lambda self, fps: 0.0


class Ctx:
    env = None
    held = frozenset()
    replay = None
    replay_i = 0
    pressed_calls = 0
    iterations = 0
    display_changes = []
    last_display = None


ctx = Ctx()
_make = gui.make_driving_env


def make_env(*args, **kwargs):
    ctx.env = _make(*args, **kwargs)
    return ctx.env


gui.make_driving_env = make_env


class FakeKeys:
    def __init__(self, keys):
        self.keys = keys

    def __getitem__(self, key):
        return key in self.keys


def fake_pressed():
    ctx.pressed_calls += 1
    if ctx.replay is not None:
        a = ctx.replay[ctx.replay_i] if ctx.replay_i < len(ctx.replay) else (0.0, 0.0, 0.0)
        ctx.replay_i += 1
        keys = set()
        if a[0] < -0.1:
            keys.add(pygame.K_LEFT)
        if a[0] > 0.1:
            keys.add(pygame.K_RIGHT)
        if a[1] > 0.1:
            keys.add(pygame.K_UP)
        if a[2] > 0.1:
            keys.add(pygame.K_DOWN)
        return FakeKeys(keys)
    return FakeKeys(set(ctx.held))


def status():
    return ctx.env.unwrapped.gui_status


def frames():
    return ctx.env.unwrapped.gui_frames


def metrics():
    return ctx.env.unwrapped.metrics


def say(text):
    print('[H] ' + text, flush=True)


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k)


def start_replay(actions):
    ctx.replay, ctx.replay_i = actions, 0


def load_lap(folder):
    paths = sorted(Path(folder).glob('action_*.npy'))
    return [np.load(p) for p in paths]


def state_line(tag):
    m = metrics()
    say(f"{tag}: gui_status={status()} recorded_frames={frames()} metrics.frames={m.frames} elapsed={m.elapsed:.2f} "
        f"finished={m.finished} aborted={m.aborted} lap_time={m.lap_time} progress={m.totals[0]:.2f} centre={m.totals[1]:.2f} "
        f"off_road_frames={m.off_road_frames}")


def wait(condition, limit=40000):
    """yield empty event lists until condition() holds"""
    n = 0
    while not condition():
        n += 1
        if n > limit:
            raise RuntimeError('harness wait limit')
        yield []


def scenario_S1(result, folder):
    lap = load_lap(SP / 'data_s2/scriptA/lap1')
    result['source_lap'] = dict(folder=str(SP / 'data_s2/scriptA/lap1'), frames=len(lap), first_action=[float(v) for v in lap[0]])
    for _ in range(5):
        yield []
    state_line('ready, before any key')
    say('TAB in the ready state ->')
    yield [key(pygame.K_TAB)]
    yield []
    state_line('after TAB in ready state')
    say('UP is already held (scripted), now ENTER ->')
    start_replay(lap)
    t0 = time.perf_counter()
    yield [key(pygame.K_RETURN)]
    state_line('one loop iteration after ENTER')
    result['frames_recorded_in_the_enter_iteration'] = frames()
    yield from wait(lambda: frames() >= 500)
    say('TAB during recording (frame 500) ->')
    yield [key(pygame.K_TAB)]
    yield []
    state_line('after TAB during recording')
    yield from wait(lambda: frames() >= 600)
    f_before = frames()
    say('ENTER during recording (frame 600) ->')
    yield [key(pygame.K_RETURN)]
    yield []
    state_line('after ENTER during recording')
    result['enter_during_recording_frames_before_after'] = [f_before, frames()]
    yield from wait(lambda: status() == 'finished')
    wall = time.perf_counter() - t0
    state_line('lap 1 finished')
    result['lap1'] = dict(recorded_frames=frames(), lap_time=metrics().lap_time, progress=float(metrics().totals[0]),
                          centre=float(metrics().totals[1]), wall_seconds=round(wall, 2),
                          wall_fps=round(frames() / wall, 1))
    f_fin, calls = frames(), ctx.pressed_calls
    say('ENTER after the lap is finished ->')
    ctx.held = frozenset({pygame.K_UP})
    ctx.replay = None
    yield [key(pygame.K_RETURN)]
    for _ in range(20):
        yield []
    state_line('20 iterations after ENTER on a finished lap (UP held)')
    result['enter_after_finish'] = dict(frames_before=f_fin, frames_after=frames(), key_polls_before=calls, key_polls_after=ctx.pressed_calls,
                                        status=status())
    say('TAB after the lap is finished ->')
    yield [key(pygame.K_TAB)]
    yield []
    state_line('after TAB on a finished lap')
    sessions = sorted(p.name for p in Path(folder).iterdir())
    result['after_first_tab_sessions'] = sessions
    calls = ctx.pressed_calls
    for _ in range(20):
        yield []
    state_line('20 iterations after the save, UP still held, no ENTER')
    result['after_save_without_enter'] = dict(status=status(), frames=frames(), key_polls_added=ctx.pressed_calls - calls)
    say('second lap: ENTER, then SPACE once finished ->')
    start_replay(lap)
    yield [key(pygame.K_RETURN)]
    yield from wait(lambda: status() == 'finished')
    state_line('lap 2 finished')
    result['lap2'] = dict(recorded_frames=frames(), lap_time=metrics().lap_time)
    yield [key(pygame.K_SPACE)]
    yield []
    state_line('after SPACE on a finished, unsaved lap')
    result['after_space_sessions'] = sorted(p.name for p in Path(folder).iterdir())
    say('third start: ENTER with UP held, 30 frames, SPACE during recording, ENTER + 25 frames, then ESC ->')
    ctx.replay = None
    ctx.held = frozenset({pygame.K_UP})
    yield [key(pygame.K_RETURN)]
    yield from wait(lambda: frames() >= 30)
    yield [key(pygame.K_SPACE)]
    yield []
    state_line('after SPACE during recording')
    yield [key(pygame.K_RETURN)]
    yield from wait(lambda: frames() >= 25)
    state_line('before ESC')
    yield [key(pygame.K_ESCAPE)]


def scenario_S2(result, folder):
    for _ in range(3):
        yield []
    say('ENTER with no key held; UP only from frame 20; then LEFT+UP for 45 frames, then UP only (drives off the road) ->')
    ctx.held = frozenset()
    yield [key(pygame.K_RETURN)]
    yield from wait(lambda: frames() >= 20)
    ctx.held = frozenset({pygame.K_UP})
    yield from wait(lambda: frames() >= 120)
    state_line('frame 120 (on the road)')
    ctx.held = frozenset({pygame.K_UP, pygame.K_LEFT})
    yield from wait(lambda: frames() >= 165)
    ctx.held = frozenset({pygame.K_UP})
    first_off = None
    track = []
    while status() not in ('aborted', 'finished'):
        m = metrics()
        if m.off_road and first_off is None:
            first_off = m.frames
            state_line('first off-road frame')
        if first_off is not None and (m.frames - first_off) in (1, 50, 100, 200, 248, 249):
            track.append((m.frames, m.off_road_frames, round(float(m.totals[0]), 3)))
        yield []
    state_line('recording stopped')
    m = metrics()
    result['abort'] = dict(status=status(), first_off_road_frame=first_off, frames_at_stop=m.frames, off_road_frames=m.off_road_frames,
                           off_road_seconds=m.off_road_seconds, progress=float(m.totals[0]), lap_time=m.lap_time,
                           samples_frame_offroadframes_progress=track)
    f = frames()
    say('TAB on an aborted lap ->')
    yield [key(pygame.K_TAB)]
    yield []
    say('ENTER on an aborted lap ->')
    yield [key(pygame.K_RETURN)]
    for _ in range(10):
        yield []
    state_line('after TAB and ENTER on an aborted lap')
    result['after_abort_tab_enter'] = dict(status=status(), frames_before=f, frames_after=frames(),
                                           sessions=sorted(p.name for p in Path(folder).iterdir()) if Path(folder).exists() else [])
    say('SPACE ->')
    yield [key(pygame.K_SPACE)]
    yield []
    state_line('after SPACE')
    result['after_space_status'] = status()
    # second part: leave the road for a short time, come back, check that the 5 s counter restarts and progress stays below 100
    say('ENTER, UP held; short excursions are covered by scenario S3; now ESC')
    yield [key(pygame.K_ESCAPE)]


def scenario_S3(result, folder):
    source = HERE / 'data' / 'offroad_lap'
    lap = load_lap(source)
    result['source_lap'] = dict(folder=str(source), frames=len(lap))
    for _ in range(3):
        yield []
    say('replaying a lap that leaves the road for a moment and still finishes: ENTER ->')
    start_replay(lap)
    yield [key(pygame.K_RETURN)]
    spans, current, prev_prog = [], None, 0.0
    lost = []
    while status() not in ('aborted', 'finished'):
        m = metrics()
        if m.off_road and current is None:
            current = [m.frames, None, float(m.totals[0])]
        if not m.off_road and current is not None:
            current[1] = m.frames
            spans.append(current)
            current = None
        yield []
    m = metrics()
    state_line('recording stopped')
    result['lap'] = dict(status=status(), lap_time=m.lap_time, progress=float(m.totals[0]), centre=float(m.totals[1]),
                         recorded_frames=frames(), off_road_spans_first_offroad_frame__first_onroad_frame=[(a, b) for a, b, _ in spans],
                         off_road_frames_total=sum(b - a for a, b, _ in spans),
                         scored_distance=m.scored_distance, length=m.length,
                         unscored_distance=m.length - m.scored_distance)
    say('TAB on a finished lap with progress < 100 ->')
    yield [key(pygame.K_TAB)]
    yield []
    state_line('after TAB')
    result['sessions_after_tab'] = sorted(p.name for p in Path(folder).iterdir()) if Path(folder).exists() else []
    yield [key(pygame.K_ESCAPE)]


def main():
    name = sys.argv[1]
    folder = HERE / 'harness_data' / name
    if folder.exists():
        shutil.rmtree(folder)          # only ever inside factcheck/harness_data
    result = dict(scenario=name)
    gen = {'S1': scenario_S1, 'S2': scenario_S2, 'S3': scenario_S3}[name](result, folder)
    folder.mkdir(parents=True)

    def fake_get(*args, **kwargs):
        ctx.iterations += 1
        if ctx.env is not None and getattr(ctx.env.unwrapped, 'metrics', None) is not None:
            m = metrics()
            shown = (round(float(m.display_totals[0]), 4), round(float(m.display_totals[1]), 4), round(float(m.display_totals[2]), 4))
            if shown != ctx.last_display:
                ctx.last_display = shown
                ctx.display_changes.append((m.frames, *shown))
        try:
            return next(gen)
        except StopIteration:
            return [pygame.event.Event(pygame.QUIT)]

    pygame.event.get = fake_get
    pygame.key.get_pressed = fake_pressed
    demonstrations.record_demonstrations(str(folder), None, language='kor')
    say('record_demonstrations returned')
    result['torch_is_stub'] = not hasattr(sys.modules['torch'], 'nn')
    result['loop_iterations'] = ctx.iterations
    result['panel_value_changes_first20__metricsframe_progress_centre_time'] = ctx.display_changes[:20]
    sessions = sorted(p for p in folder.iterdir())
    saved = []
    for session in sessions:
        obs = sorted(session.glob('observation_*.npy'))
        act = sorted(session.glob('action_*.npy'))
        entry = dict(name=session.name, name_matches_YYYYMMDD_HHMMSS_micro=bool(re.fullmatch(r'\d{8}_\d{6}_\d{6}', session.name)),
                     observations=len(obs), actions=len(act), first=obs[0].name, last=obs[-1].name,
                     obs_shape=list(np.load(obs[0]).shape), obs_dtype=str(np.load(obs[0]).dtype),
                     act_dtype=str(np.load(act[0]).dtype), first_action=[float(v) for v in np.load(act[0])])
        if 'source_lap' in result:
            src = Path(result['source_lap']['folder'])
            same_act = sum(int(np.array_equal(np.load(a), np.load(src / a.name))) for a in act if (src / a.name).exists())
            same_obs = sum(int(np.array_equal(np.load(o), np.load(src / o.name))) for o in obs if (src / o.name).exists())
            entry.update(actions_identical_to_source=same_act, observations_identical_to_source=same_obs)
        saved.append(entry)
    result['saved_sessions'] = saved
    (HERE / 'out' / f'harness_{name}.json').write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    say('result written to ' + str(HERE / 'out' / f'harness_{name}.json'))
    shutil.rmtree(folder)              # the saved copy is not needed after the comparison (58 MB per lap)


if __name__ == '__main__':
    main()
