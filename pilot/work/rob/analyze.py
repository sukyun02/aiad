import json, glob, os, sys, statistics as st
from collections import defaultdict
import numpy as np
W = os.path.dirname(os.path.abspath(__file__))
R = {}
for f in glob.glob(W + '/out/*.json'):
    cid, probe = os.path.basename(f)[:-5].split('__')
    try: R[(cid, probe)] = json.loads(open(f).read())
    except Exception as e: print('bad', f, e)
cks = sorted({c for c, _ in R}, key=lambda c: (c.split('-')[0], int(c.split('-')[1][1:]), c))
probes = ['cuda', 'cpu', 'thr4', 'ln3', 'hudzero', 'hudshift', 'huddim', 'n1s0', 'n1s1', 'n1s2', 'n2s0', 'n2s1', 'n2s2', 'n4s0', 'n4s1', 'n4s2']
probes += sorted({p for _, p in R} - set(probes))
def cell(r):
    if r is None: return '   .  '
    if r['status'] != 'finished': return 'A%3d%% ' % r['progress']
    return '%5.2f%s' % (r['lap_time'], ' ' if r['progress'] >= 99.99 else '*')
def trace(cid, probe):
    f = f'{W}/trace/{cid}__{probe}.json'
    return json.load(open(f)) if os.path.exists(f) else None
def group(c):
    a, e, s = c.split('-'); return a + '-' + e

if 'table' in sys.argv or len(sys.argv) == 1:
    print('## per-checkpoint outcomes: lap time (s); * = finished but progress<100 (drove off-road part); A nn% = aborted at nn% progress; . = not run')
    print('%-20s' % 'ckpt' + ''.join('%-9s' % p for p in probes))
    for c in cks:
        print('%-20s' % c + ''.join('%-9s' % cell(R.get((c, p))) for p in probes))

if 'dev' in sys.argv or len(sys.argv) == 1:
    print('\n## Q1 device determinism: trace comparison (class sequence) cuda vs cpu, cpu vs thr4')
    for c in cks:
        out = []
        for a, b in (('cuda', 'cpu'), ('cpu', 'thr4')):
            ta, tb = trace(c, a), trace(c, b)
            if ta is None or tb is None: out.append(f'{a}/{b}: n/a'); continue
            n = min(len(ta), len(tb))
            div = next((i for i in range(n) if ta[i][2] != tb[i][2]), None)
            md = max(abs(ta[i][3] - tb[i][3]) for i in range(n if div is None else div + 1))
            same = (div is None and len(ta) == len(tb))
            out.append(f'{a}/{b}: ' + ('IDENTICAL(%d fr, max|dmargin|=%.3f)' % (n, md) if same else f'DIVERGE at frame {div} of {n} (margin there {ta[div][3] if div is not None else None})'))
        print('%-20s' % c, ' | '.join(out))

def nominal(c):
    return R.get((c, 'cpu'))
PERT = ['thr4', 'ln3', 'hudzero', 'hudshift', 'huddim', 'n1s0', 'n1s1', 'n1s2', 'n2s0', 'n2s1', 'n2s2', 'n4s0', 'n4s1', 'n4s2']
FAM = {'float(thr4)': ['thr4'], 'logit1e-3': ['ln3'], 'noise1': ['n1s0', 'n1s1', 'n1s2'], 'noise2': ['n2s0', 'n2s1', 'n2s2'], 'noise4': ['n4s0', 'n4s1', 'n4s2'],
       'hudzero': ['hudzero'], 'hudshift': ['hudshift'], 'huddim': ['huddim'], 'flip': sorted({p for _, p in R if p.startswith('flip')})}
def ok(r): return r is not None and r['status'] == 'finished'
def clean(r): return ok(r) and r['progress'] >= 99.99

if 'frag' in sys.argv or len(sys.argv) == 1:
    print('\n## Q2 fragility by group: nominal(cpu) finish, then per probe family finished/n over ALL ckpts of the group, and restricted to ckpts whose nominal finished [fin|nominal-fin]; clean = progress 100')
    groups = defaultdict(list)
    for c in cks: groups[group(c)].append(c)
    hdr = '%-16s %-8s' % ('group', 'nominal') + ''.join('%-22s' % k for k in FAM)
    print(hdr)
    for g, cs in groups.items():
        line = '%-16s %-8s' % (g, '%d/%d' % (sum(ok(nominal(c)) for c in cs), len(cs)))
        for k, ps in FAM.items():
            allr = [R[(c, p)] for c in cs for p in ps if (c, p) in R]
            nomr = [R[(c, p)] for c in cs for p in ps if (c, p) in R and ok(nominal(c))]
            line += '%-22s' % ('%d/%d [%d/%d] cl%d' % (sum(map(ok, allr)), len(allr), sum(map(ok, nomr)), len(nomr), sum(map(clean, allr))))
        print(line)
    print('\n## Q2b flips: nominal finisher -> aborted under perturbation, and nominal aborter -> finished (noise families n1,n2,n4 only; then all perturbations)')
    for g, cs in groups.items():
        for name, ps in (('noise', FAM['noise1'] + FAM['noise2'] + FAM['noise4']), ('all', PERT)):
            f2a = sum(1 for c in cs for p in ps if (c, p) in R and ok(nominal(c)) and not ok(R[(c, p)]))
            nf = sum(1 for c in cs for p in ps if (c, p) in R and ok(nominal(c)))
            a2f = sum(1 for c in cs for p in ps if (c, p) in R and nominal(c) is not None and not ok(nominal(c)) and ok(R[(c, p)]))
            na = sum(1 for c in cs for p in ps if (c, p) in R and nominal(c) is not None and not ok(nominal(c)))
            print('%-16s %-6s fin->abort %d/%d   abort->fin %d/%d' % (g, name, f2a, nf, a2f, na))

if 'pred' in sys.argv or len(sys.argv) == 1:
    print('\n## Q3/Q4 per checkpoint: nominal(cpu) lap properties vs perturbation outcomes (all perturbation probes available)')
    print('%-20s %-7s %6s %6s %6s %6s %5s %6s %6s %6s %6s | %5s %5s %5s %6s %6s %6s %6s' % ('ckpt', 'nomlap', 'center', 'maxoff', 'meanof', 'steer', 'sw', 'offrd', 'm<.1', 'm<.5', 'medmar', 'nPert', 'fin', 'clean', 'lapmin', 'lapmed', 'lapmax', 'lapsd'))
    rows = []
    for c in cks:
        n = nominal(c)
        if n is None: continue
        t = trace(c, 'cpu')
        m = np.array([r[3] for r in t[60:]]) if t else np.array([np.nan])
        pr = [R[(c, p)] for p in PERT + FAM['flip'] if (c, p) in R]
        laps = [r['lap_time'] for r in pr if ok(r)]
        row = dict(c=c, nom=n, fin=sum(map(ok, pr)), clean=sum(map(clean, pr)), n=len(pr), laps=laps,
                   mlt1=float((m < 0.1).mean()), mlt5=float((m < 0.5).mean()), medm=float(np.median(m)))
        rows.append(row)
        print('%-20s %-7s %6.1f %6.2f %6.2f %6.3f %5d %6d %6.3f %6.3f %6.2f | %5d %5d %5d %6s %6s %6s %6s' % (
            c, cell(n).strip(), n['center'], n['max_offset'], n['mean_offset'], n['steer_frac'], n['switches'], n['offroad_frames'],
            row['mlt1'], row['mlt5'], row['medm'], row['n'], row['fin'], row['clean'],
            '%.2f' % min(laps) if laps else '-', '%.2f' % st.median(laps) if laps else '-', '%.2f' % max(laps) if laps else '-',
            '%.3f' % st.pstdev(laps) if len(laps) > 1 else '-'))
    # predictors of fragility among nominal finishers
    from scipy.stats import spearmanr
    fins = [r for r in rows if ok(r['nom']) and r['n'] >= 5]
    if len(fins) >= 5:
        y = np.array([1 - r['fin'] / r['n'] for r in fins]); y2 = np.array([1 - r['clean'] / r['n'] for r in fins])
        print('\nSpearman rho (n=%d nominal finishers) of nominal-lap property vs abort fraction / non-clean fraction under perturbation' % len(fins))
        for name, fn in (('max_offset', lambda r: r['nom']['max_offset']), ('mean_offset', lambda r: r['nom']['mean_offset']), ('center', lambda r: r['nom']['center']),
                         ('steer_frac', lambda r: r['nom']['steer_frac']), ('switches', lambda r: r['nom']['switches']), ('offroad_frames', lambda r: r['nom']['offroad_frames']),
                         ('path_len', lambda r: r['nom']['path_len']), ('lap_time', lambda r: r['nom']['lap_time']),
                         ('frac margin<0.1', lambda r: r['mlt1']), ('frac margin<0.5', lambda r: r['mlt5']), ('median margin', lambda r: r['medm'])):
            x = np.array([fn(r) for r in fins])
            a, pa = spearmanr(x, y); b, pb = spearmanr(x, y2)
            print('  %-18s abort: rho=%+.2f p=%.3f   nonclean: rho=%+.2f p=%.3f' % (name, a, pa, b, pb))
        print('\nThreshold tables (nominal finishers): rule -> ckpts passing: mean abort frac / mean non-clean frac ; failing: same')
        for name, fn in (('max_offset<=5.0', lambda r: r['nom']['max_offset'] <= 5.0), ('max_offset<=6.0', lambda r: r['nom']['max_offset'] <= 6.0), ('max_offset<=6.67', lambda r: r['nom']['max_offset'] <= 6.67),
                         ('offroad_frames==0', lambda r: r['nom']['offroad_frames'] == 0), ('center>=60', lambda r: r['nom']['center'] >= 60), ('lap<=43.5', lambda r: r['nom']['lap_time'] <= 43.5), ('path<=960', lambda r: r['nom']['path_len'] <= 960)):
            P = [r for r in fins if fn(r)]; F = [r for r in fins if not fn(r)]
            def agg(L): return 'n=%d runs fin %d/%d clean %d/%d' % (len(L), sum(r['fin'] for r in L), sum(r['n'] for r in L), sum(r['clean'] for r in L), sum(r['n'] for r in L)) if L else 'n=0'
            print('  %-20s pass: %-40s fail: %s' % (name, agg(P), agg(F)))
    print('\n## Q4 lap-time spread: within one checkpoint under perturbation (finished runs) vs between seeds (nominal cpu)')
    groups = defaultdict(list)
    for r in rows: groups[group(r['c'])].append(r)
    for g, rs in groups.items():
        nom = [r['nom']['lap_time'] for r in rs if ok(r['nom'])]
        within_sd = [st.pstdev(r['laps']) for r in rs if len(r['laps']) > 2]
        within_rng = [max(r['laps']) - min(r['laps']) for r in rs if len(r['laps']) > 2]
        allp = [l for r in rs for l in r['laps']]
        print('%-16s between-seed nominal: n=%d %s | within-ckpt under perturbation: median sd=%s, median range=%s, max range=%s | pooled perturbed laps n=%d med=%s min=%s max=%s' % (
            g, len(nom), ('med=%.2f range=[%.2f,%.2f] sd=%.3f' % (st.median(nom), min(nom), max(nom), st.pstdev(nom))) if nom else '-',
            '%.3f' % st.median(within_sd) if within_sd else '-', '%.2f' % st.median(within_rng) if within_rng else '-', '%.2f' % max(within_rng) if within_rng else '-',
            len(allp), '%.2f' % st.median(allp) if allp else '-', '%.2f' % min(allp) if allp else '-', '%.2f' % max(allp) if allp else '-'))
