"""Offline indicators for every existing checkpoint, computed on the recorded data (eval mode)."""
import json, sys, glob
from pathlib import Path
SP = Path('/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad')
sys.path.insert(0, str(SP / 'kit')); import kit
import numpy as np, torch
D = np.load(SP / 'cache/data.npz')
order = np.lexsort((D['frame'], D['lap']))
obs = torch.as_tensor(D['obs'][order]); lab = D['labels'][order]; lap = D['lap'][order]; frame = D['frame'][order]
along = np.concatenate([np.array(json.load(open(SP / f'expert/lap{i}.json')))[:, 0] for i in range(7)])
assert len(along) == len(lab)
obs_z = obs.clone(); obs_z[:, 86:92, 11:30, :] = 0
torch.manual_seed(0); torch.set_num_threads(6)
obs_n = (obs.to(torch.int16) + torch.randint(-2, 3, obs.shape, dtype=torch.int16)).clamp(0, 255).to(torch.uint8)
dirs = np.zeros(9); dirs[[3, 5, 7]] = -1; dirs[[4, 6, 8]] = 1
ldir = dirs[lab]
T = np.load(SP / 'work/diag/track.npz')
def predict(model, o):
    with torch.inference_mode():
        return torch.cat([model(o[i:i + 1024].cuda().float()).cpu() for i in range(0, len(o), 1024)]).float()
def bursts(d):
    out = []; i = 0
    while i < len(d):
        if d[i] != 0:
            j = i
            while j + 1 < len(d) and d[j + 1] == d[i] and lap[j + 1] == lap[i]: j += 1
            out.append((i, j, d[i])); i = j + 1
        else: i += 1
    return out
B = bursts(ldir)
models = []
for c, tag in [('cnnA', 'b256_e100'), ('cnnA_crop', 'b256_e100'), ('base', 'b256_e100'), ('base', 'b4096_e100'), ('cnnA', 'b4096_e100')]:
    for p in sorted(glob.glob(str(SP / f'cands/{c}/runs/{tag}/s*.pth'))):
        st = Path(p).stem
        models.append((f"{c}_{tag.split('_')[0]}_{st if '_e' in st else st + '_e100'}", p))
models.append(('baseline', str(SP / 'blobs/baseline.pth')))
rows = []
for name, p in models:
    Net, _ = kit.load_network_class(Path(p).with_suffix('.network.py'), 'n')
    m = Net(device='cuda'); m.load_state_dict(torch.load(p, map_location='cpu', weights_only=True)['state_dict']); m.device = torch.device('cuda'); m.eval()
    lg = predict(m, obs); pr = lg.argmax(1).cpu().numpy()
    lt = torch.as_tensor(lab)
    loss = torch.nn.functional.cross_entropy(lg, lt).item()
    top = torch.topk(lg, 2, dim=1).values; margin = (top[:, 0] - top[:, 1]).cpu().numpy()
    pz = predict(m, obs_z).argmax(1).cpu().numpy(); pn = predict(m, obs_n).argmax(1).cpu().numpy()
    pdir = dirs[pr]
    r = dict(name=name, acc=(pr == lab).mean(), loss=loss,
             rec_L=(pr[lab == 5] == 5).mean(), rec_R=(pr[lab == 6] == 6).mean(), rec_0=(pr[lab == 0] == 0).mean(), rec_gas=(pr[lab == 1] == 1).mean(),
             prec_L=(lab[pr == 5] == 5).mean() if (pr == 5).any() else np.nan, prec_R=(lab[pr == 6] == 6).mean() if (pr == 6).any() else np.nan,
             psr=(pdir != 0).mean(), pL=(pdir < 0).mean(), pR=(pdir > 0).mean(),
             wrong_dir=((pdir * ldir) < 0).mean(), false0=((pr == 0) & (frame > 60)).mean(), pred0_after60=int(((pr == 0) & (frame > 60)).sum()),
             bal_acc=np.mean([(pr[lab == k] == k).mean() for k in (0, 1, 5, 6)]),
             margin_med=float(np.median(margin)), low_margin=float((margin < 0.5).mean()),
             flip_hudzero=(pz != pr).mean(), flip_noise2=(pn != pr).mean(),
             acc_hudzero=(pz == lab).mean())
    # burst-level: detected within burst, onset lag
    det = {-1: [], 1: []}; lag = {-1: [], 1: []}
    for i, j, d in B:
        hit = np.where(pdir[i:j + 1] == d)[0]
        det[d].append(len(hit) > 0); lag[d].append(hit[0] if len(hit) else (j - i + 1))
    r['burst_det_L'] = np.mean(det[-1]); r['burst_det_R'] = np.mean(det[1])
    r['burst_lag_L'] = np.mean(lag[-1]); r['burst_lag_R'] = np.mean(lag[1])
    # windowed net steering error: 25-frame (0.5 s) sums of (pred_dir - label_dir), per lap
    e = pdir - ldir; W = 25; worst = []; mae = []
    for k in range(7):
        s = np.convolve(e[lap == k], np.ones(W), 'valid'); worst.append(np.abs(s).max()); mae.append(np.abs(s).mean())
    r['net25_mae'] = np.mean(mae); r['net25_worst'] = np.mean(worst)
    # per 40 m track bin: steering-direction recall on steering frames; take the worst bin (bins with >=40 steering frames) and corner-only recall
    rec = []
    for b in range(0, 960, 40):
        msk = (along >= b) & (along < b + 40) & (ldir != 0) & (frame > 60)
        if msk.sum() >= 40: rec.append(((pdir[msk] == ldir[msk]).mean(), b))
    r['binrec_min'] = min(rec)[0]; r['binrec_min_at'] = min(rec)[1]; r['binrec_p10'] = float(np.percentile([x[0] for x in rec], 10))
    rc = (along >= 740) & (along < 780) & (lab == 6); r['rec_R_chicane'] = (pr[rc] == 6).mean()
    # zoom-in
    z = frame < 50
    r['zoom_pred0'] = (pr[z] == 0).mean(); r['zoom_lab0'] = (lab[z] == 0).mean(); r['zoom_steer'] = (pdir[z] != 0).mean()
    rows.append({k: (float(v) if not isinstance(v, str) else v) for k, v in r.items()})
    np.save(str(SP / f'work/diag/traces/offpred_{name}.npy'), np.stack([pr, margin]).astype(np.float32))
json.dump(rows, open(SP / 'work/diag/offline.json', 'w'), indent=1)
np.savez(str(SP / 'work/diag/offline_data.npz'), lab=lab, lap=lap, frame=frame, along=along)
keys = ['acc', 'loss', 'rec_L', 'rec_R', 'rec_0', 'prec_L', 'prec_R', 'psr', 'wrong_dir', 'pred0_after60', 'flip_hudzero', 'flip_noise2', 'burst_det_L', 'burst_det_R', 'burst_lag_L', 'burst_lag_R', 'net25_mae', 'net25_worst', 'binrec_min', 'binrec_min_at', 'rec_R_chicane', 'zoom_pred0', 'margin_med']
print('name'.ljust(28) + ' '.join(k[:9].rjust(9) for k in keys))
for r in rows: print(r['name'].ljust(28) + ' '.join(f"{r[k]:9.3f}" for k in keys))
print('bursts L', sum(1 for b in B if b[2] < 0), 'R', sum(1 for b in B if b[2] > 0), 'median len', np.median([b[1] - b[0] + 1 for b in B]))
