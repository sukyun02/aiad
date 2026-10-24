import sys, numpy as np, torch
np.set_printoptions(linewidth=250, precision=3, suppress=True)
SP='/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad'
sys.path.insert(0, SP+'/kit'); import kit
d = np.load(SP+'/cache/data.npz')
obs, lap, frame, labels = d['obs'], d['lap'], d['frame'], d['labels']
N = len(obs)
def feats(name, o):
    Net, _ = kit.load_network_class(f'{SP}/cands/{name}/network.py', 'v_'+name)
    m = Net(device='cpu')
    t = torch.as_tensor(o).float()
    s, a, st, g = m.extract_sensor_values(t.clone(), len(t))
    return torch.cat([s, a, st, g], 1).numpy()
rng = np.random.default_rng(0)
zero = obs.copy(); zero[:, 86:92, 11:30] = 0
shift = obs.copy(); shift[:, 86:92, 11:30] = np.roll(shift[:, 86:92, 11:30], 1, axis=2)
swap = obs.copy(); perm = rng.permutation(N); swap[:, 86:92, 11:30] = obs[perm][:, 86:92, 11:30]     # digits of a random other frame
swapall = obs.copy(); swapall[:, 84:92] = obs[perm][:, 84:92]; swapall[:, 88, 36:] = obs[:, 88, 36:]  # whole text band rows 84-91 from another frame (keeps steer/gyro row segment)
late = frame >= 150
names = ['speed', 'abs0', 'abs1', 'abs2', 'abs3', 'steer', 'gyro']
for cand in ['cnnA', 'hud_cs']:
    f0 = feats(cand, obs)
    print(f'\n== {cand}: extract_sensor_values on all {N} cached frames ==')
    print(' cruise (frame>=150) std per feature :', dict(zip(names, np.round(f0[late].std(0), 4))))
    print(' cruise min                          :', dict(zip(names, np.round(f0[late].min(0), 3))))
    print(' cruise max                          :', dict(zip(names, np.round(f0[late].max(0), 3))))
    for label, alt in [('hud_zero (digits blanked)', zero), ('hud_shift (digits rolled 1px)', shift), ('digits swapped with random frame', swap), ('rows 84-91 text band swapped with random frame', swapall)]:
        f1 = feats(cand, alt)
        print(f' max |feature change| under {label:48s}:', dict(zip(names, np.round(np.abs(f1 - f0).max(0), 4))))
    print(' first second, lap 3: frame label | speed abs0 abs1 abs2 abs3 steer gyro')
    idx = np.where(lap == 3)[0]; idx = idx[np.argsort(frame[idx])]
    for i in list(idx[:60:4]) + list(idx[60:140:20]):
        print('  ', frame[i], labels[i], '|', f0[i])
    if cand == 'hud_cs':
        z = (f0[:, 0] == 0)
        print(' frames with clean speed == 0:', int(z.sum()), ' label histogram there:', np.bincount(labels[z], minlength=9).tolist(), ' max frame index with speed==0:', int(frame[z].max()))
        print(' label-0 frames:', int((labels == 0).sum()), ' of which clean speed == 0:', int(((labels == 0) & z).sum()), ' label-0 frame index range:', int(frame[labels == 0].min()), int(frame[labels == 0].max()))
        for thr in (0.0,):
            pass
        print(' per lap: first frame with label 1 / first frame with clean speed>0:', [(int(frame[(lap == L) & (labels != 0)].min()), int(frame[(lap == L) & (f0[:, 0] > 0)].min())) for L in range(7)])
        print(' unique clean speed values:', np.unique(np.round(f0[:, 0], 3)).tolist()[:40])
        print(' frames (frame>=150) where any of the 5 clean values deviates from its median:', int((np.abs(f0[late][:, :5] - np.median(f0[late][:, :5], 0)) > 1e-6).any(1).sum()), 'of', int(late.sum()))
# fc input layout of hud_crop_sg: check that the two pad layers deliver [steer, gyro, conv...]
Net, _ = kit.load_network_class(f'{SP}/cands/hud_crop_sg/network.py', 'v_sg')
m = Net(device='cpu').eval()
x = torch.cat([torch.arange(1024.).repeat(3, 1), torch.tensor([[-1., -2, -3, -4, -5, 77, 88]]).repeat(3, 1)], 1)
y = m.fc[1](m.fc[0](x))
print('\nhud_crop_sg: fc[0:2] maps 1031 ->', tuple(y.shape), ' first 4 entries', y[0, :4].tolist(), ' last 2', y[0, -2:].tolist(), '(expect 77, 88, 0, 1 ... 1022, 1023)')
Net, _ = kit.load_network_class(f'{SP}/cands/hud_crop_ns/network.py', 'v_ns')
m2 = Net(device='cpu').eval(); y2 = m2.fc[0](x); print('hud_crop_ns: fc[0] maps 1031 ->', tuple(y2.shape), ' last 2', y2[0, -2:].tolist())
# end-to-end invariance of the logits to the digits (random-init nets, 256 random cached frames)
sel = rng.choice(N, 256, replace=False)
for cand in ['cnnA', 'cnnA_crop', 'hud_cs', 'hud_crop_cs', 'hud_crop_sg', 'hud_crop_ns']:
    Net, _ = kit.load_network_class(f'{SP}/cands/{cand}/network.py', 'e_'+cand)
    torch.manual_seed(0); net = Net(device='cpu').eval()
    with torch.inference_mode():
        a = net(torch.as_tensor(obs[sel]).float()); b = net(torch.as_tensor(zero[sel]).float()); c = net(torch.as_tensor(swap[sel]).float())
    print(f'{cand:12s} max |logit change| digits blanked: {float((a-b).abs().max()):.6f}   digits swapped: {float((a-c).abs().max()):.6f}')
