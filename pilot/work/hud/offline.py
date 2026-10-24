"""Offline (CPU) digit-sensitivity of trained checkpoints on the cached expert frames."""
import sys, glob, json, numpy as np, torch
SP='/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad'
sys.path.insert(0, SP+'/kit'); import kit
torch.set_num_threads(4)
d = np.load(SP+'/cache/data.npz')
obs, lap, frame, labels = d['obs'], d['lap'], d['frame'], d['labels']
N = len(obs); rng = np.random.default_rng(0)
zero = obs.copy(); zero[:, 86:92, 11:30] = 0
perm = rng.permutation(N); swap = obs.copy(); swap[:, 86:92, 11:30] = obs[perm][:, 86:92, 11:30]
lab = torch.as_tensor(labels)
def logits(net, o):
    out = []
    with torch.inference_mode():
        for i in range(0, N, 1024):
            out.append(net(torch.as_tensor(o[i:i+1024]).float()))
    return torch.cat(out)
def load(path, netpath=None):
    Net, _ = kit.load_network_class(netpath or path.replace('.pth', '.network.py'), 'o')
    net = Net(device='cpu'); net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict']); net.device = torch.device('cpu'); return net.eval()
specs = sys.argv[1:]
rows = []
for spec in specs:
    cand, tag, ep = spec.split(':')
    for path in sorted(glob.glob(f'{SP}/cands/{cand}/runs/{tag}/s?' + ('' if ep == 'final' else f'_e{ep}') + '.pth')):
        net = load(path)
        a, z, s = logits(net, obs), logits(net, zero), logits(net, swap)
        pa, pz, ps = a.argmax(1), z.argmax(1), s.argmax(1)
        r = dict(cand=cand, ep=ep, seed=path.split('/')[-1][:2], acc=round(float((pa == lab).float().mean()), 4), acc_zero=round(float((pz == lab).float().mean()), 4),
                 acc_swap=round(float((ps == lab).float().mean()), 4), flip_zero=round(float((pa != pz).float().mean()), 4), flip_swap=round(float((pa != ps).float().mean()), 4),
                 dlogit_swap_mean=round(float((a - s).abs().max(1).values.mean()), 3))
        rows.append(r); print(json.dumps(r), flush=True)
import collections
g = collections.defaultdict(list)
for r in rows: g[(r['cand'], r['ep'])].append(r)
print('\ncand ep n | acc acc_zero acc_swap | flip_zero flip_swap | mean max|dlogit| swap   (means over seeds)')
for k, v in g.items():
    m = lambda f: np.mean([x[f] for x in v])
    print(f'{k[0]:10s} {k[1]:>5s} {len(v)} | {m("acc"):.4f} {m("acc_zero"):.4f} {m("acc_swap"):.4f} | {m("flip_zero"):.4f} {m("flip_swap"):.4f} | {m("dlogit_swap_mean"):.3f}')
