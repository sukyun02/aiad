import sys, numpy as np, torch
SP='/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad'
sys.path.insert(0, SP+'/kit'); import kit
torch.set_num_threads(4)
d = np.load(SP+'/cache/data.npz'); obs = d['obs']
x = torch.as_tensor(obs[::15]).float()
NetA, _ = kit.load_network_class(SP+'/cands/cnnA/network.py', 'A'); NetC, _ = kit.load_network_class(SP+'/cands/cnnA_crop/network.py', 'C')
for p in ['cnnA/runs/b256_e100/s0.pth', 'cnnA_crop/runs/b256_e100/s1.pth']:
    sd = torch.load(SP+'/cands/'+p, map_location='cpu', weights_only=True)['state_dict']
    a = NetA(device='cpu').eval(); c = NetC(device='cpu').eval()
    print(p, 'keys A', list(a.state_dict().keys())[:3], 'keys C', list(c.state_dict().keys())[:3])
    sdA = dict(zip(a.state_dict().keys(), sd.values())); sdC = dict(zip(c.state_dict().keys(), sd.values()))
    a.load_state_dict(sdA); c.load_state_dict(sdC)
    with torch.inference_mode():
        la, lc = a(x), c(x)
    print('  same weights in cnnA and cnnA_crop on', len(x), 'real frames: max |logit diff| =', float((la - lc).abs().max()), ' argmax differs on', int((la.argmax(1) != lc.argmax(1)).sum()), 'frames')
# receptive field of the conv: gradient of conv output sum w.r.t. input pixels
a = NetA(device='cpu'); xi = torch.rand(1, 3, 96, 96, requires_grad=True); a.conv(xi).abs().sum().backward()
g = xi.grad.abs().sum(1)[0]
print('cnnA conv: rows with nonzero gradient', int(torch.where(g.sum(1) > 0)[0].min()), '..', int(torch.where(g.sum(1) > 0)[0].max()), ' cols', int(torch.where(g.sum(0) > 0)[0].min()), '..', int(torch.where(g.sum(0) > 0)[0].max()))
# same-seed init equality
torch.manual_seed(3); a = NetA(device='cpu'); torch.manual_seed(3); c = NetC(device='cpu')
print('same-seed init identical:', all(torch.equal(p, q) for p, q in zip(a.parameters(), c.parameters())))
