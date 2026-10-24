"""rob lens custom driver: kit.drive with extra perturbations (threads, logit noise, forced single-frame flips).
Wraps the candidate's ClassificationNetwork in a subclass; kit files are not modified."""
import sys, json, argparse
from pathlib import Path
KIT = '/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad/kit'
sys.path.insert(0, KIT)
import kit
import torch

p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('--device', default='cpu')
p.add_argument('--threads', type=int, default=1)
p.add_argument('--logit-noise', type=float, default=0.0)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--flip', default='')          # comma list of frame indices where top-2 logits are swapped
p.add_argument('--hud', default='none')
p.add_argument('--noise', type=int, default=0)
p.add_argument('--noise-seed', type=int, default=0)
p.add_argument('--trace')
a = p.parse_args()
torch.set_num_threads(a.threads)
flips = {int(v) for v in a.flip.split(',') if v}
gen = torch.Generator().manual_seed(a.seed)
orig = kit.load_network_class
counter = [0]

def patched(path, name='cand_network'):
    Net, source = orig(path, name)
    class Perturbed(Net):
        def forward(self, observation):
            s = Net.forward(self, observation)
            i = counter[0]; counter[0] += 1
            if a.logit_noise:
                s = s + (a.logit_noise * torch.randn(s.shape, generator=gen)).to(s.device)
            if i in flips:
                top = torch.topk(s[0], 2).indices
                s = s.clone()
                v0, v1 = s[0, top[0]].item(), s[0, top[1]].item()
                s[0, top[0]] = v1 - 1e-3
                s[0, top[1]] = v0
            return s
    return Perturbed, source

kit.load_network_class = patched
with kit.Slot('eval', kit.EVAL_SLOTS):
    r = kit.drive(a.model, a.device, hud=a.hud, trace=bool(a.trace), noise=a.noise, noise_seed=a.noise_seed)
if a.trace:
    Path(a.trace).write_text(json.dumps(r.pop('trace')))
print(json.dumps(r))
