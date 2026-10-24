"""Per-lap agreement of one checkpoint with the recorded keys (laps it was trained on score clearly higher).
usage: lap_acc.py MODEL.pth LAP_ROOT [LAP_ROOT ...]     Start through the kit:  k job -- python lap_acc.py ..."""
import sys
from pathlib import Path

REPO = Path('/home/sukyun02/workspace/aiad/assignment1')
sys.path.insert(0, str(REPO / 'src'))
import numpy as np
import torch
from model_io import load_model

device = torch.device('cuda')
model, checkpoint = load_model(str(Path(sys.argv[1]).resolve()), device, allow_equivalent_source=True)
model.eval()
model.device = device
print(Path(sys.argv[1]).name, 'epoch', checkpoint.get('epoch'))
with torch.inference_mode():
    for root in sys.argv[2:]:
        for lap in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            paths = sorted(lap.glob('observation_*.npy'))
            obs = torch.as_tensor(np.stack([np.load(p) for p in paths]), dtype=torch.float32, device=device)
            act = [torch.as_tensor(np.load(p.with_name(p.name.replace('observation_', 'action_', 1)))) for p in paths]
            labels = torch.cat(model.actions_to_classes(act)).numpy()
            predicted = torch.cat([model(obs[i:i + 1024]).argmax(1) for i in range(0, len(obs), 1024)]).cpu().numpy()
            steer = labels >= 3
            print(f"{Path(root).name:16s} {lap.name}  frames {len(labels):5d}  agreement {100 * (labels == predicted).mean():6.2f}%  "
                  f"on steering frames {100 * (predicted[steer] == labels[steer]).mean():6.2f}%")
