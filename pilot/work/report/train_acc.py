"""Training-set accuracy of one official checkpoint (how well it reproduces the recorded keys).
usage: train_acc.py MODEL.pth DATA_DIR      Start through the kit:  k job -- python train_acc.py ..."""
import json
import sys
from pathlib import Path

REPO = Path('/home/sukyun02/workspace/aiad/assignment1')
sys.path.insert(0, str(REPO / 'src'))
import numpy as np
import torch
from model_io import load_model

model_path, data_dir = sys.argv[1], Path(sys.argv[2])
device = torch.device('cuda')
model, checkpoint = load_model(str(Path(model_path).resolve()), device, allow_equivalent_source=True)
model.eval()
model.device = device
paths = sorted(data_dir.rglob('observation_*.npy'))
labels, predicted = [], []
with torch.inference_mode():
    for start in range(0, len(paths), 1024):
        chunk = paths[start:start + 1024]
        obs = np.stack([np.load(p) for p in chunk])
        act = [torch.as_tensor(np.load(p.with_name(p.name.replace('observation_', 'action_', 1)))) for p in chunk]
        labels.append(torch.cat(model.actions_to_classes(act)).numpy())
        predicted.append(model(torch.as_tensor(obs, dtype=torch.float32, device=device)).argmax(1).cpu().numpy())
labels, predicted = np.concatenate(labels), np.concatenate(predicted)
per_class = {int(c): dict(count=int((labels == c).sum()), recall=round(float((predicted[labels == c] == c).mean()), 4),
                          predicted=int((predicted == c).sum())) for c in np.unique(labels)}
print(json.dumps(dict(model=model_path, data=str(data_dir), frames=len(labels), laps=len({p.parent for p in paths}),
                      epoch=checkpoint.get('epoch'), accuracy=round(float((labels == predicted).mean()), 4), per_class=per_class)))
