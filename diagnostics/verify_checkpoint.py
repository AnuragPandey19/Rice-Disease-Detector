"""
Verify a suspected bug in model.ipynb: the saved checkpoint may contain the
FINAL epoch's weights while reporting the BEST epoch's accuracy.

The training loop does:

    best_val_acc  = 0.0
    best_model_state = None
    for epoch in ...:
        ...
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_model_state = model.state_dict().copy()   # <-- shallow copy
    torch.save({'model_state_dict': best_model_state,
                'best_val_acc':     best_val_acc, ...})

`OrderedDict.copy()` is a SHALLOW copy. The new dict holds references to the
same tensor objects. Adam updates parameters in place, so those tensors keep
changing for every remaining epoch. `best_model_state` therefore tracks the
live weights, not a snapshot.

Net effect: the checkpoint stores the last epoch's weights, labelled with the
best epoch's accuracy.

This script re-evaluates each checkpoint on the validation set and compares the
measured accuracy against the `best_val_acc` stored inside it. It also replays
the shallow-copy behaviour on a toy tensor so you can see the mechanism.

Run:  python diagnostics/verify_checkpoint.py
"""
import os
from pathlib import Path

# This script lives in diagnostics/ but reads train/, validation/ and
# saved_models/ from the project root, so resolve paths explicitly
# rather than depending on the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import transforms, models, datasets

device = torch.device('cpu')

# ---------------------------------------------------------------------------
# Part 1 - demonstrate the mechanism on a toy state dict (no models needed)
# ---------------------------------------------------------------------------
print("=" * 74)
print("PART 1 - does .copy() snapshot the weights?")
print("=" * 74)

w = torch.tensor([1.0, 2.0, 3.0])
live = OrderedDict([('weight', w)])

snapshot_shallow = live.copy()                                   # what the notebook does
snapshot_deep = OrderedDict((k, v.clone()) for k, v in live.items())  # what it should do

w.add_(100.0)   # simulate an optimizer step after the "best" epoch

print(f"  original tensor after later training : {live['weight'].tolist()}")
print(f"  .copy()      snapshot                : {snapshot_shallow['weight'].tolist()}   <-- followed the change")
print(f"  .clone()     snapshot                : {snapshot_deep['weight'].tolist()}")
print()
if torch.equal(snapshot_shallow['weight'], live['weight']):
    print("  CONFIRMED: .copy() does not snapshot. It tracks the live tensors.")
    print("  So best_model_state ends up holding the FINAL epoch weights.")
else:
    print("  .copy() snapshotted correctly - the bug does not apply on this build.")
print()

# ---------------------------------------------------------------------------
# Part 2 - re-evaluate the real checkpoints
# ---------------------------------------------------------------------------
print("=" * 74)
print("PART 2 - re-evaluate each checkpoint on the validation set")
print("=" * 74)

if not os.path.isdir('validation'):
    print("  validation/ not found - run from the project root. Skipping Part 2.")
    raise SystemExit(0)


def eff(n):
    m = models.efficientnet_b3(weights=None)
    f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(.4), nn.Linear(f, 512), nn.ReLU(),
                                 nn.BatchNorm1d(512), nn.Dropout(.3), nn.Linear(512, n))
    return m


def dense(n):
    m = models.densenet121(weights=None)
    f = m.classifier.in_features
    m.classifier = nn.Sequential(nn.Dropout(.5), nn.Linear(f, 512), nn.ReLU(),
                                 nn.BatchNorm1d(512), nn.Dropout(.3), nn.Linear(512, n))
    return m


def mob(n):
    m = models.mobilenet_v3_large(weights=None)
    f = m.classifier[0].in_features
    m.classifier = nn.Sequential(nn.Linear(f, 512), nn.Hardswish(), nn.Dropout(.4),
                                 nn.Linear(512, 256), nn.Hardswish(), nn.Dropout(.2),
                                 nn.Linear(256, n))
    return m


BUILDERS = {'efficientnet_b3': eff, 'densenet121': dense, 'mobilenetv3': mob}
d1 = os.path.join('saved_models', 'stage1_models')

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val = datasets.ImageFolder('validation', transform=tf)
loader = torch.utils.data.DataLoader(val, batch_size=32, shuffle=False, num_workers=0)
print(f"  validation set: {len(val)} images, {len(val.classes)} classes")
print(f"  classes: {val.classes}\n")

print(f"{'checkpoint':44s} {'claimed':>9s} {'measured':>9s} {'delta':>8s}")
print("-" * 74)

for name, build in BUILDERS.items():
    for fn in sorted(f for f in os.listdir(d1) if f.startswith(name + '_') and f.endswith('.pth')):
        ck = torch.load(os.path.join(d1, fn), map_location=device, weights_only=False)
        m = build(7).to(device)
        m.load_state_dict(ck['model_state_dict'])
        m.eval()

        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                pred = m(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        measured = 100.0 * correct / total
        claimed = ck.get('best_val_acc', float('nan'))
        delta = measured - claimed
        flag = "  <-- mismatch" if abs(delta) > 0.5 else ""
        print(f"{fn:44s} {claimed:8.2f}% {measured:8.2f}% {delta:+7.2f}%{flag}")

print()
print("=" * 74)
print("HOW TO READ THIS")
print("=" * 74)
print("  measured == claimed   -> checkpoint really holds the best-epoch weights")
print("  measured <  claimed   -> confirms the shallow-copy bug: the file holds")
print("                           final-epoch weights but is labelled with the")
print("                           best epoch's score")
print()
print("  Compare 'measured' against the 'Final Validation Accuracy' figures in")
print("  saved_models/training_summary_report.txt. If measured matches FINAL")
print("  rather than BEST, the bug is confirmed.")
