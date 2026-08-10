"""
Decisive test: is the model keying on BACKGROUND rather than leaf morphology?

Run:  python diagnostics/test_background.py

Direction A - your field photo (t2.jpg), cropped progressively tighter so less
              background is visible. If background is the trigger, the
              prediction should stop being not_rice_leaf as the crop tightens.

Direction B - the reverse. A known-good TRAINING leaf (which the model
              classifies correctly at 100%) composited onto a busy natural
              background vs a plain one. Identical leaf pixels in both.
              If B1 flips to not_rice_leaf and B2 does not, the model is
              reading the background, not the leaf.
"""
import os
from pathlib import Path

# This script lives in diagnostics/ but reads train/, validation/ and
# saved_models/ from the project root, so resolve paths explicitly
# rather than depending on the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
import glob

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models

CLASS_NAMES = ['bacterial_leaf_blight', 'brown_spot', 'healthy',
               'leaf_blast', 'leaf_scald', 'narrow_brown_spot', 'not_rice_leaf']
device = torch.device('cpu')
d1 = os.path.join('saved_models', 'stage1_models')


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


print("Loading ensemble...")
ens = {}
for name, build in [('efficientnet_b3', eff), ('densenet121', dense), ('mobilenetv3', mob)]:
    m = build(7).to(device)
    cand = sorted(f for f in os.listdir(d1) if f.startswith(name + '_') and f.endswith('.pth'))
    ck = torch.load(os.path.join(d1, cand[-1]), map_location=device, weights_only=False)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    ens[name] = m
print("done.\n")

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(pil):
    x = tf(pil.convert('RGB')).unsqueeze(0)
    ps = []
    with torch.no_grad():
        for m in ens.values():
            ps.append(torch.softmax(m(x), 1).numpy())
    avg = np.mean(ps, 0)[0]
    i = int(avg.argmax())
    return CLASS_NAMES[i], float(avg[i]), float(avg[CLASS_NAMES.index('not_rice_leaf')])


# --- build variants ---------------------------------------------------------
if not os.path.exists('diagnostics/images/t2.jpg'):
    print("t2.jpg not found in project root.")
    raise SystemExit(1)

t2 = Image.open('diagnostics/images/t2.jpg').convert('RGB')
ref_path = sorted(glob.glob('train/bacterial_leaf_blight/*'))[0]
leaf = Image.open(ref_path).convert('RGB')
busy = Image.open(sorted(glob.glob('train/not_rice_leaf/*'))[0]).convert('RGB').resize(leaf.size)

lw, lh = leaf.size
strip = leaf.crop((int(lw * .30), 0, int(lw * .70), lh))
b1 = busy.copy()
b1.paste(strip, (int(lw * .30), 0))
b2 = Image.new('RGB', leaf.size, (245, 242, 238))
b2.paste(strip, (int(lw * .30), 0))

variants = [
    ("A0  t2.jpg full field photo",      t2),
    ("A1  t2 cropped to leaf cluster",   t2.crop((175, 210, 330, 520))),
    ("A2  t2 cropped tighter",           t2.crop((200, 250, 310, 470))),
    ("A3  t2 single blade, no bg",       t2.crop((215, 280, 300, 430))),
    ("--", None),
    ("B0  training leaf, untouched",     leaf),
    ("B1  SAME leaf on BUSY bg",         b1),
    ("B2  SAME leaf on PLAIN bg",        b2),
]

print("=" * 86)
print("BACKGROUND HYPOTHESIS TEST")
print("=" * 86)
print(f"{'variant':34s} {'prediction':24s} {'conf':>8s}  {'P(not_rice_leaf)':>16s}")
print("-" * 86)

results = {}
for label, v in variants:
    if v is None:
        print("-" * 86)
        continue
    cls, conf, p_not = predict(v)
    results[label[:2]] = cls
    mark = "  <-- REJECTED" if cls == 'not_rice_leaf' else ""
    print(f"{label:34s} {cls:24s} {conf*100:7.2f}%  {p_not*100:15.2f}%{mark}")

print("=" * 86)
print("\nVERDICT")
print("-" * 86)

a_freed = any(results.get(k) not in (None, 'not_rice_leaf') for k in ('A1', 'A2', 'A3'))
b_flip = results.get('B1') == 'not_rice_leaf' and results.get('B2') != 'not_rice_leaf'

if b_flip:
    print("  CONFIRMED (reverse test). Identical leaf pixels: on a plain background")
    print("  the model classifies the disease correctly; on a busy natural background")
    print("  it says not_rice_leaf. The model is reading the BACKGROUND.")
elif a_freed:
    print("  LIKELY. Cropping background out of your field photo changes the verdict.")
else:
    print("  Not reproduced by background alone. The gap may also involve zoom level,")
    print("  focus, lighting, or leaf scale relative to the frame.")

print()
print("  Root cause: every image in the 6 rice classes is a single isolated leaf on")
print("  plain light paper. The not_rice_leaf class is stock photos of gardens, pets,")
print("  food, landscapes. The model never had to learn leaf morphology - background")
print("  statistics alone separate the classes perfectly on this dataset.")
print()
print("  That is why validation accuracy is 96.68% and real field photos still fail.")
