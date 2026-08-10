"""
Diagnostic for the "healthy rice leaf -> not_rice_leaf" bug.

Run:  python diagnostics/diagnose.py  [optional/path/to/your_test_image.jpg]

Tests three hypotheses in order:
  H1. app.py loads the WRONG checkpoints (os.listdir order is arbitrary)
  H2. The model learned "square aspect ratio == rice leaf" (shortcut learning)
  H3. Individual ensemble members disagree / have random weights
"""
import os
from pathlib import Path

# This script lives in diagnostics/ but reads train/, validation/ and
# saved_models/ from the project root, so resolve paths explicitly
# rather than depending on the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
import sys
import glob
import json

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models

IMG_SIZE = 224
CLASS_NAMES_STAGE1 = [
    'bacterial_leaf_blight', 'brown_spot', 'healthy',
    'leaf_blast', 'leaf_scald', 'narrow_brown_spot', 'not_rice_leaf'
]
device = torch.device('cpu')

# ----------------------------------------------------------------------------
# H1: which checkpoints does app.py actually load?
# ----------------------------------------------------------------------------
print("=" * 78)
print("H1  CHECKPOINT SELECTION")
print("=" * 78)

d1 = os.path.join('saved_models', 'stage1_models')
listing = os.listdir(d1)

cfg_path = os.path.join('production_deployment', 'production_config.json')
intended = {}
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    for k, v in cfg['stage1']['models'].items():
        intended[k] = os.path.basename(v['path'].replace('\\', '/'))

print(f"{'model':18s} {'app.py loads':38s} {'config says':38s} match")
print("-" * 100)
mismatch = False
for name in ['efficientnet_b3', 'densenet121', 'mobilenetv3']:
    matches = [f for f in listing if f.startswith(name) and f.endswith('.pth')]
    picked = matches[0] if matches else "!!! NONE -> RANDOM WEIGHTS !!!"
    want = intended.get(name, '?')
    ok = (picked == want)
    if not ok:
        mismatch = True
    print(f"{name:18s} {picked:38s} {want:38s} {'OK' if ok else 'MISMATCH'}")

if mismatch:
    print("\n  >> app.py is loading STALE checkpoints from an earlier training run.")
    print("     os.listdir() order is arbitrary; matching_files[0] is a coin flip.")

# ----------------------------------------------------------------------------
# Model builders (must match app.py exactly)
# ----------------------------------------------------------------------------
def create_efficientnet_b3(n):
    m = models.efficientnet_b3(weights=None)
    f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(f, 512), nn.ReLU(),
        nn.BatchNorm1d(512), nn.Dropout(0.3), nn.Linear(512, n))
    return m

def create_densenet121(n):
    m = models.densenet121(weights=None)
    f = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(f, 512), nn.ReLU(),
        nn.BatchNorm1d(512), nn.Dropout(0.3), nn.Linear(512, n))
    return m

def create_mobilenetv3_large(n):
    m = models.mobilenet_v3_large(weights=None)
    f = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Linear(f, 512), nn.Hardswish(), nn.Dropout(0.4),
        nn.Linear(512, 256), nn.Hardswish(), nn.Dropout(0.2),
        nn.Linear(256, n))
    return m

BUILDERS = {
    'efficientnet_b3': create_efficientnet_b3,
    'densenet121': create_densenet121,
    'mobilenetv3': create_mobilenetv3_large,
}

def load_stage1(use_config: bool):
    """use_config=False reproduces app.py's buggy selection."""
    loaded = {}
    for name, build in BUILDERS.items():
        m = build(7).to(device)
        if use_config and name in intended:
            fn = intended[name]
        else:
            cand = [f for f in listing if f.startswith(name) and f.endswith('.pth')]
            fn = cand[0] if cand else None
        if fn is None:
            print(f"  WARNING: no checkpoint for {name} -> RANDOM WEIGHTS")
        else:
            ckpt = torch.load(os.path.join(d1, fn), map_location=device, weights_only=False)
            m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        loaded[name] = m
    return loaded

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(modelset, pil_img, show_each=False):
    x = tf(pil_img.convert('RGB')).unsqueeze(0).to(device)
    probs = []
    with torch.no_grad():
        for nm, m in modelset.items():
            p = torch.softmax(m(x), dim=1).cpu().numpy()
            probs.append(p)
            if show_each:
                i = int(p.argmax())
                print(f"      {nm:18s} -> {CLASS_NAMES_STAGE1[i]:22s} {p[0][i]*100:6.2f}%")
    avg = np.mean(probs, axis=0)[0]
    i = int(avg.argmax())
    return CLASS_NAMES_STAGE1[i], float(avg[i]), avg

print("\nLoading Stage 1 ensemble (as app.py does)...")
ens = load_stage1(use_config=False)
print("done.\n")

# ----------------------------------------------------------------------------
# H2: aspect-ratio shortcut
# ----------------------------------------------------------------------------
print("=" * 78)
print("H2  ASPECT-RATIO SHORTCUT")
print("=" * 78)
print("All 6 rice classes in train/ are 1600x1600 SQUARE.")
print("not_rice_leaf is 93% NON-SQUARE.")
print("Test: take a real rice-leaf training image, change ONLY its aspect ratio.\n")

def _find_reference_image():
    """A known-good square training image, used as the control."""
    for c in ['bacterial_leaf_blight', 'healthy', 'leaf_blast']:
        hits = sorted(glob.glob(f'train/{c}/*'))
        if hits:
            return hits[0]
    return None

# --- resolve the user's image, with a graceful fallback ---------------------
user_img_path = None
if len(sys.argv) > 1:
    if os.path.exists(sys.argv[1]):
        user_img_path = sys.argv[1]
    else:
        print(f"  (!) '{sys.argv[1]}' not found - looking for loose images instead\n")

if user_img_path is None:
    loose = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        loose.extend(glob.glob(ext))
    loose = sorted(set(loose))
    if loose:
        print(f"  Found in project root: {', '.join(loose)}")
        user_img_path = loose[0]
        print(f"  Using: {user_img_path}\n")

ref_path = _find_reference_image()
if ref_path is None:
    print("No training images found - run this from the project root.")
    sys.exit(1)

test_img_path = user_img_path or ref_path
print(f"Control (training) image : {ref_path}")
img_ref = Image.open(ref_path).convert('RGB')
print(f"   size {img_ref.size}  aspect {img_ref.size[0]/img_ref.size[1]:.3f}")
if user_img_path:
    print(f"Your image               : {user_img_path}")
    _u = Image.open(user_img_path)
    print(f"   size {_u.size}  aspect {_u.size[0]/_u.size[1]:.3f}")
print()

img = Image.open(test_img_path).convert('RGB')

def center_square(im):
    w, h = im.size
    if w == h:
        return im
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


# --- Test A: distort a KNOWN-GOOD square training image ---------------------
print("-" * 78)
print("A. Control image (square, from train/) - distort aspect ratio only")
print("-" * 78)
w, h = img_ref.size
lb = Image.new('RGB', (int(h * 4 / 3), h), (0, 0, 0))
lb.paste(img_ref, ((lb.width - w) // 2, 0))

variants = [
    ("original (square 1:1)",    img_ref),
    ("cropped to 4:3",           img_ref.crop((0, int(h * .125), w, int(h * .875)))),
    ("cropped to 16:9",          img_ref.crop((0, int(h * .219), w, int(h * .781)))),
    ("cropped to 3:4 portrait",  img_ref.crop((int(w * .125), 0, int(w * .875), h))),
    ("letterboxed to 4:3",       lb),
]

print(f"{'variant':26s} {'size':14s} {'aspect':8s} {'prediction':22s} conf")
print("-" * 78)
flipped = False
for label, v in variants:
    cls, conf, _ = predict(ens, v)
    a = v.size[0] / v.size[1]
    flag = ""
    if cls == 'not_rice_leaf':
        flag = "  <-- REJECTED"
        if 'square' not in label:
            flipped = True
    print(f"{label:26s} {str(v.size):14s} {a:<8.3f} {cls:22s} {conf*100:6.2f}%{flag}")

print()
if flipped:
    print("  >> CONFIRMED. Same pixels, only the shape changed, and the prediction")
    print("     flipped to not_rice_leaf. The model keyed on aspect ratio.")
else:
    print("  >> Aspect ratio alone did not flip the control image.")

# --- Test B: the user's actual failing image, raw vs center-cropped ---------
if user_img_path:
    print()
    print("-" * 78)
    print(f"B. YOUR image ({os.path.basename(user_img_path)}) - raw vs center-cropped to square")
    print("-" * 78)
    u = Image.open(user_img_path).convert('RGB')
    us = center_square(u)

    print(f"{'version':26s} {'size':14s} {'aspect':8s} {'prediction':22s} conf")
    print("-" * 78)
    c_raw, p_raw, _ = predict(ens, u)
    print(f"{'as uploaded':26s} {str(u.size):14s} {u.size[0]/u.size[1]:<8.3f} {c_raw:22s} {p_raw*100:6.2f}%")
    c_sq, p_sq, _ = predict(ens, us)
    print(f"{'center-cropped square':26s} {str(us.size):14s} {us.size[0]/us.size[1]:<8.3f} {c_sq:22s} {p_sq*100:6.2f}%")

    print()
    if c_raw == 'not_rice_leaf' and c_sq != 'not_rice_leaf':
        print("  >> THIS IS YOUR BUG. Your photo is rejected as-is, but accepted once")
        print("     center-cropped to a square. CENTER_CROP_TO_SQUARE=True in app.py")
        print("     already applies this. It is a mitigation - the real fix is")
        print("     retraining on data where aspect ratio does not encode the label.")
    elif c_raw == 'not_rice_leaf' and c_sq == 'not_rice_leaf':
        print("  >> Still rejected even when square. Aspect ratio is not the only")
        print("     factor here - check the per-model votes in H3 below, and compare")
        print("     your photo's lighting/zoom/background against train/ samples.")
    else:
        print(f"  >> Your image is classified as '{c_raw}' - not rejected.")

# ----------------------------------------------------------------------------
# H3: per-model breakdown + stale vs config checkpoints
# ----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("H3  PER-MODEL VOTES  (original square image)")
print("=" * 78)
predict(ens, img, show_each=True)

if mismatch and intended:
    print("\n" + "-" * 78)
    print("Same image, but loading the checkpoints production_config.json intends:")
    print("-" * 78)
    ens2 = load_stage1(use_config=True)
    predict(ens2, img, show_each=True)
    c1, p1, _ = predict(ens, img)
    c2, p2, _ = predict(ens2, img)
    print(f"\n  app.py checkpoints  -> {c1} ({p1*100:.2f}%)")
    print(f"  config checkpoints  -> {c2} ({p2*100:.2f}%)")

# ----------------------------------------------------------------------------
# Batch sanity check
# ----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BATCH SANITY CHECK  (10 training images per class)")
print("=" * 78)
print("If accuracy is high here but your own photos fail, the model is fine and")
print("the problem is a train/serve distribution gap (image shape, camera, crop).\n")

for cls in CLASS_NAMES_STAGE1:
    files = sorted(glob.glob(f'train/{cls}/*'))[:10]
    if not files:
        continue
    correct = 0
    preds = []
    for f in files:
        try:
            p, _, _ = predict(ens, Image.open(f))
            preds.append(p)
            if p == cls:
                correct += 1
        except Exception:
            pass
    if preds:
        from collections import Counter
        top = Counter(preds).most_common(2)
        got = ', '.join(f'{k}:{v}' for k, v in top)
        print(f"  {cls:22s} {correct}/{len(preds)} correct   (predicted: {got})")

print("\nDone.")
