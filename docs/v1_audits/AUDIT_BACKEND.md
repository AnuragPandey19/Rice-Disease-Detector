# Backend & Model Audit — Rice Leaf Disease Detection

**Scope:** `app.py` (505 lines), `Dockerfile`, `requirements.txt`, `model.ipynb`, dataset layout, `saved_models/`
**Date:** August 2026
**Auditor:** self-audit prior to v2

Severity key: **Critical** = the system produces wrong results or is exploitable · **High** = real defect under normal operation · **Medium** = correctness/maintainability risk · **Low** = hygiene

Findings are prefixed **M-** (model/data) or **B-** (backend/infra).

---

## Summary

| Severity | Model/Data | Backend/Infra |
|---|---|---|
| Critical | 3 | 1 |
| High | 5 | 5 |
| Medium | 7 | 6 |
| Low | 2 | 4 |

### Status

Every finding that can be fixed in code has been fixed. Two cannot be, and they are the two that matter most.

| ID | Finding | Status |
|---|---|---|
| **M-01** | Model reads background | **NOT FIXED — requires retraining.** No inference-time workaround exists. `v2/` sandbox is scaffolded for it. UI and API now warn users instead of pretending otherwise. |
| **M-02** | No held-out test set | **NOT FIXED — requires re-split and retrain.** `v2/scripts/02_make_splits.py` produces a real 70/15/15 split for v2. |
| **M-12** | Checkpoints hold final-epoch weights | Fixed in `model.ipynb` — `copy.deepcopy`. Existing checkpoints are still affected; re-train or re-verify with `diagnostics/verify_checkpoint.py`. |
| M-03 | `BACTERIAL_DISEASES` misnomer | Fixed — renamed `DISEASE_CLASSES` |
| M-04 | `leaf_scald` called bacterial | Fixed — corrected, plus explicit `pathogen` field with genus/species |
| M-05 | Confidence = product of two softmaxes | Fixed — reports Stage 2 confidence directly; Stage 1 exposed separately |
| M-06 | Duplicate/unused checkpoints | Mitigated — excluded from image via `.dockerignore`; files still on disk pending your decision to delete |
| M-07 | VFlip / 45° rotation on directional pathology | Documented — needs an ablation in v2, not a blind change |
| M-08 | `pretrained=` deprecated | Fixed — `weights=None` |
| M-09 | Hardcoded OneDrive path | Fixed — `RICE_DATA_PATH` env var |
| M-10 | Unrendered f-strings in report | Cosmetic, in a generated artefact; regenerated on next training run |
| M-11 | Stale 3-model Stage 2 docs | Fixed — `saved_models/README.md` corrected |
| M-13 | Stage 2 has no escape hatch | Fixed — abstains below 0.45, returns `uncertain` |
| M-14 | Uniform ensemble weighting | Fixed — accuracy-weighted, toggleable |
| M-15 | Freezing by parameter index | Fixed — `freeze_all_but_last()` by named module |
| M-16 | Focal loss `alpha=1` | Fixed — per-class alpha from inverse frequency |
| M-17 | Scheduler/selection mismatch | Fixed — both on `val_acc`, plus early stopping |
| B-01 | `str(e)` leaked to client | Fixed — generic message + correlation ref |
| B-02 | No upload limit | Fixed — `MAX_CONTENT_LENGTH`, PIL bomb guard, 413 handler |
| B-03 | No rate limiting | Fixed — per-IP sliding window, dependency-free |
| B-04 | CORS open to all | Fixed — allowlist via `ALLOWED_ORIGINS`, off by default |
| B-05 | `/health` always 200 | Fixed — 503 when not ready; `/live` split out |
| B-06 | Bare `except:` swapping architecture | Fixed — fallbacks deleted |
| B-07 | 3.8GB build context | Fixed — `.dockerignore`, selective checkpoint COPY |
| B-08 | HEALTHCHECK needs `requests` | Fixed — stdlib `urllib` |
| B-09 | Container runs as root | Fixed — non-root `appuser` (UID 1000) |
| B-10 | No model reload | Fixed — lazy, retryable, lock-guarded |
| B-11 | Single worker, no strategy | Fixed — threads + `--preload`, rationale documented |
| B-12 | Relative model path | Fixed — resolved against `__file__` |
| B-13 | `numpy` unpinned | Fixed — pinned |
| B-14 | Unused imports | Fixed |
| B-15 | No structured logging | Fixed — request IDs on every response and log line |
| B-16 | No `/metrics` | Fixed — implemented |

**Read the two "NOT FIXED" rows as the real state of the project.** Everything else was hygiene. The system still cannot classify a field photograph, and the accuracy figures still describe a benchmark rather than a capability.

**The headline:** the reported 96.68% / 98.18% accuracy does not measure disease detection. It measures the model's ability to tell studio photographs from stock photographs. Everything else in this document is secondary to M-01.

---

# Part A — Model & Data

## Critical

### M-01 · The model classifies on background, not leaf morphology

**Evidence:** `diagnostics/test_background.py`, reproducible.

The dataset has a structural flaw that gives the classifier a perfect shortcut:

- All 2,100 images across the 6 rice classes are **a single isolated leaf on plain light paper** (white / beige / grey studio backdrop), all 1600×1600.
- All 420 `not_rice_leaf` images are **stock photography of busy natural scenes** — flower gardens, dogs on grass, bowls of food, a mountain range, a green sports car.

Background statistics alone separate those groups perfectly, so the models never had to learn what a rice leaf looks like.

**Direction A** — remove background from a failing field photo:

| Input | Prediction | P(not_rice_leaf) |
|---|---|---|
| Full field photograph | `not_rice_leaf` | **99.08%** |
| Cropped to leaf cluster | `brown_spot` | 0.65% |
| Cropped tighter | `brown_spot` | 0.31% |
| Single blade, no background | `brown_spot` | 1.12% |

**Direction B** — reverse test, identical leaf pixels, background swapped:

| Input | Prediction | Confidence |
|---|---|---|
| Training leaf, untouched | `bacterial_leaf_blight` | 99.87% |
| Same leaf on a **busy** background | `not_rice_leaf` | **84.21%** |
| Same leaf on a **plain** background | `bacterial_leaf_blight` | 79.42% |

Direction B is decisive: byte-identical leaf pixels, only the surround changes, and the verdict flips from correct diagnosis to outright rejection.

A competing hypothesis — that the model keyed on **aspect ratio** (all rice images square, 93% of negatives non-square) — was tested and **rejected**: distorting a training image to 4:3, 16:9, 3:4 and letterbox returned the correct disease at 100% confidence in every variant.

**Impact:** The system is unusable on field photography, which is its entire stated purpose. Reported accuracy is not transferable.

**Fix — requires retraining, no inference-time workaround exists:**

1. Rebuild `not_rice_leaf` with **hard negatives**: other crop leaves (maize, sugarcane, grass, tomato, potato) photographed on the *same plain paper*. PlantVillage is a strong source — its images are already single leaves on uniform backgrounds, which is exactly the distribution needed.
2. Add **field photographs** to the disease classes so background varies within positives too.
3. Optionally add a leaf detection/segmentation stage so the classifier only sees a cropped leaf region.
4. Re-measure on a held-out set of genuine field photos and publish the gap.

---

### M-02 · No held-out test set — reported metrics are validation-set numbers

**Files:** `model.ipynb` cell 3, `saved_models/v1_archive/production_deployment/production_config.json`

The data is split train/validation only (~80/20 by folder). The same validation set was used for:

- early stopping / best-checkpoint selection (`best_val_acc`),
- the decision to drop EfficientNetV2-S from the Stage 2 ensemble,
- the headline accuracy figures published in the README.

Selecting a model on a set and then reporting that set's score overstates generalisation. The +0.91% improvement from dropping EfficientNetV2-S is exactly the size of effect that can be validation-set noise.

**Fix:** Carve a true test split that is touched once, at the end. Re-report.

---

### M-12 · Saved checkpoints hold the FINAL epoch's weights, labelled with the BEST epoch's accuracy

**File:** `model.ipynb` cell 9, `train_model()`

```python
best_val_acc = 0.0
best_model_state = None
for epoch in range(num_epochs):
    ...
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()   # shallow copy
...
torch.save({'model_state_dict': best_model_state,
            'best_val_acc': best_val_acc, ...}, model_path)
```

`OrderedDict.copy()` is a **shallow** copy. The new dict holds references to the same tensor objects, and Adam updates parameters **in place**. So `best_model_state` does not snapshot anything — it tracks the live weights for every remaining epoch.

The checkpoint therefore contains the **last** epoch's weights while advertising the **best** epoch's score. Every downstream number inherits the mismatch: `saved_models/v1_archive/production_deployment/production_config.json`, the README table, the ensemble-pruning decision.

The training report shows exactly the gap this predicts:

| Model | Best val acc (claimed in checkpoint) | Final val acc (actually saved) | Overstated by |
|---|---|---|---|
| efficientnet_b3 | 93.21% | 93.05% | 0.16% |
| densenet121 | 96.05% | 95.42% | 0.63% |
| mobilenetv3 | 94.47% | 93.36% | 1.11% |
| vit_base | 98.41% | 97.27% | 1.14% |
| convnext_tiny | 98.64% | 96.14% | **2.50%** |

Note the last row against **M-06/ensemble pruning**: EfficientNetV2-S was dropped because it scored 94.09% while ConvNeXt-Tiny scored 98.64%. If the deployed ConvNeXt weights are really the 96.14% ones, the margin that justified that decision was overstated by 2.5 points.

**Verify:** `python diagnostics/verify_checkpoint.py` — re-evaluates each checkpoint on the validation set and compares measured against claimed. If measured matches the *Final* column above rather than *Best*, the bug is confirmed.

**Fix:**

```python
import copy
best_model_state = copy.deepcopy(model.state_dict())
# or
best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
```

---

## High

### M-03 · `BACTERIAL_DISEASES` contains four fungal diseases

**File:** `app.py:47-50`

```python
BACTERIAL_DISEASES = [
    'bacterial_leaf_blight', 'brown_spot', 'leaf_blast',
    'leaf_scald', 'narrow_brown_spot'
]
```

Only the first is bacterial. This is the list that gates Stage 2 routing (`app.py:274`), so the logic is correct — it means "any disease" — but the name asserts something false, and it has propagated: `saved_models/README.md` calls Stage 2 "specific bacterial disease classification", the training report says "HEALTHY vs BACTERIAL vs NON-LEAF", and the UI says "5 different bacterial diseases".

**Fix:** Rename to `DISEASE_CLASSES`. Correct the derived documentation.

---

### M-04 · `leaf_scald` described as bacterial in the payload sent to users

**File:** `app.py:91`

```python
'description': 'Bacterial disease causing lesions with wavy edges and yellowing.',
```

Leaf scald is caused by *Microdochium oryzae*, a fungus. This text is returned by the API and rendered in the results panel, so users receive incorrect pathogen information alongside a treatment recommendation.

---

### M-05 · Final confidence is a product of two probabilities, presented as a single confidence

**File:** `app.py:296`

```python
result['final_confidence'] = float(stage1_conf * stage2_conf)
```

Multiplying two softmax outputs is not a calibrated joint probability — the two models are not independent (they see the same image) and neither is calibrated. The product systematically understates confidence: 0.99 × 0.98 = 0.97 is displayed as if it were a meaningful 97%.

No calibration analysis was performed, and the 0.95 routing threshold (`app.py:34`) is heuristic.

**Fix:** Either report Stage 2's confidence directly when Stage 2 runs, or perform temperature scaling on a held-out set and report a calibrated number. Document the choice.

---

## Medium

### M-06 · Duplicate and unused checkpoints committed via Git LFS

**Directory:** `saved_models/` — 1.2GB

Two training runs are committed for every model (10 checkpoints where 5 are used), plus `efficientnetv2_s_*.pth` (×2, 168MB) which `app.py` never loads. `.git` is **1.8GB**.

| Model | Committed | Used |
|---|---|---|
| efficientnet_b3 | 2 | 1 |
| densenet121 | 2 | 1 |
| mobilenetv3 | 2 | 1 |
| vit_base | 2 | 1 |
| convnext_tiny | 2 | 1 |
| efficientnetv2_s | 2 | **0** |

**Fix:** Keep one checkpoint per model. Move weights to the HF Model Hub and download at build time rather than storing in Git LFS.

---

### M-07 · Augmentation includes vertical flip and 45° rotation for a directional pathology

**File:** `model.ipynb` cells 3, 15

`RandomVerticalFlip(p=0.3)` and `RandomRotation(45)` (Stage 2). Some rice lesions have orientation-dependent morphology relative to the leaf's long axis and venation — leaf blast lesions are spindle-shaped along the vein, leaf scald progresses from the tip. Aggressive rotation and vertical flipping may destroy discriminative structure.

Not proven harmful here, but untested. Worth an ablation.

---

### M-08 · `pretrained=False` in serving code relies on a deprecated API

**File:** `app.py:111, 124, 138, 155, 181`

`torchvision` deprecated `pretrained=` in 0.13 in favour of `weights=`. It still functions with a warning but is scheduled for removal. When it goes, every model constructor raises — and see B-06 for why that failure would be silent and confusing.

**Fix:** `weights=None`.

---

### M-13 · Stage 2 has no escape hatch — a Stage 1 false positive is locked in

**File:** `app.py:274-296`

```python
is_bacterial = stage1_label in BACTERIAL_DISEASES
if not is_bacterial:
    result['final_diagnosis'] = stage1_label     # healthy / not_rice_leaf exit here
    return result
if stage1_conf < self.confidence_threshold:
    stage2_class, ... = self.predict_stage2(image_tensor)   # 5 disease classes ONLY
```

Stage 2's output space is the five disease classes. It has no `healthy` and no `not_rice_leaf` option. So if Stage 1 wrongly predicts a disease on a healthy leaf, Stage 2 is *required* to pick a disease — it cannot correct the error, only relabel it.

Stage 2 is invoked precisely when Stage 1 is **uncertain** (`conf < 0.95`), i.e. exactly the cases where Stage 1 is most likely wrong. The refinement stage cannot recover from the error class it is most likely to be handed.

**Fix:** Give Stage 2 a 7-class head, or add an abstain path when Stage 2's top probability is also low.

---

### M-14 · Ensemble weights all members equally despite a 3-point accuracy spread

**File:** `app.py:233, 249` — `avg_prediction = np.mean(predictions, axis=0)`

Stage 1 members score 93.21%, 96.05% and 94.47%. A plain mean gives the weakest model the same vote as the strongest. Accuracy-weighted or logit-averaged ensembling is a cheap improvement and is worth an ablation, particularly since the project already established (by dropping EfficientNetV2-S) that member quality changes ensemble behaviour.

---

### M-15 · Layer freezing by parameter index is arbitrary and architecture-dependent

**File:** `model.ipynb` cells 5, 17

```python
for param in list(model.parameters())[:-30]:   # EfficientNet-B3
for param in list(model.parameters())[:-40]:   # DenseNet-121
for param in list(model.parameters())[:-25]:   # MobileNetV3
```

"Last 30 parameter tensors" is not a meaningful unit — it does not correspond to a layer boundary, and the same number means something completely different across architectures (a DenseNet block contributes far more parameter tensors than a MobileNet block). The three constants appear to have been chosen by trial.

**Fix:** Freeze by named module (`model.features[:6]`) or unfreeze the last *N* blocks explicitly, so the intent is legible and portable.

---

### M-16 · Focal loss `alpha=1` provides no class balancing

**File:** `model.ipynb` cell 11 — `FocalLoss(alpha=1, gamma=2)`

`gamma=2` down-weights easy examples, which helps hard boundaries. `alpha=1` is a uniform scalar and does **not** address class imbalance — the `not_rice_leaf` class has 420 training images against 350 for every other class. The README describes focal loss as the imbalance remedy; it is not, as configured.

Minor in practice (the imbalance is mild), but the stated rationale does not match the code.

---

### M-17 · Model selection criterion and LR scheduler criterion disagree

**File:** `model.ipynb` cell 9

`scheduler.step(val_loss)` reduces LR on validation **loss** plateau, while the best checkpoint is selected on validation **accuracy**. These can move in opposite directions, particularly late in training when a model becomes more confident on things it already gets right. Not wrong, but the two signals should be a deliberate choice rather than an accident.

Related: there is no early stopping. Every model trains the full 20/25 epochs regardless of plateau.

---

### M-09 · Training path hardcoded to a personal machine

**File:** `model.ipynb` cell 1

```python
DATA_PATH = r"C:\Users\anura\OneDrive\Documents\Semester 5\Deep Learning\sample\RiceLeafsDisease"
```

The notebook is not runnable by anyone else without editing. Should read from an environment variable or a config file with a documented default.

---

## Low

### M-10 · `saved_models/v1_archive/reports/training_summary_report.txt` contains unrendered f-string literals
`saved_models/v1_archive/reports/training_summary_report.txt:85, 105, 107` show `{'='*70}` verbatim — a missing `f` prefix when the report was generated.

### M-11 · Stage 2 summary lists three models but production uses two
`saved_models/v1_archive/reports/training_summary_report.txt:50-53` and `saved_models/README.md:13` still describe the 3-model Stage 2 ensemble. Stale relative to `saved_models/v1_archive/production_deployment/production_config.json`.

---

# Part B — Backend & Infrastructure

## Critical

### B-01 · Internal exception text returned to the client

**File:** `app.py:440`

```python
except Exception as e:
    logger.error(f"Prediction error: {e}")
    return jsonify({'error': str(e)}), 500
```

`str(e)` on a PIL or torch exception routinely contains absolute filesystem paths, library internals, and tensor shapes. On a public endpoint this leaks server layout to anyone who uploads a malformed file.

**Fix:** Log the detail, return a generic message and a correlation ID.

```python
except Exception:
    err_id = uuid.uuid4().hex[:8]
    logger.exception(f"Prediction error [{err_id}]")
    return jsonify({'error': 'Analysis failed', 'ref': err_id}), 500
```

---

## High

### B-02 · No upload size limit — unbounded memory allocation on a public endpoint

**File:** `app.py` — no `MAX_CONTENT_LENGTH` anywhere.

`file.read()` (`app.py:397`) loads the entire request body into memory, then PIL decodes it. A single large upload, or a decompression-bomb PNG, exhausts the container's memory. The 10MB check exists only in JavaScript (`static/script.js:278`) and is bypassed by posting directly.

**Fix:**

```python
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000   # PIL decompression-bomb guard
```

---

### B-03 · No rate limiting on an expensive endpoint

**File:** `app.py:385`

`/predict` runs 3–5 CNN/ViT forward passes per call on free-tier CPU. There is no throttling, no authentication, and CORS is fully open (B-04). A trivial loop saturates the Space indefinitely.

**Fix:** `flask-limiter` with a per-IP cap (e.g. 10/minute). Already sketched in `SETUP_GUIDE.md:365` but never implemented.

---

### B-04 · CORS open to all origins

**File:** `app.py:26` — `CORS(app)` with no arguments allows every origin.

Any site can embed and drive this API. Combined with B-03, any page can use the Space as free compute.

**Fix:** `CORS(app, origins=[...])` with an explicit allowlist, or drop CORS entirely — the frontend is same-origin and does not need it.

---

### B-05 · `/health` returns HTTP 200 even when no model is loaded

**File:** `app.py:442-450`

```python
return jsonify({
    'status': 'healthy',
    'model_loaded': predictor is not None,
    ...
})
```

Always 200. An orchestrator, uptime monitor, or load balancer polling this endpoint sees a healthy service while every `/predict` call returns 500. The UptimeRobot monitor currently pointed at `/health` would report 100% uptime through a total model-load failure.

**Fix:** Return 503 when `predictor is None`. Separate liveness from readiness.

---

### B-06 · Bare `except:` silently substitutes a different architecture

**File:** `app.py:154-176` and `:180-202`

```python
def create_vit_base(num_classes):
    try:
        model = models.vit_b_16(pretrained=False)
        ...
    except:                                    # bare except
        model = models.efficientnet_b4(pretrained=False)   # different architecture
```

If constructing ViT fails for any reason — including a `KeyboardInterrupt` or `SystemExit`, which a bare `except` also swallows — the code silently builds an **EfficientNet-B4** instead. Loading ViT weights into it then fails with a confusing state-dict mismatch far from the actual cause. Same pattern for ConvNeXt-Tiny → ResNeXt-50.

**Fix:** Delete the fallbacks. If the architecture cannot be built, that is a fatal configuration error and should say so.

---

## Medium

### B-07 · No `.dockerignore` — 3.8GB build context

**Measured:**

| Path | Size |
|---|---|
| `.git` | 1.8G |
| `saved_models` | 1.2G |
| `train` | 500M |
| `train_stage2` | 204M |
| `validation` | 129M |
| `validation_stage2` | 51M |
| **Total context** | **3.8G** |

Every build ships all of this to the Docker daemon. The `Dockerfile` only `COPY`s four paths, so most of it is transferred and discarded — but `COPY saved_models/ saved_models/` does bake in all 1.2GB including the five unused checkpoints.

**Fix:** Add `.dockerignore` (train/, validation/, .git/, *.ipynb, v2/) and copy only the five checkpoints actually loaded. Realistic image size drops from ~2GB to well under 1GB.

---

### B-08 · Dockerfile `HEALTHCHECK` can never pass

**File:** `Dockerfile:36-37`

```dockerfile
HEALTHCHECK ... CMD python -c "import requests; requests.get('http://localhost:7860/health')"
```

`requests` is **not in `requirements.txt`**. The healthcheck raises `ModuleNotFoundError` on every invocation, so the container is permanently marked unhealthy.

**Fix:** Use `urllib.request` from the stdlib, or add `requests` to requirements.

---

### B-09 · Container runs as root

**File:** `Dockerfile` — no `USER` directive.

**Fix:** Create and switch to a non-root user before `CMD`.

---

### B-10 · Models loaded at import time with no recovery

**File:** `app.py:350-358`

Loading happens at module import. On failure `predictor = None` and the process stays up serving 500s forever, with no retry and no way to reload without a restart. Combined with B-05, this failure is invisible to monitoring.

---

### B-11 · Single Gunicorn worker, 120s timeout, no concurrency strategy

**File:** `Dockerfile:40` — `--workers 1 --timeout 120`

One worker means one request at a time; a second user waits behind the first for the full inference duration. Raising worker count multiplies memory by the full model set (~1.2GB of weights per worker), so this needs thought rather than a number change — likely a single worker with threads, or batching.

---

### B-12 · Relative model path breaks when run from another directory

**File:** `app.py:325` — `model_dir = 'saved_models'` *(fixed in current working tree; noted because it shipped)*

Resolved relative to the process working directory, not the file. `cd /tmp && python /path/to/app.py` fails.

---

## Low

### B-13 · `numpy` unpinned
`requirements.txt` pins torch and torchvision but numpy arrives transitively. A numpy 2.x resolution difference between the build that worked and a later rebuild is a plausible source of drift.

### B-14 · Unused imports
`app.py:14` `base64`, `app.py:11` `models` imported twice via two paths in the notebook.

### B-15 · No structured logging or request IDs
`logging.basicConfig(level=INFO)` only. No correlation between a user-visible error and a log line.

### B-16 · No `/metrics` endpoint
Sketched in `SETUP_GUIDE.md:399` but never implemented. No visibility into prediction volume or class distribution in production.

---

## Prioritised fix order

| # | Finding | Effort | Why first |
|---|---|---|---|
| 1 | **M-01** retrain with hard negatives | days | Nothing else matters if the model reads backgrounds |
| 2 | **M-12** deep-copy the best checkpoint | 1 line | One-line fix; without it every retrain saves the wrong weights |
| 3 | B-01, B-02, B-03, B-04 | 2–3 h | Public endpoint, currently unprotected |
| 4 | B-05 | 15 min | Monitoring is currently lying |
| 5 | M-03, M-04 | 20 min | Factually wrong information sent to users |
| 6 | M-02 | 1 h | Makes every future metric trustworthy |
| 7 | B-07, B-08, B-09 | 1–2 h | Build and deploy hygiene |
| 8 | M-13 | 2 h | Stage 2 cannot correct Stage 1's likeliest errors |
| 9 | B-06, B-10, B-11 | 3–4 h | Failure modes and concurrency |
| 10 | M-14, M-15, M-16, M-17 | — | Training-quality improvements, fold into v2 |
| 11 | Remainder | — | Hygiene |

**M-12 is the one to fix before anything else in v2.** It is a single line, and until it is fixed every training run you do — including all of v2 — silently saves the wrong weights.

---

## What this section demonstrates

The most useful finding here (M-01) came from refusing to accept a passing metric. 96.68% validation accuracy is a number that invites you to stop looking. The reason it was wrong is not visible in any confusion matrix, any loss curve, or any per-class score — every one of those artefacts looked healthy. It only appeared by asking *why* a specific input failed, forming a hypothesis, testing it, **discarding the first hypothesis when the evidence contradicted it**, and constructing a controlled experiment that held leaf pixels constant while varying background.

The secondary findings show a pattern worth naming: several bugs here are **silent-failure paths**. `os.listdir()[0]` picking an arbitrary checkpoint, `if matching_files:` skipping a model that then votes with random weights, a bare `except` substituting a different architecture, `/health` reporting 200 while nothing works, a `HEALTHCHECK` that cannot run. None of these produce an error message. Each fails quietly in a way that looks like success — which is the failure mode that survives longest in production and is the hardest to notice from the outside.

Designing systems that fail loudly is a different skill from making them work, and it is the one this codebase was missing.
