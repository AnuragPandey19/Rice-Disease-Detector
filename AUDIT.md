# Project Audit — Rice Leaf Disease Detection

**Date:** 9 August 2026 · **Revised** 10 August 2026 after the Stage 2 rebuild
**Audited version:** Stage 1 v2 (retrained 2026-08-09) · Stage 2 v2 (retrained 2026-08-10) · `app.py` 2.0
**Method:** every factual claim below was verified programmatically against the working tree before being written. Numbers come from `v2/reports/*.json` and from direct inspection of the files, not from memory.

Supersedes `docs/v1_audits/AUDIT_BACKEND.md` and `docs/v1_audits/AUDIT_FRONTEND.md`, which describe the pre-rebuild state and are retained as history.

---

## 1. Verified current state

| Component | Value | How verified |
|---|---|---|
| `app.py` | 727 lines | `wc -l` |
| `templates/index.html` | 428 lines | `wc -l` |
| `static/script.js` | 563 lines | `wc -l` |
| `static/style.css` | 1,204 lines | `wc -l` |
| Live Stage 1 weights | 3 files, 87.4 MB, timestamp `20260809_202820` | `ls saved_models/stage1_models` |
| Live Stage 2 weights | 2 files, 437.6 MB, timestamp `20260810_050916` | `ls saved_models/stage2_models` |
| Archived weights | 1.2 GB in `saved_models/v1_archive/` (both stages) | `du -sh` |
| Training data on disk | 884 MB across 4 folders | `du -sh` |

**Measured performance**, from `v2/reports/stage1_v2_20260809_202820.json`:

| Metric | Value |
|---|---|
| Stage 1 test accuracy (656 held-out images) | **94.51%** |
| — studio images (402) | 96.52% |
| — real field photographs (120) | **81.67%** |
| — PlantVillage hard negatives (54) | 100.00% |
| — stock-photo negatives (80) | 100.00% |
| `not_rice_leaf` precision / recall | **100% / 100%** |

Individual Stage 1 models, best validation accuracy (2 dp; raw values 93.28125 / 92.96875 / 92.34375): EfficientNet-B3 93.28% (epoch 17), DenseNet-121 92.97% (epoch 19), MobileNetV3 92.34% (epoch 20).

**Stage 2**, from `v2/reports/stage2_v2_20260810_050916.json`:

| Metric | Value |
|---|---|
| Stage 2 test accuracy (425 held-out images) | **95.29%** (405/425) |
| — studio images (335) | 97.61% |
| — real field photographs (90) | **86.67%** |
| Weakest class | `leaf_blast` — 90.91% precision, 92.78% recall |

Individual Stage 2 models, best validation accuracy: ViT-B/16 96.87% (epoch 24), ConvNeXt-Tiny 95.42% (epoch 20).

---

## 2. Open findings

**Progress as of 10 August 2026:** A-03, A-04, A-06, A-07 and A-09 are closed. A-12 was found while validating the Stage 2 rebuild and its user-facing half shipped the same day. A-01, A-02, A-05, A-11 remain open; A-08 needs a manual delete.

The web application was rebuilt alongside these fixes: light and dark themes on semantic tokens, structured management guidance per disease replacing a one-line string, a browsable disease library generated from the same source as the API response, and the A-12 warnings. Two defects surfaced during that work and are recorded under A-04 and A-12.

A-06 produced an unexpected result — the model is under-confident rather than over-confident — which led to lowering the Stage 2 routing threshold from 0.95 to 0.85.

Closing A-03 did not only add a number. It confirmed A-02 in a second, independent place, and it exposed A-12. A rebuild that had gone perfectly would have been the more suspicious outcome.

Three further stale descriptions were found in the README while fixing the above and corrected: it still described the multiplicative confidence formula that `app.py` had already replaced, recommended copper bactericides for a fungal disease, and omitted the abstain path from the flow diagram.

Severity: **High** = affects correctness or users · **Medium** = misleading or risky · **Low** = hygiene

### High

#### A-01 · Field accuracy is 81.67%, and the headline number hides it

`v2/reports/stage1_v2_20260809_202820.json` → `test_by_source`.

94.51% is an average over a test set that is 61% studio images. On the imagery the system is actually for, roughly one prediction in five is wrong. The README and UI now report both numbers, so this is disclosed rather than hidden — but it remains the single most important fact about the model.

Sub-finding, from `v2/scripts/07_confusion.py` output: on field photographs, `leaf_blast → healthy` occurred 5 times. **A missed disease costs a farmer more than a false alarm**, and this is the most common serious error in the confusion matrix.

**Fix:** more field photographs, particularly for leaf blast. Paddy Doctor has 1,738 blast images; only 200 were used.

---

#### A-02 · Leaf scald and narrow brown spot have no field training data

Verified: `v2/data/raw/field_photos/` contains four class folders (`bacterial_leaf_blight`, `brown_spot`, `healthy`, `leaf_blast`), 200 images each. The other two rice classes have none.

Both score **100% precision and 100% recall** on the Stage 1 test set. That is not a good result — it is a symptom. Those classes exist only in studio form, so the model can identify them from photographic style rather than leaf morphology. The original background shortcut, surviving in miniature.

**Confirmed independently by the Stage 2 rebuild (10 Aug 2026).** Stage 2 was trained from scratch on the same splits, and the same two classes came out on top: `narrow_brown_spot` 100.00% / 100.00%, `leaf_scald` 97.10% / 100.00%, against `leaf_blast` at 90.91% / 92.78%.

The support column makes the mechanism explicit rather than merely suspected:

| Class | Test support | Field images |
|---|---|---|
| bacterial_leaf_blight, brown_spot, leaf_blast | 97 each | 30 each |
| leaf_scald, narrow_brown_spot | **67 each** | **0** |

The 30-image difference *is* the field photographs. Two models with different architectures, trained independently, both score near-perfectly on exactly the two classes that were never photographed in a field. That is no longer an inference from one result — it is a reproduced one.

**Fix:** source field photographs for both, or drop them and ship a 3-disease model. Neither is in a mainstream public field dataset, so this is genuinely hard.

---

#### A-03 · Stage 2 was never rebuilt and has no held-out test set

Retraining Stage 1 only was the correct call: the `not_rice_leaf` class lives in Stage 1, so that is where the shortcut was. But Stage 2's 98.18% carried **exactly the caveat Stage 1's old 96.68% did** — it was validation accuracy on data that was also used to choose the model (it is the number that justified dropping EfficientNetV2-S).

Stage 2 also trained on studio-only images, so when Stage 1 routed a field photo to it, Stage 2 was operating out of distribution and nobody had measured the cost.

**Fix:** **DONE (10 Aug 2026).** `v2/scripts/09_train_stage2.py` reused the v2 splits with `healthy` and `not_rice_leaf` filtered out — 1,950 train / 415 validation / 425 test, 600 field photographs — so the split assignment matches Stage 1 exactly and nothing leaks across the boundary.

| | Before | After |
|---|---|---|
| Headline | 98.18% validation, selection-contaminated | **95.29% test, measured once** |
| Field photographs | never measured | **86.67%** |
| Studio images | not reported separately | 97.61% |

Two results worth stating plainly:

**The number went down, and that is the point.** 95.29% and 98.18% do not measure the same thing. The old figure was computed on data used to pick the models; the new one on 425 images touched once, including field photographs the old Stage 2 had never seen.

**Stage 2 beats Stage 1 on field photographs — 86.67% against 81.67%.** This relocates the remaining weakness. The bottleneck is Stage 1's seven-way decision, not the disease refinement downstream of it. A-01's fix (more field photographs, especially leaf blast) is therefore the correct next investment, not further Stage 2 work.

**Verified through the real pipeline before promotion.** `diagnostics/compare_stage2.py` ran the three post-mortem photographs through `app.py` twice — once with the shipping weights, once with the candidates, Stage 1 held constant. All three reached Stage 2, so all three were informative. One verdict changed: `t2.jpg` moved from `Uncertain` (39.94%, below the abstain line) to leaf blast at 78.28%, agreeing with Stage 1, with 23.48% of spurious `narrow_brown_spot` mass collapsing to 0.16%. `t1` was unchanged (+0.60 pts). `t3` is discussed in A-12.

Promoted 10 Aug 2026; v1 Stage 2 weights moved to `saved_models/v1_archive/stage2_models/` and the `Dockerfile` COPY lines repointed.

Checkpoints carry `eval_set: v2_stage2_validation_20260810`, so the provenance problem A-07 describes cannot recur here.

**Not fixed by this:** the abstain threshold is still 0.45 by judgement. The rebuild *unblocks* measuring it — see A-06 — but the measurement has not been run.

---

### Medium

#### A-04 · The report `.docx` still contains superseded numbers

Verified by extracting `word/document.xml`: contains `96.68` and `98.18`, does **not** contain `94.51` or `81.67`.

`Rice_Leaf_Disease_Detection_Report.docx` describes the pre-rebuild model throughout, including the "at a glance" box. If shared as-is it presents figures the project has since disproved.

**Fix:** **DONE (9 Aug 2026), regenerated again 10 Aug 2026.** Rewritten around the find-fix-measure narrative. Verified by extracting `word/document.xml`: contains `94.51`, `81.67`, `96.52`, `0 of 656`, `95.29`, `86.67`, `97.61`; `96.68` and `98.18` appear only as historical context.

The second pass caught staleness the first had introduced by being written mid-flight: the document still described the routing threshold as 0.95 after A-06 lowered it to 0.85, still said calibration had never been measured after it had, and still said Stage 2 was never rebuilt. **A generated document goes stale the moment the thing it describes changes.** It is regenerated from `build_report_v2.js`, so the fix is to re-run the generator on every substantive change rather than to edit the `.docx` by hand.

A PDF render is now kept alongside it so the report can be read without Word.

---

#### A-05 · A hostile background probe still flips the prediction

`v2/reports/shortcut_check.json`: 8 trials, 8 accepted as rice on a plain background, **7 flipped to `not_rice_leaf`** when the identical leaf strip was composited onto a stock photo.

Interpretation matters here. The training data now contains two different kinds of "busy": paddy-field clutter (labelled rice) and stock-photo scenes of dogs, cars and houses (labelled not-rice). The probe uses the second kind, which the model has seen 524 times as a negative. Rejecting it is arguably correct rather than a bug, and the composite is unlike any real photograph.

Recorded as a **known behavioural boundary**, not as a defect: paddy-field backgrounds are handled, arbitrary scenes are not.

---

#### A-06 · Calibration was never measured, and two thresholds depend on it

`app.py:46` `CONFIDENCE_THRESHOLD = 0.95` (Stage 2 routing) and `app.py:74` `STAGE2_ABSTAIN_THRESHOLD = 0.45` (abstain).

Both were chosen by judgement. No reliability diagram or ECE was computed, so there is no evidence either sits at a sensible point — and the abstain threshold in particular controls how often the system says "I don't know" to a farmer.

The three real photographs in `diagnostics/images/` were classified at 70%, 38% and 41% confidence. Whether 38% should have triggered an abstain is currently unanswerable.

**RESULT (9 Aug 2026), `v2/reports/calibration.json`:** the ensemble is systematically **under**-confident, not over-confident — the opposite of the usual deep-network failure. Every reliability bin has a negative gap, up to 30 points:

| Says | Actually correct | Gap |
|---|---|---|
| 65.0% | 94.4% | −29.5 |
| 75.2% | 100.0% | −24.8 |
| 85.2% | 100.0% | −14.8 |
| 96.9% | 100.0% | −3.1 |

**ECE 11.98%.** Averaging three softmax outputs pulls the maximum down whenever members disagree slightly, while the argmax stays correct. The confidence figure shown in the UI is therefore not a probability and understates the model.

**Two consequences, handled differently:**

**Routing threshold — CHANGED to 0.85.** An under-confident model rarely cleared 0.95, so Stage 2 ran on more than half of all requests, defeating the purpose of a fast first pass. In the 0.80–0.90 band Stage 1 was already 100% accurate across 120 validation images, so routing those to Stage 2 could only degrade them. `CONFIDENCE_THRESHOLD` is now 0.85.

**Abstain threshold — DELIBERATELY UNCHANGED at 0.45.** The script sweeps **Stage 1** ensemble confidence, but this threshold governs **Stage 2** confidence — a different distribution over a different label space (5 classes, not 7). Tuning one from the other would be unsound, so the sweep table is recorded as evidence about Stage 1 and the threshold is left as it was. This is a limitation of the script, now documented in it.

**Still open, but now unblocked.** A-03 has produced a Stage 2 held-out test set and a Stage 2 validation split, so the same analysis can finally be run against Stage 2 confidence. `08_calibration.py` would need its model directory, class count and ensemble membership parameterised — it currently hardcodes the three Stage 1 models.

That 0.45 is not theoretical. Across the three post-mortem photographs it decided two outcomes in opposite directions: `t2.jpg` at 39.94% was suppressed to `Uncertain` under the old weights when both stages in fact agreed on leaf blast, and `t3.jpg` at 49.16% is published as a confident answer while the two stages disagree (A-12). One threshold, two wrong-looking calls, still unmeasured.

Temperature scaling remains unapplied.

---

#### A-07 · The accuracy-weighted ensemble uses v1-comparable numbers unequally

`app.py` weights ensemble members by the `best_val_acc` stored in each checkpoint. Verified present in both stages.

At the time of the original audit, Stage 1 checkpoints carried v2 numbers (93.28 / 92.97 / 92.34, rebuilt validation set) while Stage 2 carried v1 numbers (98.41 / 98.64, old validation set). The two stages weight independently, so this never mixed scales inside a single vote, but the values were not comparable across the system and a future change could easily have assumed they were.

**Fix:** **DONE.** `v2/scripts/03_train_stage1.py` and `09_train_stage2.py` write `eval_set` and `eval_set_size` into every checkpoint, and `app.py` logs the provenance on load (`val acc 93.28% on v2_validation_2026-08-09`).

**Improved further by A-03.** Both Stage 2 checkpoints now carry accuracies from the same v2 validation split (96.87 / 95.42, `eval_set: v2_stage2_validation_20260810`). Weighting Stage 2 members against each other is therefore meaningful for the first time — previously the two numbers came from different v1 splits and the weighting was arithmetic on incomparable quantities.

---

#### A-12 · The two stages can disagree, and the response does not say so

*New, 10 August 2026. Found by `diagnostics/compare_stage2.py`; raw output in `diagnostics/stage2_ab_results.json`.*

On `t3.jpg`, three things are true at once:

| | |
|---|---|
| Stage 1 says | `leaf_blast` @ 40.95% |
| Stage 2 says | `leaf_scald` @ 49.16% |
| Stage 2 runner-up | `leaf_blast` @ 37.44% |

The two stages pick **different diseases**. The published answer clears the 0.45 abstain threshold by four points. The runner-up — which happens to be Stage 1's choice — is 11.7 points behind, not a decisive margin. And the API returns `Leaf Scald` as a plain label with a severity and a treatment recommendation attached.

Not a regression: the shipping model behaved the same way (50.58%), and the rebuild moved it 1.42 points closer to the abstain line rather than resolving it. It is recorded now because the A/B made it visible, not because the rebuild caused it.

Two failure modes are being conflated. `abstained` fires only on low absolute Stage 2 confidence. It says nothing about **stage disagreement** or about a **narrow top-2 margin**, and both are present here. A prediction can be well above the threshold and still be one the system has no business stating flatly.

**Fix:** **DONE (10 Aug 2026).** `/predict` now returns `stages_agree` and `runner_up` (label, confidence, margin in points). The UI shows one of two strips: an amber *"Not a confident result"* when the stages name different diseases, and a quieter *"Close call"* when they agree but the runner-up is within 15 points. `stages_agree` is `null` when Stage 2 did not run, which is deliberately distinct from `true`.

The wording is written for the reader, not the author. A first pass said *"Stage 1 read this as Leaf Blast, Stage 2 as Leaf Scald, with Leaf Blast 11.72 points behind"* — accurate, and useless to a grower. It now says *"It could also be Leaf Blast. Compare the symptoms of both, and check with an agronomist before treating."* A regression test asserts that no stage or model vocabulary reappears in that string.

**Still open:** this flags a contested diagnosis, it does not resolve one. And the frequency is still unquantified — counting stage disagreements across the 425-image Stage 2 test set has not been done.

**Caveat on scope.** This rests on one photograph of unknown ground truth. It shows the response format cannot express the uncertainty it holds — it does not establish how often this happens. Quantifying that means counting stage disagreements across the 425-image Stage 2 test set, which has not been done.

---

### Low

#### A-08 · `_DELETE_ME/` needs deleting by hand

Contains `__pycache__/` and `bg_test/` (1.1 MB of scratch composites). The sandbox could not remove them — "Operation not permitted". Gitignored, so harmless, but should be deleted from Explorer.

#### A-09 · Test images are copyrighted stock photography

`diagnostics/images/t2.jpg` carries a visible **Science Photo Library** watermark; `t1.webp` and `t3.jpg` are of unclear provenance. All three were downloaded from the web on 6 August 2026.

They are the evidence behind the post-mortem and are referenced throughout the documentation, but they are **gitignored** — verified with `git check-ignore` — so they are not redistributed. That is the correct call, and the rationale is now recorded in `.gitignore` so nobody removes the rule thinking it is about file size.

**Consequence for reproducibility:** anyone cloning the repository cannot run `v2/scripts/06_test_real_photos.py` as-is. **DONE** — the README now carries a "Bring your own photographs" note explaining that any field images dropped into `diagnostics/images/` are picked up automatically.

---

#### A-10 · `git-push.bat` runs `git add .`

Safe as of this audit: `.gitignore` covers `v2/`, `PlantVillage-Dataset/`, `_DELETE_ME/`, `*.zip` and the test images — verified with `git check-ignore`. Worth remembering that this script commits whatever is not ignored, so new large artefacts need an ignore rule *before* the next push.

#### A-11 · Augmentation includes vertical flip and 30° rotation

Carried over from v1 unchanged, deliberately, so that v2 differed from v1 only in the data. Some rice lesions are orientation-dependent relative to the leaf axis — leaf blast lesions run along the vein, leaf scald starts at the tip. Never ablated. Unknown whether it helps or hurts.

---

## 3. Closed since the previous audit

Verified as no longer present in the working tree.

**The original defect.** The model classified on background rather than leaf morphology. Fixed by rebuilding the dataset: 350 hard negatives (14 species, PlantVillage) filled "plain background, not rice"; 800 field photographs (Paddy Doctor) filled "busy background, is rice". Confirmed by `v2/scripts/06_test_real_photos.py` — three photographs the original rejected at 88–99% confidence are now classified as rice, and the test set shows zero false rejections in 656 images.

**Checkpoint integrity.** `model.ipynb` used `state_dict().copy()` — a shallow copy that tracked live weights, so every checkpoint held final-epoch weights labelled with best-epoch accuracy. Now `copy.deepcopy`. Verified absent from executable code.

**Non-deterministic model loading.** `os.listdir(...)[0]` picked an arbitrary checkpoint from a directory containing two training runs per model, differently on different machines, and silently left a model at random initialisation if no file matched. Now sorts by timestamp and raises.

**Backend security.** Exception text leaked to clients, no upload limit, no rate limiting, CORS open to all origins, `/health` returning 200 while the model was unloaded — all verified fixed in `app.py`.

**False claims in the UI.** "98.2% Accuracy" appeared three times and matched no measured figure; "5 different bacterial diseases" when four are fungal; `leaf_scald` described as bacterial in the API payload. All corrected, and the hero now shows 94.5% with the studio/field split spelled out.

**Accessibility.** 273 lines of HTML with zero `aria-`, `role` or `tabindex` attributes. Now has a skip link, keyboard-activatable upload zone, focus trap, Escape handling and `aria-live` results.

Full detail in `docs/v1_audits/`.

---

## 4. Recommended order

Revised 10 August 2026. A-03 is done, and its result reorders what follows: Stage 2 now outperforms Stage 1 on field photographs, so effort belongs in Stage 1 and in the data.

| # | Finding | Effort | Why |
|---|---|---|---|
| 1 | A-06 measure Stage 2 calibration, then set the abstain threshold | ~1 h | Unblocked by A-03, and now the last unmeasured number in the system |
| 2 | A-01 more field photos, esp. leaf blast | 1–2 h | Directly targets the worst real-world error, and A-03 confirmed Stage 1 is the bottleneck |
| 3 | A-02 leaf scald / narrow brown spot | hard | Blocked on data availability; now confirmed in both stages |
| 4 | A-12 quantify how often the stages disagree | ~30 min | The warning ships; the frequency behind it is still a guess |
| 5 | A-08 delete `_DELETE_ME/` | 1 min | — |
| 6 | A-11 augmentation ablation | 1 h | Might be free accuracy, might be nothing |

A-03, A-04, A-07, A-09 and the user-facing half of A-12 are closed. A-05 is recorded as a known behavioural boundary rather than a defect.

---

## 5. Note on this audit

The previous two audits contained errors that only surfaced when their claims were tested: an aspect-ratio hypothesis asserted confidently and later disproved, field photographs described as "optional" when they were essential, and a shortcut-check script that selected its images by sort order and silently tested the wrong thing.

Every claim in this document was checked against the working tree before being written — file existence, line counts, checkpoint contents, JSON report values, and reference integrity across all documentation. Where a number is quoted, its source is named.

Two things are stated as unknown rather than guessed: whether the augmentation settings help (A-11), and whether the confidence thresholds are well placed (A-06). Both would need experiments that have not been run.

**Revision note, 10 August 2026.** The Stage 2 rebuild was verified through `app.py` itself before promotion, not through a standalone evaluation script, so the numbers describe the deployed system rather than a model in isolation. Three claims in this document changed as a result: A-03 closed, A-02 gained independent confirmation from a second architecture, and A-12 was added.

A-12 is deliberately scoped to what one photograph can support. It shows the response format cannot express an uncertainty the system is holding; it does not show how often that matters. The honest version of that finding is smaller than the tempting one.
