# Frontend Audit — Rice Leaf Disease Detection

**Scope:** `templates/index.html` (273 lines), `static/script.js` (456 lines), `static/style.css` (843 lines)
**Date:** August 2026
**Auditor:** self-audit prior to v2

Severity key: **Critical** = ships wrong information or breaks core flow · **High** = real user-facing defect · **Medium** = quality/maintainability · **Low** = polish

---

## Status — all findings resolved

| Severity | Count | Fixed |
|---|---|---|
| Critical | 3 | 3 |
| High | 5 | 5 |
| Medium | 7 | 7 |
| Low | 4 | 4 |

| ID | Finding | Status |
|---|---|---|
| F-01 | "98.2% Accuracy" ×3 | Fixed — now 96.7% with "on curated dataset" caveat + limitation link |
| F-02 | "5 bacterial diseases" | Fixed — "one bacterial, four fungal" |
| F-03 | No photo guidance | Fixed — guidance panel with do/avoid, plus per-response `input_guidance` |
| F-04 | `result.details` unguarded | Fixed — `??` fallbacks throughout `displayResults()` |
| F-05 | Zero a11y | Fixed — skip link, roles, `tabindex`, keyboard activation, focus trap, Escape, `aria-live`, focus-visible rings |
| F-06 | No crop tool | Fixed — canvas cropper, no dependency; auto-offered on rejection |
| F-07 | One media query | Fixed — tablet, phone, landscape, small-phone, reduced-motion, print |
| F-08 | Toasts stack | Fixed — container, dedupe, max 3, dismiss button |
| F-09 | Google Fonts CDN | Fixed — removed, system font stack |
| F-10 | Client-only size limit | Fixed server-side — see B-02 |
| F-11 | `console.log` ×3 | Fixed — removed |
| F-12 | 1920×1080 capture | Fixed — square guide, centre-square crop |
| F-13 | No button loading state | Fixed — `is-loading`, disabled, `isAnalyzing` guard |
| F-14 | Hardcoded colours in JS | Fixed — CSS classes/vars |
| F-15 | No favicon | Fixed — inline SVG data URI |
| F-16 | No meta/OG | Fixed |
| F-17 | No scroll-spy | Fixed — IntersectionObserver |
| F-18 | Weak alt text | Fixed — includes filename and state |
| F-19 | Hardcoded year | Fixed — set from `Date` |

Original findings follow.

The UI is visually competent — the layout, camera integration, and drag-drop flow are all above what a typical student project ships. The problems are not aesthetic. They are: **the page states accuracy figures that are wrong and misleading**, **it gives users no guidance on the one thing that determines whether the model works**, and **it is effectively unusable with a keyboard or screen reader**.

---

## Critical

### F-01 · Hero displays "98.2% Accuracy" — a number that appears nowhere in the results

**Files:** `templates/index.html:44`, `:235`, `:241`

```html
<div class="stat-value">98.2%</div>          <!-- line 44 -->
...and 5 different bacterial diseases with 98.2% accuracy.   <!-- line 235 -->
<p>98.2% detection accuracy</p>              <!-- line 241 -->
```

The actual measured figures are **96.68%** (Stage 1, 7-class) and **98.18%** (Stage 2, 5-class). There is no metric equal to 98.2%. It appears to be Stage 2 rounded up and then presented as whole-system accuracy — which it is not, because every prediction passes through Stage 1 first.

Worse, since the background-shortcut finding (see `AUDIT_BACKEND.md` M-01), *neither* number describes field performance. The page presents a studio-condition benchmark as a general capability claim, three times, in the largest text on the page.

**Fix:** Replace with the honest framing:

```html
<div class="stat-value">96.7%</div>
<div class="stat-label">Accuracy on curated dataset</div>
```

and add a one-line qualifier near the upload box linking to the README limitation section.

---

### F-02 · "5 different bacterial diseases" — 4 of the 5 are fungal

**File:** `templates/index.html:235`

Bacterial Leaf Blight is bacterial. Brown Spot, Leaf Blast, Leaf Scald and Narrow Brown Spot are all **fungal**. The copy is factually wrong about the domain the product operates in, and an agronomist would notice immediately.

This mirrors the same error in the backend (`app.py` `DISEASE_INFO['leaf_scald']` describes it as "A bacterial disease") and in the variable name `BACTERIAL_DISEASES`, which actually holds all five disease classes.

**Fix:** "5 common rice leaf diseases (1 bacterial, 4 fungal)".

---

### F-03 · No guidance on how to photograph the leaf — the single biggest determinant of success

**File:** `templates/index.html:73-74`

```html
<p class="upload-text">Drop your image here or <span class="upload-link">browse</span></p>
<p class="upload-hint">Supports: JPG, PNG, JPEG (Max 10MB)</p>
```

The hint text covers file formats. It says nothing about what the model actually requires. Every training image is a single isolated leaf on plain light paper; the model rejects cluttered field photographs as `not_rice_leaf` with ~99% confidence. A user photographing a plant in a field — the obvious thing to do — gets a confident wrong answer with no indication why.

The "Use Camera" button (line 77) actively encourages the failure mode. A phone camera pointed at a rice plant produces exactly the input the model cannot handle.

**Fix:** Add a short illustrated requirement above the drop zone:

> **For accurate results:** place a single leaf flat on plain paper (white works best), fill the frame with the leaf, avoid shadows and other foliage in shot.

Ideally with a good/bad example image pair. This is a two-hour change that would meaningfully improve real-world accuracy without touching the model.

---

## High

### F-04 · `displayResults()` will throw if the API omits `details`

**File:** `static/script.js:381-384`

```js
stage1Pred.textContent = result.details.stage1_prediction;
stage1Conf.textContent = result.details.stage1_confidence;
modelsUsed.textContent = result.details.models_used;
```

No guard on `result.details`. Any backend change or error path that returns a response without `details` produces an uncaught `TypeError`, the loading overlay is already hidden, and the user sees a frozen page with no error. The `try/catch` in `analyzeImage()` does not cover `displayResults()` — it is called after the `await`, inside the `try`, so it *is* covered, but the catch then shows a misleading "An error occurred during analysis" instead of the real problem.

**Fix:** Optional chaining with fallbacks.

```js
const d = result.details || {};
stage1Pred.textContent = d.stage1_prediction ?? '—';
stage1Conf.textContent = d.stage1_confidence ?? '—';
modelsUsed.textContent = d.models_used ?? '—';
```

---

### F-05 · Zero accessibility affordances

**File:** `templates/index.html` — grep for `aria-`, `role=`, `tabindex` returns **0 matches** across 273 lines.

Concrete consequences:

- The upload area is a `<div>` with a click handler (`script.js:49`). It is not focusable and not reachable by keyboard. A keyboard-only user cannot upload a file except by tabbing to the hidden `<input type="file">` — which is `hidden`, so it is removed from the tab order entirely.
- The camera modal (`index.html:101`) does not trap focus, has no `role="dialog"`, no `aria-modal`, and does not close on `Escape`.
- Result content updates dynamically with no `aria-live` region, so a screen reader announces nothing when the diagnosis appears.
- Icon-only buttons (`removeImage` `✕`, `closeResults` `✕`, `captureBtn`) have no accessible name. `switchCameraBtn` and `cancelCameraBtn` have `title` attributes, which is weak but better than nothing.

**Fix:** `role="button"` + `tabindex="0"` + keydown handler on the drop zone; `aria-label` on every icon button; `role="dialog" aria-modal="true"` plus focus trap and Escape handling on the camera modal; `aria-live="polite"` on the results container.

---

### F-06 · No crop or zoom control before submission

**File:** `static/script.js:268-297` (`handleFileSelect`) — the file goes straight from picker to preview to `/predict` untouched.

This matters more than it normally would. The `diagnostics/test_background.py` experiment showed that cropping a failing field photo down to the leaf changes the prediction from `not_rice_leaf` (99.08%) to `brown_spot` — the model is capable of the right answer, it just needs the background removed. A client-side crop widget would convert a large class of failures into successes with no model change.

**Fix:** Add a crop step between preview and submit. Cropper.js is ~30KB and would cover this; a minimal canvas-based rectangle selector is maybe 80 lines if avoiding the dependency.

---

### F-07 · Single 35-line media query for an 843-line stylesheet

**File:** `static/style.css:810` — the only `@media` block in the file, covering `max-width: 768px`.

843 lines of desktop-first CSS with one mobile breakpoint. There is no tablet range (768–1024px), no handling of landscape phones, and no `prefers-reduced-motion` despite `background-animation` (`index.html:12`) and multiple keyframe animations running continuously.

Given the stated audience is farmers and field officers — i.e. **mobile-first users** — this is backwards. The layout should be built for a phone and enhanced for desktop, not the reverse.

**Fix:** At minimum add a 768–1024px range and a `@media (prefers-reduced-motion: reduce)` block disabling the background animation. Longer term, restructure mobile-first.

---

### F-08 · Error notifications stack without limit

**File:** `static/script.js:396-421`

Each `showError()` call appends a new absolutely-positioned `div` at `top: 20px; right: 20px` with a 5-second lifetime. Three rapid errors produce three divs stacked on top of each other at the same coordinates, unreadable. There is no dedupe, no queue, and no container.

**Fix:** Single reusable toast element, or a stacking container with vertical offset.

---

## Medium

### F-09 · Google Fonts loaded from CDN

**File:** `templates/index.html:8`

An external network request on every page load. Adds latency on the free-tier HF Space, sends user IPs to Google, and the page renders with fallback fonts if the CDN is blocked (common on institutional networks in India). For a self-contained Docker image this is an avoidable external dependency.

**Fix:** Self-host the two Inter weights actually used as `.woff2` in `static/fonts/`. ~40KB, removes the dependency.

---

### F-10 · Client-side 10MB limit not enforced server-side

**File:** `static/script.js:278` checks `file.size > 10 * 1024 * 1024`. `app.py` has **no** `MAX_CONTENT_LENGTH`.

The check is trivially bypassed by posting directly to `/predict`. See `AUDIT_BACKEND.md` B-02 — this is really a backend finding, noted here because the frontend check creates a false sense that the limit exists.

---

### F-11 · `console.log` statements left in production

**File:** `static/script.js:454-456`

Three startup logs. Harmless but unprofessional, and they run on every page load in the deployed demo.

---

### F-12 · Camera captures at 1920×1080 with no framing guide

**File:** `static/script.js:172-173, 208-209`

Requests `width: {ideal: 1920}, height: {ideal: 1080}` — a wide landscape frame, which encourages capturing scenery rather than a single leaf. There is no overlay guide showing the user what to frame.

**Fix:** Add a square framing overlay in the camera view and capture a square crop, matching the training distribution.

---

### F-13 · No loading state on the analyze button itself

**File:** `static/script.js:312` — only a full-screen overlay. The button remains enabled underneath and can be clicked repeatedly, firing concurrent requests to an endpoint that runs 3–5 CNN forward passes per call.

**Fix:** Disable the button and show inline spinner state for the duration of the request.

---

### F-14 · Hardcoded colour values scattered rather than tokenised

**File:** `static/script.js:403` sets `background: #ef4444` inline; `style.css` defines CSS custom properties but the JS bypasses them.

**Fix:** Use `var(--error)` from the stylesheet.

---

### F-15 · No favicon

**File:** `templates/index.html` `<head>` — browser requests `/favicon.ico`, Flask returns 404 on every page load.

---

## Low

### F-16 · No `<meta name="description">` or Open Graph tags
Sharing the demo link produces no preview card. For a portfolio piece meant to be shared, this is a cheap win.

### F-17 · Nav links point to sections but there is no scroll-spy
`script.js:144-158` sets the active class on click only. Scrolling manually leaves the wrong link highlighted.

### F-18 · `alt="Preview"` on the uploaded image is not descriptive
`index.html:86`. Should reflect the actual filename or "Uploaded rice leaf image, pending analysis".

### F-19 · Footer says "© 2025" — hardcoded year
`index.html:267`. Now stale.

---

## Prioritised fix order

| # | Finding | Effort | Impact |
|---|---|---|---|
| 1 | F-01, F-02 — correct the accuracy and pathogen claims | 15 min | Removes false claims from a public page |
| 2 | F-03 — add photo-guidance copy | 1–2 h | Largest real-world accuracy gain available without retraining |
| 3 | F-06 — client-side crop | 3–4 h | Converts a large class of failures into correct predictions |
| 4 | F-04 — guard `result.details` | 10 min | Removes a hard-freeze failure mode |
| 5 | F-05 — accessibility pass | 4–6 h | Makes the tool usable by keyboard and screen reader |
| 6 | F-07 — responsive breakpoints | 3–4 h | Target audience is mobile |
| 7 | Everything else | — | Polish |

---

## What this section demonstrates

Auditing your own frontend against the backend's actual behaviour surfaces a class of bug that neither side shows in isolation: **the UI was confidently advertising a capability the model does not have, and simultaneously withholding the one instruction that would have made the model work.** F-01 and F-03 are the same failure viewed from two directions — a mismatch between what was measured and what was claimed.

The accessibility gap (F-05) is worth naming honestly rather than quietly fixing. Zero ARIA attributes across 273 lines is not an oversight in one place; it reflects not having considered non-mouse users at all during the original build. That is a normal thing to get wrong on a first project and a useful thing to be able to say you found and corrected.
