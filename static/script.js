/*
 * Rice Leaf Disease Detection — frontend
 * v3.0
 *
 * v2 addressed the AUDIT_FRONTEND findings (IDs still referenced inline below).
 * v3 adds the theme switch, a mobile nav, tabbed management guidance, and the
 * two advisory strips for A-12. Everything v2 fixed is carried over unchanged:
 * the focus trap, keyboard activation of the drop zone, toast de-duplication,
 * the concurrent-request guard, and guards around result.details.
 */

'use strict';

const $ = (id) => document.getElementById(id);

// Upload
const uploadArea = $('uploadArea');
const fileInput = $('fileInput');
const uploadContent = $('uploadContent');
const imagePreview = $('imagePreview');
const previewImage = $('previewImage');
const removeImage = $('removeImage');
const analyzeButton = $('analyzeButton');
const cropButton = $('cropButton');
const loadingOverlay = $('loadingOverlay');
const resultsSection = $('resultsSection');
const resultStatus = $('resultStatus');
const closeResults = $('closeResults');
const analyzeAnother = $('analyzeAnother');
const toastContainer = $('toastContainer');

// Camera
const cameraButton = $('cameraButton');
const cameraModal = $('cameraModal');
const cameraVideo = $('cameraVideo');
const cameraCanvas = $('cameraCanvas');
const captureBtn = $('captureBtn');
const switchCameraBtn = $('switchCameraBtn');
const closeCameraBtn = $('closeCameraBtn');
const cancelCameraBtn = $('cancelCameraBtn');

// Crop
const cropModal = $('cropModal');
const cropStage = $('cropStage');
const cropCanvas = $('cropCanvas');
const cropSelection = $('cropSelection');
const cropApplyBtn = $('cropApplyBtn');
const cropResetBtn = $('cropResetBtn');
const closeCropBtn = $('closeCropBtn');
const rejectionHelp = $('rejectionHelp');
const rejectionCropBtn = $('rejectionCropBtn');

// Results — header
const diagnosisBanner = $('diagnosisBanner');
const diagnosisIcon = $('diagnosisIcon');
const diagnosisTitle = $('diagnosisTitle');
const confidenceValue = $('confidenceValue');
const severityBadge = $('severityBadge');
const pathogenIndicator = $('pathogenIndicator');
const pathogenValue = $('pathogenValue');

// Results — advisory strips (A-12, A-02)
const agreementNotice = $('agreementNotice');
const agreementText = $('agreementText');
const reliabilityNotice = $('reliabilityNotice');
const reliabilityText = $('reliabilityText');

// Results — care panels
const careSummary = $('careSummary');
const factRow = $('factRow');
const descriptionText = $('descriptionText');
const symptomsBlock = $('symptomsBlock');
const symptomsList = $('symptomsList');
const recommendationText = $('recommendationText');
const firstStepsBlock = $('firstStepsBlock');
const firstStepsList = $('firstStepsList');
const culturalBlock = $('culturalBlock');
const culturalList = $('culturalList');
const chemBlock = $('chemBlock');
const chemActives = $('chemActives');
const chemCaution = $('chemCaution');
const preventionBlock = $('preventionBlock');
const preventionList = $('preventionList');
const escalateBox = $('escalateBox');
const escalateText = $('escalateText');
const linksBlock = $('linksBlock');
const linkList = $('linkList');
const helplineBlock = $('helplineBlock');
const helplineList = $('helplineList');

const requestRef = $('requestRef');

const MAX_BYTES = 10 * 1024 * 1024;
const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp'];

let selectedFile = null;
let cameraStream = null;
let currentFacingMode = 'environment';
let isAnalyzing = false;
let lastFocusedBeforeModal = null;

// ---------------------------------------------------------------------------
// Theme
// The stored preference is applied by an inline script in <head>, before first
// paint. This only handles switching it afterwards. Storage can throw in
// private mode, so every access is guarded — a failed write must not take the
// toggle down with it.
// ---------------------------------------------------------------------------

const themeToggle = $('themeToggle');

function systemPrefersDark() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function currentTheme() {
    return document.documentElement.getAttribute('data-theme')
        || (systemPrefersDark() ? 'dark' : 'light');
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('theme', theme); } catch (e) { /* non-fatal */ }
    if (themeToggle) {
        themeToggle.setAttribute(
            'aria-label',
            theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme',
        );
    }
}

if (themeToggle) {
    applyTheme(currentTheme());   // sync the label with whatever <head> decided
    themeToggle.addEventListener('click', () => {
        applyTheme(currentTheme() === 'dark' ? 'light' : 'dark');
    });
}

// If the user has expressed no preference, follow the OS when it changes.
if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', (e) => {
        let stored = null;
        try { stored = localStorage.getItem('theme'); } catch (err) { /* ignore */ }
        if (!stored) applyTheme(e.matches ? 'dark' : 'light');
    });
}

// ---------------------------------------------------------------------------
// Mobile navigation
// ---------------------------------------------------------------------------

const navToggle = $('navToggle');
const primaryNav = $('primaryNav');

function setNav(open) {
    if (!primaryNav || !navToggle) return;
    primaryNav.dataset.open = String(open);
    navToggle.setAttribute('aria-expanded', String(open));
    navToggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
}

if (navToggle) {
    navToggle.addEventListener('click', () => setNav(primaryNav.dataset.open !== 'true'));
}

// data-open only takes effect under the mobile breakpoint; above it the CSS
// ignores the attribute, so resetting on resize keeps the two in step.
window.addEventListener('resize', () => { if (window.innerWidth > 760) setNav(false); });

// ---------------------------------------------------------------------------
// Toasts — F-08
// v1 appended each error as an absolutely-positioned div at the same
// coordinates with no container, so simultaneous errors stacked on top of each
// other and became unreadable. They also carried hardcoded colours (F-14).
// ---------------------------------------------------------------------------

function showError(message) { showToast(message, 'error'); }

function showToast(message, kind = 'error') {
    // Collapse duplicates rather than stacking them.
    const existing = [...toastContainer.children]
        .find((n) => n.dataset.message === message);
    if (existing) {
        existing.classList.remove('toast-pulse');
        void existing.offsetWidth;
        existing.classList.add('toast-pulse');
        return;
    }

    const toast = document.createElement('div');
    toast.className = `toast toast-${kind}`;
    toast.dataset.message = message;
    toast.textContent = message;

    let timer;
    const remove = () => {
        clearTimeout(timer);
        toast.classList.add('toast-out');
        toast.addEventListener('animationend', () => toast.remove(), { once: true });
        setTimeout(() => toast.remove(), 400);
    };

    const dismiss = document.createElement('button');
    dismiss.type = 'button';
    dismiss.className = 'toast-close';
    dismiss.setAttribute('aria-label', 'Dismiss message');
    dismiss.textContent = '✕';
    dismiss.addEventListener('click', remove);
    toast.appendChild(dismiss);

    toastContainer.appendChild(toast);
    while (toastContainer.children.length > 3) toastContainer.firstElementChild.remove();
    timer = setTimeout(remove, 6000);
}

// ---------------------------------------------------------------------------
// Modal helpers — F-05
// v1 modals had no role, no focus trap and no Escape handling, so keyboard
// users could tab out of an open modal into the page behind it.
// ---------------------------------------------------------------------------

const FOCUSABLE = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

function openModal(modal) {
    lastFocusedBeforeModal = document.activeElement;
    modal.hidden = false;
    document.body.classList.add('modal-open');
    const first = modal.querySelector(FOCUSABLE);
    if (first) first.focus();
    modal.addEventListener('keydown', trapFocus);
}

function closeModal(modal) {
    modal.hidden = true;
    document.body.classList.remove('modal-open');
    modal.removeEventListener('keydown', trapFocus);
    if (lastFocusedBeforeModal) lastFocusedBeforeModal.focus();
}

function trapFocus(e) {
    if (e.key !== 'Tab') return;
    const nodes = [...e.currentTarget.querySelectorAll(FOCUSABLE)]
        .filter((n) => !n.disabled && n.offsetParent !== null);
    if (!nodes.length) return;
    const first = nodes[0];
    const last = nodes[nodes.length - 1];
    if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
    }
}

document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (!cropModal.hidden) closeCropTool();
    else if (!cameraModal.hidden) closeCamera();
    else if (primaryNav && primaryNav.dataset.open === 'true') {
        setNav(false);
        navToggle.focus();
    }
});

// ---------------------------------------------------------------------------
// Tabs — WAI-ARIA tab pattern
// Roving tabindex: exactly one tab is in the tab order, and the arrow keys move
// between them. Without this, a keyboard user has to Tab through every tab to
// reach the panel content.
// ---------------------------------------------------------------------------

const tabs = [...document.querySelectorAll('.tablist [role="tab"]')];
const panels = tabs.map((t) => $(t.getAttribute('aria-controls')));

function selectTab(index, { focus = true } = {}) {
    tabs.forEach((tab, i) => {
        const on = i === index;
        tab.setAttribute('aria-selected', String(on));
        tab.tabIndex = on ? 0 : -1;
        if (panels[i]) panels[i].hidden = !on;
    });
    if (focus && tabs[index]) tabs[index].focus();
}

tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => selectTab(i, { focus: false }));
    tab.addEventListener('keydown', (e) => {
        const keys = {
            ArrowRight: (i + 1) % tabs.length,
            ArrowLeft: (i - 1 + tabs.length) % tabs.length,
            Home: 0,
            End: tabs.length - 1,
        };
        if (!(e.key in keys)) return;
        e.preventDefault();
        selectTab(keys[e.key]);
    });
});

// ---------------------------------------------------------------------------
// File handling
// ---------------------------------------------------------------------------

function handleFileSelect(file) {
    if (!file) return;

    if (!ACCEPTED.includes(file.type)) {
        showError('Please select a JPG, PNG or WebP image.');
        return;
    }
    // Mirrors the server limit. The server enforces it too (B-02) — this is
    // only to fail fast, not a security control.
    if (file.size > MAX_BYTES) {
        showError('That image is larger than 10MB. Please choose a smaller file.');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.alt = `Selected image ${file.name}, awaiting analysis`;
        uploadContent.hidden = true;
        imagePreview.hidden = false;
        analyzeButton.disabled = false;
        cropButton.hidden = false;
        uploadArea.setAttribute('aria-label', `Image selected: ${file.name}. Press Enter to choose a different file.`);
    };
    reader.onerror = () => showError('Could not read that file.');
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    uploadContent.hidden = false;
    imagePreview.hidden = true;
    previewImage.src = '';
    analyzeButton.disabled = true;
    cropButton.hidden = true;
    rejectionHelp.hidden = true;
    uploadArea.setAttribute('aria-label', 'Upload a rice leaf image. Press Enter to browse for a file, or drag a file here.');
}

// ---------------------------------------------------------------------------
// Crop tool — F-06
// Cropping background away is the single most effective thing a user can do:
// diagnostics/test_background.py showed a field photo going from not_rice_leaf
// (99.08%) to the correct disease purely by trimming the surroundings.
// Canvas-based, no external dependency.
// ---------------------------------------------------------------------------

let cropImage = null;
let cropRect = null;
let dragStart = null;

function openCropTool() {
    if (!selectedFile) return;
    const img = new Image();
    img.onload = () => {
        cropImage = img;
        const maxW = Math.min(window.innerWidth - 80, 640);
        const scale = Math.min(maxW / img.width, 420 / img.height, 1);
        cropCanvas.width = Math.round(img.width * scale);
        cropCanvas.height = Math.round(img.height * scale);
        cropCanvas.getContext('2d').drawImage(img, 0, 0, cropCanvas.width, cropCanvas.height);
        cropRect = null;
        cropSelection.hidden = true;
        openModal(cropModal);
    };
    img.onerror = () => showError('Could not open that image for cropping.');
    img.src = URL.createObjectURL(selectedFile);
}

function closeCropTool() { closeModal(cropModal); }

function pointerPos(e) {
    const r = cropCanvas.getBoundingClientRect();
    const src = e.touches ? e.touches[0] : e;
    return {
        x: Math.max(0, Math.min(cropCanvas.width, src.clientX - r.left)),
        y: Math.max(0, Math.min(cropCanvas.height, src.clientY - r.top)),
    };
}

function beginDrag(e) {
    e.preventDefault();
    dragStart = pointerPos(e);
    cropSelection.hidden = false;
    updateSelection(dragStart);
}

function moveDrag(e) {
    if (!dragStart) return;
    e.preventDefault();
    updateSelection(pointerPos(e));
}

function endDrag() { dragStart = null; }

function updateSelection(pos) {
    const x = Math.min(dragStart.x, pos.x);
    const y = Math.min(dragStart.y, pos.y);
    const w = Math.abs(pos.x - dragStart.x);
    const h = Math.abs(pos.y - dragStart.y);
    cropRect = { x, y, w, h };
    const canvasBox = cropCanvas.getBoundingClientRect();
    const stageBox = cropStage.getBoundingClientRect();
    const offX = canvasBox.left - stageBox.left;
    const offY = canvasBox.top - stageBox.top;
    Object.assign(cropSelection.style, {
        left: `${offX + x}px`,
        top: `${offY + y}px`,
        width: `${w}px`,
        height: `${h}px`,
    });
}

function applyCrop() {
    if (!cropRect || cropRect.w < 20 || cropRect.h < 20) {
        showError('Drag a larger area over the leaf first.');
        return;
    }
    const sx = cropImage.width / cropCanvas.width;
    const sy = cropImage.height / cropCanvas.height;
    const out = document.createElement('canvas');
    out.width = Math.round(cropRect.w * sx);
    out.height = Math.round(cropRect.h * sy);
    out.getContext('2d').drawImage(
        cropImage,
        Math.round(cropRect.x * sx), Math.round(cropRect.y * sy),
        out.width, out.height,
        0, 0, out.width, out.height,
    );
    out.toBlob((blob) => {
        if (!blob) { showError('Crop failed. Please try again.'); return; }
        handleFileSelect(new File([blob], 'cropped.jpg', { type: 'image/jpeg' }));
        closeCropTool();
        showToast('Cropped. Analyse again to re-check.', 'info');
    }, 'image/jpeg', 0.95);
}

cropCanvas.addEventListener('mousedown', beginDrag);
cropCanvas.addEventListener('mousemove', moveDrag);
window.addEventListener('mouseup', endDrag);
cropCanvas.addEventListener('touchstart', beginDrag, { passive: false });
cropCanvas.addEventListener('touchmove', moveDrag, { passive: false });
window.addEventListener('touchend', endDrag);

cropApplyBtn.addEventListener('click', applyCrop);
cropResetBtn.addEventListener('click', () => { cropRect = null; cropSelection.hidden = true; });
closeCropBtn.addEventListener('click', closeCropTool);
cropButton.addEventListener('click', openCropTool);
rejectionCropBtn.addEventListener('click', () => { resultsSection.hidden = true; openCropTool(); });
cropModal.addEventListener('click', (e) => { if (e.target === cropModal) closeCropTool(); });

// ---------------------------------------------------------------------------
// Camera — F-12
// v1 requested 1920x1080 and captured the full landscape frame, encouraging
// scenery shots. Capture is now cropped to the centred square shown in the
// framing guide, matching the 1:1 training images.
// ---------------------------------------------------------------------------

async function openCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
        showError('This browser does not support camera capture.');
        return;
    }
    try {
        openModal(cameraModal);
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: currentFacingMode, width: { ideal: 1440 }, height: { ideal: 1440 } },
            audio: false,
        });
        cameraVideo.srcObject = cameraStream;
    } catch (error) {
        closeCamera();
        if (error.name === 'NotAllowedError') showError('Camera access denied. Allow it in your browser settings.');
        else if (error.name === 'NotFoundError') showError('No camera found on this device.');
        else showError('Unable to access the camera.');
    }
}

async function switchCamera() {
    if (!cameraStream) return;
    currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
    cameraStream.getTracks().forEach((t) => t.stop());
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: currentFacingMode, width: { ideal: 1440 }, height: { ideal: 1440 } },
            audio: false,
        });
        cameraVideo.srcObject = cameraStream;
    } catch {
        currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
        showError('Unable to switch camera. This device may have only one.');
        openCamera();
    }
}

function capturePhoto() {
    if (!cameraStream) return;
    const vw = cameraVideo.videoWidth;
    const vh = cameraVideo.videoHeight;
    if (!vw || !vh) { showError('Camera is not ready yet.'); return; }

    const side = Math.min(vw, vh);
    cameraCanvas.width = side;
    cameraCanvas.height = side;
    cameraCanvas.getContext('2d').drawImage(
        cameraVideo,
        Math.floor((vw - side) / 2), Math.floor((vh - side) / 2), side, side,
        0, 0, side, side,
    );
    cameraCanvas.toBlob((blob) => {
        if (!blob) { showError('Capture failed.'); return; }
        handleFileSelect(new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' }));
        closeCamera();
    }, 'image/jpeg', 0.95);
}

function closeCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach((t) => t.stop());
        cameraStream = null;
    }
    cameraVideo.srcObject = null;
    closeModal(cameraModal);
}

// ---------------------------------------------------------------------------
// Analyze
// ---------------------------------------------------------------------------

async function analyzeImage() {
    if (!selectedFile || isAnalyzing) return;

    // F-13: v1 left the button enabled behind a full-screen overlay, so repeated
    // clicks fired concurrent requests at an endpoint running 3-5 forward passes.
    isAnalyzing = true;
    analyzeButton.disabled = true;
    analyzeButton.classList.add('is-loading');
    loadingOverlay.hidden = false;

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        const response = await fetch('/predict', { method: 'POST', body: formData });

        let payload = null;
        try { payload = await response.json(); } catch { /* non-JSON body */ }

        if (!response.ok) {
            if (response.status === 429) throw new Error(payload?.error || 'Too many requests. Please wait a moment.');
            if (response.status === 413) throw new Error('That image is too large. Maximum 10MB.');
            if (response.status === 503) throw new Error('The model is still loading. Try again in a minute.');
            throw new Error(payload?.error || 'Analysis failed. Please try again.');
        }
        if (!payload) throw new Error('Unexpected response from the server.');
        if (payload.error) throw new Error(payload.error);

        displayResults(payload);
    } catch (error) {
        showError(error.message || 'An error occurred during analysis.');
    } finally {
        isAnalyzing = false;
        analyzeButton.disabled = !selectedFile;
        analyzeButton.classList.remove('is-loading');
        loadingOverlay.hidden = true;
    }
}

// --- rendering helpers -----------------------------------------------------

/** Replace a list's contents. Returns false when there was nothing to show, so
 *  the caller can hide the surrounding block rather than leave an empty
 *  heading floating above nothing. */
function fillList(listEl, items) {
    if (!listEl) return false;
    listEl.replaceChildren();
    if (!Array.isArray(items) || !items.length) return false;
    items.forEach((text) => {
        const li = document.createElement('li');
        li.textContent = text;
        listEl.appendChild(li);
    });
    return true;
}

function toggleBlock(blockEl, shown) {
    if (blockEl) blockEl.hidden = !shown;
}

function fillFacts(care) {
    const rows = [
        ['Also known as', care.also_known_as],
        ['Spreads by', care.spreads_by],
        ['Favoured by', care.favoured_by],
    ].filter(([, v]) => v);
    factRow.replaceChildren();
    rows.forEach(([label, value]) => {
        const dt = document.createElement('dt');
        dt.textContent = label;
        const dd = document.createElement('dd');
        dd.textContent = value;
        factRow.append(dt, dd);
    });
    factRow.hidden = rows.length === 0;
}

function fillLinks(links) {
    linkList.replaceChildren();
    if (!Array.isArray(links) || !links.length) return false;
    links.forEach(({ label, url }) => {
        const a = document.createElement('a');
        a.href = url;
        a.target = '_blank';
        a.rel = 'noopener';
        a.textContent = label;
        linkList.appendChild(a);
    });
    return true;
}

function fillHelplines(helplines) {
    helplineList.replaceChildren();
    if (!Array.isArray(helplines) || !helplines.length) return false;
    helplines.forEach((h) => {
        const box = document.createElement('div');
        box.className = 'helpline';

        const region = document.createElement('span');
        region.className = 'helpline-region';
        region.textContent = h.region;

        const name = document.createElement('span');
        name.className = 'helpline-name';
        name.textContent = h.name;

        const contact = document.createElement('span');
        contact.className = 'helpline-contact';
        contact.textContent = h.contact;

        const note = document.createElement('span');
        note.className = 'helpline-note';
        note.textContent = h.note || '';

        box.append(region, name, contact, note);
        helplineList.appendChild(box);
    });
    return true;
}

function renderCare(care, support) {
    const c = care || {};

    careSummary.textContent = c.summary || '';
    careSummary.hidden = !c.summary;
    fillFacts(c);

    toggleBlock(symptomsBlock, fillList(symptomsList, c.symptoms));
    toggleBlock(firstStepsBlock, fillList(firstStepsList, c.first_steps));
    toggleBlock(culturalBlock, fillList(culturalList, c.cultural));
    toggleBlock(preventionBlock, fillList(preventionList, c.prevention));

    // Chemical guidance. Deliberately shows the caution even when there are no
    // actives to list — for uncertain results the caution *is* the message.
    const chem = c.chemical;
    chemActives.replaceChildren();
    if (chem) {
        (chem.actives || []).forEach((a) => {
            const pill = document.createElement('span');
            pill.className = 'chem-pill';
            pill.textContent = a;
            chemActives.appendChild(pill);
        });
        chemCaution.textContent = chem.caution || '';
    }
    toggleBlock(chemBlock, Boolean(chem));

    escalateText.textContent = c.escalate_when || '';
    toggleBlock(escalateBox, Boolean(c.escalate_when));

    // Always offer the general references, not only the disease-specific ones.
    const links = [...(c.links || []), ...((support && support.references) || [])];
    toggleBlock(linksBlock, fillLinks(links));
    toggleBlock(helplineBlock, fillHelplines(support && support.helplines));
}

function displayResults(result) {
    // F-04: v1 read result.details.* with no guard. Any response without
    // `details` threw a TypeError after the overlay had already been hidden,
    // leaving a frozen page and a misleading generic error.
    const d = result.details || {};
    const diagnosis = result.diagnosis || 'Unknown';

    diagnosisIcon.textContent = result.icon || '🌿';
    diagnosisTitle.textContent = diagnosis;
    confidenceValue.textContent = result.confidence ?? '—';

    diagnosisBanner.classList.remove('healthy', 'disease', 'non-leaf', 'uncertain');
    const isRejection = /not a rice leaf/i.test(diagnosis);
    if (result.abstained || /uncertain/i.test(diagnosis)) diagnosisBanner.classList.add('uncertain');
    else if (/healthy/i.test(diagnosis)) diagnosisBanner.classList.add('healthy');
    else if (isRejection) diagnosisBanner.classList.add('non-leaf');
    else diagnosisBanner.classList.add('disease');

    // Surface the crop path exactly when it is most likely to help.
    rejectionHelp.hidden = !isRejection;

    severityBadge.textContent = result.severity ?? 'Unknown';
    severityBadge.classList.remove('high', 'medium', 'low', 'none');
    severityBadge.classList.add({
        High: 'high', Medium: 'medium', Low: 'low', None: 'none', 'N/A': 'none',
    }[result.severity] || 'none');

    if (result.pathogen) {
        pathogenValue.textContent = result.pathogen;
        pathogenIndicator.hidden = false;
    } else {
        pathogenIndicator.hidden = true;
    }

    descriptionText.textContent = result.description ?? '—';
    recommendationText.textContent = result.recommendation ?? '—';

    // A-12 · stage disagreement. Only meaningful when Stage 2 actually ran;
    // `stages_agree` is null otherwise, which is not the same as "they agree".
    // Two kinds of doubt, and the reader only needs to know which one they are
    // looking at — not the internals that produced it. The strong wording is
    // reserved for a genuine contradiction between the stages; a close second
    // place gets the calmer blue treatment.
    const ru = result.runner_up;
    const agreementHeading = agreementNotice.querySelector('strong');
    agreementNotice.classList.remove('result-notice--info');

    if (result.stages_agree === false) {
        const alt = (d.stage1_prediction && d.stage1_prediction !== diagnosis)
            ? d.stage1_prediction
            : (ru && ru.label);
        agreementHeading.textContent = 'Not a confident result';
        agreementText.textContent = alt
            ? `It could also be ${alt}. Compare the symptoms of both, and check with an agronomist before treating.`
            : 'Compare the symptoms in the disease library, and check with an agronomist before treating.';
        agreementNotice.hidden = false;
    } else if (ru && ru.margin_points < 15 && !isRejection) {
        agreementNotice.classList.add('result-notice--info');
        agreementHeading.textContent = 'Close call';
        agreementText.textContent =
            `${ru.label} was a close second. Worth ruling it out before you treat.`;
        agreementNotice.hidden = false;
    } else {
        agreementNotice.hidden = true;
    }

    // A-02 · classes the model has weaker evidence for. Fixed wording: the
    // per-disease explanation was three sentences about training data, which is
    // the author's problem, not the reader's.
    reliabilityText.textContent = (result.care && result.care.reliability_note)
        ? 'The model is less reliable for this disease. Confirm it against the fact sheet or with an agronomist before treating.'
        : '';
    reliabilityNotice.hidden = !(result.care && result.care.reliability_note);

    renderCare(result.care, result.support);

    requestRef.textContent = result.ref ?? '—';

    selectTab(0, { focus: false });
    resultsSection.hidden = false;

    // One concise sentence for screen readers, instead of reading four panels.
    resultStatus.textContent =
        `Result: ${diagnosis}, ${result.confidence ?? 'unknown'} confidence.`
        + (result.severity ? ` Severity ${result.severity}.` : '');

    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    resultsSection.focus({ preventScroll: true });
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------

uploadArea.addEventListener('click', (e) => {
    if (e.target.closest('.camera-button') || e.target.closest('.remove-image')) return;
    if (!selectedFile) fileInput.click();
});

// F-05: the drop zone is a div, so it needs explicit keyboard activation.
uploadArea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => handleFileSelect(e.target.files[0]));

uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFileSelect(file);
    else showError('Please drop a valid image file.');
});

removeImage.addEventListener('click', (e) => { e.stopPropagation(); resetUpload(); });
analyzeButton.addEventListener('click', analyzeImage);
closeResults.addEventListener('click', () => {
    resultsSection.hidden = true;
    analyzeButton.focus();
});
analyzeAnother.addEventListener('click', () => {
    resetUpload();
    resultsSection.hidden = true;
    document.getElementById('diagnose').scrollIntoView({ behavior: 'smooth' });
    uploadArea.focus();
});

cameraButton.addEventListener('click', (e) => { e.stopPropagation(); openCamera(); });
captureBtn.addEventListener('click', capturePhoto);
switchCameraBtn.addEventListener('click', switchCamera);
closeCameraBtn.addEventListener('click', closeCamera);
cancelCameraBtn.addEventListener('click', closeCamera);
cameraModal.addEventListener('click', (e) => { if (e.target === cameraModal) closeCamera(); });

// Nav
const navLinks = [...document.querySelectorAll('.nav-link')];
navLinks.forEach((link) => {
    link.addEventListener('click', () => {
        // No preventDefault: the href is a real anchor, and CSS scroll-behavior
        // plus scroll-padding-top already handle the smooth scroll and the
        // sticky-header offset. Letting the browser do it keeps the URL hash
        // correct, which v2 discarded.
        navLinks.forEach((l) => l.classList.remove('active'));
        link.classList.add('active');
        setNav(false);
    });
});

// F-17: v1 only set the active link on click, so scrolling manually left the
// wrong item highlighted.
if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            const id = entry.target.id;
            navLinks.forEach((l) => l.classList.toggle('active', l.getAttribute('href') === `#${id}`));
        });
    }, { rootMargin: '-40% 0px -55% 0px' });
    ['home', 'diagnose', 'library', 'how-it-works', 'about'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) observer.observe(el);
    });
}

// F-19
const footerYear = $('footerYear');
if (footerYear) footerYear.textContent = new Date().getFullYear();

// F-11: v1 shipped three console.log calls that ran on every page load.
