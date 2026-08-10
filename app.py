"""
Rice Leaf Disease Detection - Flask Backend
Version: 2.0

Changes in 2.0 address findings from AUDIT_BACKEND.md. Each fix is tagged with
its finding ID so the audit and the code stay traceable to each other.

MODEL STATUS (August 2026):
    Stage 1 was retrained on a rebuilt dataset - hard negatives (other crops'
    leaves on plain backgrounds) plus real paddy-field photographs - which fixed
    the background shortcut documented in AUDIT_BACKEND.md M-01. On the held-out
    test set: 94.51% overall, 96.52% studio, 81.67% field, and zero rice images
    misclassified as not_rice_leaf.

    Stage 2 checkpoints are unchanged from v1. The shortcut lived entirely in
    Stage 1, which owns the not_rice_leaf class.

    Remaining known biases, surfaced to users via `input_guidance`:
      - leaf_scald and narrow_brown_spot have no field training photographs, so
        their perfect test scores are inflated.
      - Compositing a leaf onto an unrelated scene can still flip the prediction
        to not_rice_leaf (v2/scripts/05_shortcut_check.py).
"""

import io
import os
import uuid
import logging
import threading
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, g
from flask_cors import CORS
from PIL import Image
from torchvision import transforms, models

# ============================================
# CONFIGURATION
# ============================================

IMG_SIZE = 224

# A-06: lowered from 0.95 on evidence from v2/reports/calibration.json.
#
# The ensemble is systematically UNDER-confident - averaging three softmax
# outputs pulls the maximum down whenever the members disagree slightly, even
# though the argmax stays right. Every reliability bin is negative, by up to 30
# points (ECE 11.98%).
#
# Consequence: a 0.95 bar was almost never cleared, so Stage 2 ran on more than
# half of all requests - the opposite of what the two-stage design is for. And
# in the 0.80-0.90 band Stage 1 was already 100% accurate on 120 validation
# images, so routing those to Stage 2 could only make them worse.
#
# 0.85 skips Stage 2 on the band where Stage 1 is already perfect.
CONFIDENCE_THRESHOLD = 0.85

# B-02: cap request size. Without this, file.read() allocates whatever the
# client sends. The 10MB check in script.js is client-side only and bypassable.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# B-02: PIL decompression-bomb guard. A small file can decode to a huge bitmap.
Image.MAX_IMAGE_PIXELS = 50_000_000

# B-04: CORS allowlist. `CORS(app)` with no args allowed every origin, which
# combined with an unauthenticated expensive endpoint let any site use this
# Space as free compute. Same-origin frontend needs no CORS at all; set
# ALLOWED_ORIGINS only if you deliberately want cross-origin API access.
ALLOWED_ORIGINS = [o for o in os.environ.get('ALLOWED_ORIGINS', '').split(',') if o]

# B-03: simple in-process rate limit. Deliberately dependency-free - adding
# flask-limiter would pull in a redis/limits stack for a single-worker Space.
RATE_LIMIT_REQUESTS = int(os.environ.get('RATE_LIMIT_REQUESTS', '15'))
RATE_LIMIT_WINDOW_S = int(os.environ.get('RATE_LIMIT_WINDOW_S', '60'))

# M-14: ensemble members do not perform equally, so a plain mean gives the
# weakest model the same vote as the strongest. Set to False to restore v1
# behaviour (uniform averaging).
USE_ACCURACY_WEIGHTED_ENSEMBLE = True

# M-13: Stage 2 can only emit the 5 disease classes. When Stage 1 is uncertain
# AND Stage 2 is also uncertain, the honest answer is "I don't know" rather
# than forcing a disease label. Below this, the API abstains.
#
# A-06 note: deliberately NOT changed by the calibration study.
# v2/scripts/08_calibration.py sweeps STAGE 1 ensemble confidence, but this
# threshold applies to STAGE 2 confidence - a different distribution over a
# different label space (5 classes, not 7). Tuning one from the other would be
# unsound. The under-confidence finding almost certainly carries over, since
# both stages average softmax the same way, so 0.45 is probably conservative -
# but that needs measuring on Stage 2 directly before being changed.
STAGE2_ABSTAIN_THRESHOLD = 0.45

# B-15: structured logging with a request id so a user-visible error can be
# traced to a log line.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
)
logger = logging.getLogger('rice-detector')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES

if ALLOWED_ORIGINS:
    CORS(app, origins=ALLOWED_ORIGINS)
    logger.info(f"CORS enabled for: {ALLOWED_ORIGINS}")
else:
    logger.info("CORS disabled (same-origin only)")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# ============================================
# CLASSES
# ============================================

CLASS_NAMES_STAGE1 = [
    'bacterial_leaf_blight', 'brown_spot', 'healthy',
    'leaf_blast', 'leaf_scald', 'narrow_brown_spot', 'not_rice_leaf'
]

CLASS_NAMES_STAGE2 = [
    'bacterial_leaf_blight', 'brown_spot', 'leaf_blast',
    'leaf_scald', 'narrow_brown_spot'
]

# M-03: was named BACTERIAL_DISEASES, but only bacterial_leaf_blight is
# bacterial - the other four are fungal. The list means "any disease class",
# so the logic was right and only the name was wrong.
DISEASE_CLASSES = [
    'bacterial_leaf_blight', 'brown_spot', 'leaf_blast',
    'leaf_scald', 'narrow_brown_spot'
]

# Guidance returned with every prediction. Framing still affects accuracy
# (96.5% studio vs 81.7% field), though field photos are no longer rejected
# outright as they were before the August 2026 retrain.
INPUT_GUIDANCE = (
    "Field photographs work, but accuracy is higher on a single leaf laid flat "
    "on plain paper (96.5% vs 81.7% on the test set). Fill the frame with the "
    "leaf and avoid hard shadows."
)

# M-04: leaf_scald was described as bacterial. It is caused by Microdochium
# oryzae, a fungus. Pathogen type is now an explicit field rather than being
# buried in prose, so this class of error is harder to reintroduce.
#
# CARE CONTENT — scope and deliberate omissions
# --------------------------------------------
# Each disease carries a `care` block used by the results panel. Two rules
# governed what went into it:
#
# 1. NO DOSAGES, NO BRAND NAMES. Active ingredients are named so the user can
#    ask for them by name; rates are not, because the correct rate depends on
#    formulation, growth stage, local resistance and national registration.
#    Publishing a number here would be advice this project cannot stand behind,
#    and the README already states that treatment guidance is generic.
# 2. CULTURAL CONTROL FIRST. For most of these diseases nutrition, water and
#    variety choice do more than a spray, and they carry no resistance or
#    residue risk. Chemical control is listed last and framed as a decision to
#    confirm locally, not a default.
#
# Fact sheets are IRRI Rice Knowledge Bank, CC BY-NC-SA. Links verified live.
RKB = 'http://www.knowledgebank.irri.org/training/fact-sheets/pest-management/diseases/item'

SUPPORT_RESOURCES = {
    'helplines': [
        {
            'region': 'India',
            'name': 'Kisan Call Centre',
            'contact': '1800-180-1551',
            'note': 'Toll-free, 6am–10pm, 22 languages. Ask for a plant-protection officer.',
        },
        {
            'region': 'India',
            'name': 'Local Krishi Vigyan Kendra (KVK)',
            'contact': 'kvk.icar.gov.in',
            'note': 'District-level centres that will inspect a field in person.',
        },
    ],
    'references': [
        {
            'label': 'IRRI Rice Doctor — interactive diagnosis',
            'url': 'http://www.knowledgebank.irri.org/decision-tools/rice-doctor',
        },
        {
            'label': 'IRRI Rice Knowledge Bank — all disease fact sheets',
            'url': 'http://www.knowledgebank.irri.org/training/fact-sheets/pest-management/diseases',
        },
        {
            'label': 'IRRI Leaf Color Chart — nitrogen without guesswork',
            'url': 'http://www.knowledgebank.irri.org/step-by-step-production/growth/soil-fertility/leaf-color-chart',
        },
    ],
}

DISEASE_INFO = {
    'healthy': {
        'name': 'Healthy Rice Leaf',
        'pathogen': None,
        'severity': 'None',
        'description': 'The leaf appears healthy, with no visible signs of disease.',
        'recommendation': 'Continue regular monitoring and maintain good agricultural practices.',
        'icon': '✅',
        'care': {
            'summary': (
                'Nothing to treat. The useful work now is keeping it that way — '
                'most rice diseases are far cheaper to prevent than to cure.'
            ),
            'symptoms': [
                'Uniform green colour with no lesions, streaks or spots',
                'No water-soaking at the margins, no drying at the tip',
            ],
            'first_steps': [
                'Scout the field weekly, and after every storm — wind-driven rain spreads bacterial blight',
                'Check several plants from different parts of the field, not just the edge',
                'Use a leaf colour chart before adding nitrogen rather than applying on a fixed schedule',
            ],
            'cultural': [
                'Keep nitrogen balanced — excess nitrogen is the single largest risk factor for blast and leaf scald',
                'Maintain potassium and silicon, which harden tissue against brown spot and narrow brown spot',
                'Remove stubble and volunteer plants between seasons to break the disease cycle',
                'Use certified, disease-free seed',
            ],
            'chemical': None,
            'escalate_when': (
                'Any spreading lesion, or yellowing that appears in patches rather '
                'than uniformly. Photograph and re-check.'
            ),
            'links': [],
        },
    },
    'not_rice_leaf': {
        'name': 'Not a Rice Leaf',
        'pathogen': None,
        'severity': 'N/A',
        'description': 'The image does not appear to show a rice leaf.',
        'recommendation': (
            'If this is a rice leaf, crop tightly around a single leaf and try again. '
            'Paddy-field backgrounds are handled, but very cluttered or unrelated '
            'scenes can still be rejected.'
        ),
        'icon': '❌',
        'care': {
            'summary': (
                'No diagnosis was attempted, because the image did not read as a rice '
                'leaf. If it is one, this is almost always fixable from the photo.'
            ),
            'symptoms': [],
            'first_steps': [
                'Open the crop tool and drag a box around one leaf — this alone resolves most rejections',
                'Retake with the leaf laid flat on plain paper, filling the frame',
                'Avoid hard shadows, glare and motion blur',
            ],
            'cultural': [],
            'chemical': None,
            'escalate_when': (
                'If a tightly cropped, well-lit rice leaf is still rejected, that is '
                'a model failure rather than a photo problem — it is worth reporting '
                'on the project repository.'
            ),
            'links': [],
        },
    },
    'uncertain': {
        'name': 'Uncertain',
        'pathogen': None,
        'severity': 'Unknown',
        'description': (
            'The models disagree or are not confident enough to give a diagnosis.'
        ),
        'recommendation': (
            'Retake the photo with better lighting, a plain background and the leaf '
            'filling the frame. If the result stays uncertain, consult an agronomist.'
        ),
        'icon': '❓',
        'care': {
            'summary': (
                'The system withheld a diagnosis on purpose. A wrong confident answer '
                'would send you to the wrong treatment, which costs more than no answer.'
            ),
            'symptoms': [],
            'first_steps': [
                'Retake in even, indirect light with the leaf flat and filling the frame',
                'Photograph a second, clearly affected leaf — early lesions are genuinely ambiguous',
                'Note what else is happening in the field: water depth, recent nitrogen, recent weather',
            ],
            'cultural': [],
            'chemical': {
                'actives': [],
                'caution': (
                    'Do not spray on an uncertain diagnosis. Several of these diseases '
                    'need different actives, and brown spot is often a nutrition problem '
                    'that a fungicide will not fix.'
                ),
            },
            'escalate_when': (
                'If the result stays uncertain on a good photograph, treat it as a case '
                'for a human. Use IRRI Rice Doctor or call an extension officer.'
            ),
            'links': [
                {'label': 'IRRI Rice Doctor — step-by-step diagnosis',
                 'url': 'http://www.knowledgebank.irri.org/decision-tools/rice-doctor'},
            ],
        },
    },
    'bacterial_leaf_blight': {
        'name': 'Bacterial Leaf Blight',
        'pathogen': 'Bacterial (Xanthomonas oryzae)',
        'severity': 'High',
        'description': 'A serious bacterial disease causing water-soaked lesions that turn yellow or white.',
        'recommendation': 'Use resistant varieties, ensure field drainage, avoid excess nitrogen. Bactericides are largely ineffective once established.',
        'icon': '\U0001f9a0',
        'care': {
            'summary': (
                'The most damaging bacterial disease of rice. It moves in water and wind-driven '
                'rain, so it spreads fastest right after a storm. There is no reliable cure once '
                'it is established — control is about slowing spread and protecting next season.'
            ),
            'also_known_as': 'Kresek, when it wilts seedlings',
            'spreads_by': 'Wind-driven rain, irrigation water, infected seed and stubble, and hands and tools moving through a wet crop',
            'favoured_by': 'Storms and flooding, high nitrogen, temperatures around 25–34 °C, deep standing water',
            'symptoms': [
                'Water-soaked streaks starting at the leaf tip or margin',
                'Lesions turn yellow, then straw-white, with a wavy edge',
                'Milky bacterial ooze on cut leaves in early morning humidity',
                'In seedlings, whole leaves roll, wilt and die — this is kresek',
            ],
            'first_steps': [
                'Drain the field to shallow water. Deep standing water carries the bacterium between plants',
                'Stop all nitrogen top-dressing immediately — nitrogen accelerates this disease',
                'Stay out of the crop while leaves are wet; you will carry it down the rows on your legs and tools',
                'Do not clip leaf tips at transplanting, which is a direct entry wound',
            ],
            'cultural': [
                'Plant resistant varieties — this is the only control that reliably works. Ask for varieties carrying Xa4, xa5, Xa7 or Xa21',
                'Use certified seed; the bacterium is seed-borne',
                'Plough in stubble and remove weed hosts and ratoons after harvest',
                'Widen spacing so the canopy dries faster',
                'Let the field dry between irrigations rather than holding deep water',
            ],
            'chemical': {
                'actives': ['Copper oxychloride or copper hydroxide (suppressive only)'],
                'caution': (
                    'No bactericide reliably cures established bacterial blight, so spraying '
                    'is often wasted money. Streptomycin and other antibiotics are banned or '
                    'restricted for crop use in many countries — do not use them without '
                    'checking what is approved locally.'
                ),
            },
            'prevention': [
                'Choose a resistant variety before the season starts — the decision that matters most',
                'Treat seed and source it certified',
                'Avoid nitrogen above recommended rates',
            ],
            'escalate_when': (
                'Seedlings are wilting (kresek), or lesions appear on more than about a '
                'fifth of leaf area, or the crop was recently hit by a storm and is '
                'spreading fast. Call an extension officer — variety replacement for next '
                'season is the real decision.'
            ),
            'links': [
                {'label': 'IRRI fact sheet — Bacterial blight',
                 'url': f'{RKB}/bacterial-blight?category_id=326'},
            ],
        },
    },
    'brown_spot': {
        'name': 'Brown Spot',
        'pathogen': 'Fungal (Bipolaris oryzae)',
        'severity': 'Medium',
        'description': 'Fungal disease causing brown spots with grey centres on leaves.',
        'recommendation': 'Usually a soil-fertility problem before it is a fungus problem. Correct potassium and silicon, use treated seed; fungicide only if severe.',
        'icon': '\U0001f7e4',
        'care': {
            'summary': (
                'Treat this as a soil problem first. Brown spot is the classic disease of '
                'poor, exhausted or drought-stressed fields, and a fungicide sprayed onto a '
                'potassium deficiency fixes nothing.'
            ),
            'spreads_by': 'Infected seed and airborne spores from crop debris',
            'favoured_by': 'Potassium or silicon deficiency, unflooded or drought-stressed soil, high humidity, leaf wetness over 8 hours',
            'symptoms': [
                'Oval or circular brown spots roughly the size of a sesame seed',
                'Grey or whitish centre with a reddish-brown margin',
                'Spots scattered over the whole leaf blade, not confined to the margin',
                'On severe infections, spots also appear on glumes and discolour the grain',
            ],
            'first_steps': [
                'Check the soil before reaching for a spray — potassium deficiency is the most common underlying cause',
                'Correct water stress; brown spot is much worse on fields that dry out',
                'Look at whether it is patchy across the field, which points to soil rather than weather',
            ],
            'cultural': [
                'Apply potassium to soil-test recommendation — the highest-value action for this disease',
                'Add silicon where soils are depleted; rice straw returned to the field is a cheap source',
                'Use certified seed, or hot-water treat seed before sowing',
                'Keep the field adequately watered rather than letting it dry out',
                'Burn or bury infected stubble and remove volunteer plants',
            ],
            'chemical': {
                'actives': ['Propiconazole', 'Azoxystrobin', 'Mancozeb', 'Iprodione'],
                'caution': (
                    'Worth spraying only when infection is severe or reaching the panicle. '
                    'Confirm which of these is registered for rice in your country and at '
                    'what growth stage — availability differs widely.'
                ),
            },
            'prevention': [
                'Balanced fertiliser with adequate potassium, decided by soil test rather than habit',
                'Seed treatment before sowing',
                'Avoid planting on fields with known fertility problems without correcting them first',
            ],
            'escalate_when': (
                'Spots reach the panicle or grain, or the crop is stunted overall — that '
                'indicates a fertility problem large enough to need a soil test.'
            ),
            'links': [
                {'label': 'IRRI fact sheet — Brown spot', 'url': f'{RKB}/brown-spot'},
            ],
        },
    },
    'leaf_blast': {
        'name': 'Leaf Blast',
        'pathogen': 'Fungal (Magnaporthe oryzae)',
        'severity': 'High',
        'description': 'Fungal disease causing diamond-shaped lesions with grey centres and brown margins.',
        'recommendation': 'Stop nitrogen, keep the field flooded, use resistant varieties. Protect the neck at booting — neck blast is what destroys yield.',
        'icon': '\U0001f4a5',
        'care': {
            'summary': (
                'The most destructive rice disease worldwide. What you see on the leaf is the '
                'warning, not the damage — the yield loss comes later if the fungus reaches the '
                'neck of the panicle. The window to act is before heading, not after.'
            ),
            'spreads_by': 'Airborne spores, released at night and in high humidity; also seed-borne',
            'favoured_by': 'High nitrogen, long dew periods, cool nights with warm days, upland or aerobic soil, dense canopy',
            'symptoms': [
                'Diamond or spindle-shaped lesions, pointed at both ends',
                'Grey or white centre with a brown to reddish margin',
                'Lesions enlarge and merge, killing whole leaves',
                'Later: dark lesions at the node, collar, or the neck just below the panicle',
            ],
            'first_steps': [
                'Stop nitrogen application now. Excess nitrogen is the strongest driver of blast',
                'Keep continuous flood water on the field — flooding measurably suppresses blast',
                'Mark the calendar for booting stage. Neck blast is prevented there, and cannot be fixed afterwards',
                'Check the collar and nodes, not just the leaf blade',
            ],
            'cultural': [
                'Plant resistant varieties, and rotate between different resistance sources — blast overcomes single-gene resistance quickly',
                'Split nitrogen into several small applications rather than one large one',
                'Maintain standing water; avoid intermittent drying during high-risk periods',
                'Apply silicon where soils are depleted — it thickens the leaf surface the fungus must penetrate',
                'Avoid dense seeding, which holds humidity in the canopy',
            ],
            'chemical': {
                'actives': ['Tricyclazole', 'Isoprothiolane', 'Azoxystrobin', 'Propiconazole', 'Carpropamid'],
                'caution': (
                    'Timing decides whether this works at all. A protective spray at late '
                    'booting and again at heading is what prevents neck blast; the same '
                    'product applied after symptoms appear on the neck does nothing. '
                    'Tricyclazole is banned or restricted in the EU and some export markets — '
                    'check before using it on a crop intended for export.'
                ),
            },
            'prevention': [
                'Resistant variety chosen for your region’s prevailing races',
                'Nitrogen at recommended rate, split, guided by a leaf colour chart',
                'Seed treatment and clean, buried stubble',
            ],
            'escalate_when': (
                'Any lesion on the collar, node or neck, or more than about 10% leaf area '
                'affected at booting stage. Both mean the crop is at real yield risk and '
                'the spray decision should be made with an extension officer this week.'
            ),
            'links': [
                {'label': 'IRRI fact sheet — Blast (leaf and collar)', 'url': f'{RKB}/blast-leaf-collar'},
                {'label': 'IRRI fact sheet — Blast (node and neck)', 'url': f'{RKB}/blast-node-neck'},
            ],
        },
    },
    'leaf_scald': {
        'name': 'Leaf Scald',
        'pathogen': 'Fungal (Microdochium oryzae)',
        'severity': 'Medium',
        'description': 'Fungal disease causing zonate lesions with wavy margins, usually starting at the leaf tip.',
        'recommendation': 'Reduce nitrogen, open up the canopy, remove infected debris. Rarely worth a fungicide on its own.',
        'icon': '\U0001f525',
        'care': {
            'summary': (
                'Usually a secondary disease and rarely worth spraying by itself. It tends to '
                'appear on crops that are over-fertilised and too densely planted, so it is '
                'most useful read as a signal about nitrogen and spacing.'
            ),
            'spreads_by': 'Spores from infected debris and seed, spread by rain splash',
            'favoured_by': 'High nitrogen, dense canopy, high humidity, older leaves late in the season',
            'symptoms': [
                'Lesions start at the leaf tip or along the margin',
                'Alternating light and dark bands, giving a zonate or banded look',
                'Wavy yellow-brown margin around a bleached centre',
                'Overall appearance as if the leaf edge had been scalded',
            ],
            'first_steps': [
                'Reduce or stop nitrogen — this is the usual driver',
                'Check plant spacing; a canopy that never dries favours the fungus',
                'Confirm it is not tip burn from salinity or fertiliser scorch, which looks similar but has no banding',
            ],
            'cultural': [
                'Avoid nitrogen above the recommended rate',
                'Widen spacing so air moves through the canopy',
                'Remove and destroy infected straw and stubble after harvest',
                'Use clean seed',
            ],
            'chemical': {
                'actives': ['Propiconazole', 'Mancozeb', 'Benomyl'],
                'caution': (
                    'Seldom economic for leaf scald alone. If you are already spraying for '
                    'blast or brown spot, a product with activity on both is the sensible '
                    'choice rather than a separate application.'
                ),
            },
            'prevention': [
                'Balanced nitrogen and sensible spacing',
                'Clean seed and residue management between seasons',
            ],
            'escalate_when': (
                'It reaches the flag leaf, or appears alongside another disease — the '
                'combination matters more than leaf scald on its own.'
            ),
            'reliability_note': (
                'The model is less reliable for leaf scald than for the other diseases '
                'here. Confirm it against the fact sheet or with an agronomist before '
                'acting.'
            ),
            'links': [
                {'label': 'IRRI fact sheet — Leaf scald', 'url': f'{RKB}/leaf-scald'},
            ],
        },
    },
    'narrow_brown_spot': {
        'name': 'Narrow Brown Spot',
        'pathogen': 'Fungal (Cercospora janseana)',
        'severity': 'Low',
        'description': 'Fungal disease causing narrow, linear brown lesions running along the leaf.',
        'recommendation': 'Correct potassium, choose resistant varieties. A boot-stage fungicide only if it is reaching the sheath and neck.',
        'icon': '\U0001f4cf',
        'care': {
            'summary': (
                'A late-season disease that is usually mild, but matters when it moves onto '
                'the sheath and neck, where it can cause lodging and premature ripening. '
                'Like brown spot, it is closely tied to potassium supply.'
            ),
            'spreads_by': 'Airborne spores from infected debris; also seed-borne',
            'favoured_by': 'Potassium deficiency, susceptible varieties, late crop stages, warm humid weather',
            'symptoms': [
                'Short, narrow, linear lesions running parallel to the veins',
                'Reddish-brown to dark brown, much narrower than brown spot',
                'Mostly on older leaves late in the season',
                'In severe cases also on the sheath, neck and glumes',
            ],
            'first_steps': [
                'Check potassium status — deficiency is the most common predisposing factor',
                'Note the growth stage. Late-season leaf infection alone rarely justifies action',
                'Inspect the sheath and neck, which is where this disease actually costs yield',
            ],
            'cultural': [
                'Apply potassium to soil-test recommendation',
                'Choose resistant varieties where narrow brown spot is a recurring problem',
                'Consider earlier-maturing varieties, which escape the late-season peak',
                'Remove infected residue after harvest',
            ],
            'chemical': {
                'actives': ['Propiconazole', 'Azoxystrobin'],
                'caution': (
                    'Only worth it at boot stage and only when the disease is already '
                    'reaching the sheath. A spray on late-season leaf symptoms alone will '
                    'not pay for itself.'
                ),
            },
            'prevention': [
                'Potassium at the right rate, decided by soil test',
                'Resistant variety selection',
                'Residue management between seasons',
            ],
            'escalate_when': (
                'Lesions appear on the sheath or neck, or plants begin to lodge or ripen '
                'unevenly.'
            ),
            # Presence of this key is what triggers the caution strip in the UI.
            # The strip uses its own fixed wording; this text is what the disease
            # library shows, so it stays short and free of model internals.
            'reliability_note': (
                'The model is less reliable for narrow brown spot than for the other '
                'diseases here. Confirm it against the fact sheet or with an agronomist '
                'before acting.'
            ),
            'links': [
                {'label': 'IRRI fact sheet — Narrow brown spot', 'url': f'{RKB}/narrow-brown-spot'},
            ],
        },
    },
}

# ============================================
# MODEL ARCHITECTURES
# ============================================
# M-08: `pretrained=` was deprecated in torchvision 0.13 and is scheduled for
# removal. `weights=None` is the supported spelling.
#
# B-06: the bare `except:` fallbacks that silently substituted EfficientNet-B4
# for ViT and ResNeXt-50 for ConvNeXt are gone. They swallowed KeyboardInterrupt
# and SystemExit too, and produced a confusing state-dict error far from the
# real cause. If an architecture cannot be built that is fatal, and it should
# say so.


def create_efficientnet_b3(num_classes):
    model = models.efficientnet_b3(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def create_densenet121(num_classes):
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def create_mobilenetv3_large(num_classes):
    model = models.mobilenet_v3_large(weights=None)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.Hardswish(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


def create_vit_base(num_classes):
    model = models.vit_b_16(weights=None)
    num_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def create_convnext_tiny(num_classes):
    model = models.convnext_tiny(weights=None)
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.GELU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


# ============================================
# PREDICTOR
# ============================================

class TwoStageEnsemblePredictor:
    """Two-stage ensemble.

    Stage 1 (7 classes) always runs. Stage 2 (5 disease classes) runs only when
    Stage 1 predicts a disease AND is below `confidence_threshold`.
    """

    def __init__(self, stage1_models, stage2_models, device,
                 confidence_threshold=CONFIDENCE_THRESHOLD,
                 stage1_weights=None, stage2_weights=None):
        self.stage1_models = stage1_models
        self.stage2_models = stage2_models
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.stage1_weights = stage1_weights
        self.stage2_weights = stage2_weights

        for model in list(stage1_models.values()) + list(stage2_models.values()):
            model.eval()

    @staticmethod
    def _combine(probs, weights):
        """M-14: weighted mean over ensemble members instead of a plain mean.

        Members differ by ~3 accuracy points; uniform averaging gives the
        weakest model an equal vote. Weights are the checkpoints' validation
        accuracies, normalised. Falls back to a plain mean if unavailable.
        """
        stacked = np.concatenate(probs, axis=0)  # (n_models, n_classes)
        if weights is None or len(weights) != stacked.shape[0]:
            return stacked.mean(axis=0)
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        return (stacked * w[:, None]).sum(axis=0)

    def _run(self, models_dict, image_tensor, weights):
        probs = []
        with torch.no_grad():
            for model in models_dict.values():
                output = model(image_tensor)
                probs.append(torch.softmax(output, dim=1).cpu().numpy())
        avg = self._combine(probs, weights)
        idx = int(np.argmax(avg))
        return idx, float(avg[idx]), avg

    def predict(self, image_tensor):
        s1_idx, s1_conf, s1_probs = self._run(
            self.stage1_models, image_tensor, self.stage1_weights)
        s1_label = CLASS_NAMES_STAGE1[s1_idx]

        result = {
            'stage1': {
                'class': s1_label,
                'confidence': s1_conf,
                'probabilities': {
                    CLASS_NAMES_STAGE1[i]: float(s1_probs[i])
                    for i in range(len(CLASS_NAMES_STAGE1))
                },
            },
            'stage2_executed': False,
            'abstained': False,
        }

        # Healthy / not-a-rice-leaf resolve at Stage 1.
        if s1_label not in DISEASE_CLASSES:
            result['final_diagnosis'] = s1_label
            result['final_confidence'] = s1_conf
            return result

        # Confident enough at Stage 1 - no refinement needed.
        if s1_conf >= self.confidence_threshold:
            result['final_diagnosis'] = s1_label
            result['final_confidence'] = s1_conf
            return result

        s2_idx, s2_conf, s2_probs = self._run(
            self.stage2_models, image_tensor, self.stage2_weights)
        s2_label = CLASS_NAMES_STAGE2[s2_idx]

        result['stage2'] = {
            'disease_type': s2_label,
            'confidence': s2_conf,
            'probabilities': {
                CLASS_NAMES_STAGE2[i]: float(s2_probs[i])
                for i in range(len(CLASS_NAMES_STAGE2))
            },
        }
        result['stage2_executed'] = True

        # M-13: Stage 2's output space excludes healthy and not_rice_leaf, so it
        # cannot correct a Stage 1 false positive - it can only relabel it. And
        # Stage 2 runs precisely when Stage 1 was unsure, i.e. when Stage 1 is
        # most likely wrong. Rather than force a disease label through, abstain
        # when Stage 2 is also unconvinced.
        if s2_conf < STAGE2_ABSTAIN_THRESHOLD:
            result['final_diagnosis'] = 'uncertain'
            result['final_confidence'] = s2_conf
            result['abstained'] = True
            return result

        # M-05: v1 reported stage1_conf * stage2_conf as "confidence". Those two
        # are not independent (same image, correlated models) and neither is
        # calibrated, so the product is not a joint probability - and it is
        # lowest exactly when Stage 2 was most useful. Report Stage 2's own
        # confidence and expose the Stage 1 value separately.
        result['final_diagnosis'] = s2_label
        result['final_confidence'] = s2_conf
        return result


# ============================================
# MODEL LOADING
# ============================================

_predictor = None
_predictor_error = None
_load_lock = threading.Lock()


def _resolve_checkpoint(folder, name):
    """Pick one checkpoint deterministically.

    v1 used `os.listdir(...)[0]`. os.listdir() returns arbitrary filesystem
    order, and this directory holds two training runs per model, so every model
    silently loaded a stale checkpoint - and a different one on a different
    machine. Sorting by the trailing YYYYMMDD_HHMMSS timestamp fixes that.

    v1 also wrapped the load in `if matching_files:`, so a missing file left the
    model at its random initialisation while it kept voting in the ensemble.
    That path now raises.
    """
    candidates = sorted(
        f for f in os.listdir(folder)
        if f.startswith(name + '_') and f.endswith('.pth')
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint for '{name}' in {folder}. "
            f"Expected {name}_YYYYMMDD_HHMMSS.pth"
        )
    chosen = candidates[-1]
    if len(candidates) > 1:
        logger.warning(
            f"{name}: {len(candidates)} checkpoints present, using newest "
            f"({chosen}); ignoring {candidates[:-1]}"
        )
    return os.path.join(folder, chosen)


def load_models():
    """Build and load both ensembles. Raises on any failure."""
    logger.info("Loading models...")

    # B-12: resolve relative to this file, not the process working directory.
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')

    stage1 = {
        'efficientnet_b3': create_efficientnet_b3(7).to(device),
        'densenet121': create_densenet121(7).to(device),
        'mobilenetv3': create_mobilenetv3_large(7).to(device),
    }
    stage2 = {
        'vit_base': create_vit_base(5).to(device),
        'convnext_tiny': create_convnext_tiny(5).to(device),
    }

    # Sandbox override, for evaluating candidate weights BEFORE promoting them.
    # A v2 checkpoint should be provable through the real pipeline while
    # saved_models/ still holds the shipping weights, so a bad candidate is
    # never one forgotten `git add` away from production. Absent these vars the
    # behaviour is exactly as before.
    overrides = {
        'stage1_models': os.environ.get('STAGE1_MODEL_DIR'),
        'stage2_models': os.environ.get('STAGE2_MODEL_DIR'),
    }

    accuracies = {}
    for subdir, group in (('stage1_models', stage1), ('stage2_models', stage2)):
        override = overrides[subdir]
        if override:
            folder = os.path.abspath(override)
            # Loud on purpose: a container that silently served sandbox weights
            # would be indistinguishable from one serving the real thing.
            logger.warning(
                f"{subdir}: OVERRIDE ACTIVE - loading from {folder} instead of "
                f"saved_models/{subdir}. This is for evaluation only."
            )
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Override dir does not exist: {folder}")
        else:
            folder = os.path.join(model_dir, subdir)
        for name, model in group.items():
            path = _resolve_checkpoint(folder, name)
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            acc = float(checkpoint.get('best_val_acc', 0.0) or 0.0)
            accuracies[name] = acc
            # A-07: accuracies are only comparable within a stage if they were
            # measured on the same set. Stage 1 checkpoints (v2) carry an
            # eval_set tag; Stage 2 checkpoints (v1) predate it.
            # eval_set was added to the training scripts after the v2 Stage 1
            # run, so those checkpoints carry version='v2' but no eval_set tag.
            # Fall back to version rather than mislabelling them "pre-v2".
            eval_set = checkpoint.get('eval_set')
            if not eval_set:
                ver = checkpoint.get('version')
                eval_set = (f"{ver} run, eval set not recorded" if ver
                            else "v1 checkpoint, eval set not recorded")
            logger.info(
                f"Loaded {name} from {os.path.basename(path)} "
                f"(val acc {acc:.2f}% on {eval_set})"
            )

    # Weighting is applied WITHIN a stage only, never across stages. Stage 1
    # accuracies come from the v2 validation split; Stage 2 accuracies are v1
    # numbers on the old split. Comparing them would be meaningless, and the
    # eval_set tag logged above makes that visible (A-07).
    #
    # Historical note: v1 checkpoints stored the best epoch's accuracy next to
    # the FINAL epoch's weights, because the training loop used a shallow
    # state_dict().copy(). v2 checkpoints use copy.deepcopy and do not have
    # this problem.
    s1_w = [accuracies[n] for n in stage1] if USE_ACCURACY_WEIGHTED_ENSEMBLE else None
    s2_w = [accuracies[n] for n in stage2] if USE_ACCURACY_WEIGHTED_ENSEMBLE else None

    logger.info("All models loaded successfully")
    return stage1, stage2, s1_w, s2_w


def get_predictor(force_reload=False):
    """B-10: lazy, retryable model loading.

    v1 loaded at import time and set `predictor = None` on failure, leaving the
    process serving 500s forever with no way to recover short of a restart.
    """
    global _predictor, _predictor_error
    if _predictor is not None and not force_reload:
        return _predictor

    with _load_lock:
        if _predictor is not None and not force_reload:
            return _predictor
        try:
            s1, s2, s1_w, s2_w = load_models()
            _predictor = TwoStageEnsemblePredictor(
                s1, s2, device, CONFIDENCE_THRESHOLD,
                stage1_weights=s1_w, stage2_weights=s2_w,
            )
            _predictor_error = None
            logger.info("Predictor initialised")
        except Exception as exc:
            _predictor_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Model loading failed")
            _predictor = None
    return _predictor


# ============================================
# PREPROCESSING
# ============================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image.load()                      # force decode inside our try/except
    image = image.convert('RGB')
    return transform(image).unsqueeze(0).to(device), image


def _runner_up(result):
    """A-12: the second-place class, and how far behind it is.

    A prediction can clear the abstain threshold and still be close to a tie.
    The margin is already computed in order to pick a winner and was then
    thrown away, so the response could not distinguish "leaf scald, clearly"
    from "leaf scald by 11 points over leaf blast". Those warrant different
    confidence from the reader, and now the UI can tell them apart.
    """
    src = result['stage2'] if result['stage2_executed'] else result['stage1']
    probs = src.get('probabilities') or {}
    if len(probs) < 2:
        return None
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    (_, top_p), (second, second_p) = ranked[0], ranked[1]
    return {
        'label': second.replace('_', ' ').title(),
        'confidence': f"{second_p * 100:.2f}%",
        'margin_points': round((top_p - second_p) * 100, 2),
    }


# ============================================
# RATE LIMITING  (B-03)
# ============================================

_hits = {}
_hits_lock = threading.Lock()


def rate_limited(remote_addr):
    now = datetime.now().timestamp()
    cutoff = now - RATE_LIMIT_WINDOW_S
    with _hits_lock:
        bucket = [t for t in _hits.get(remote_addr, []) if t > cutoff]
        if len(bucket) >= RATE_LIMIT_REQUESTS:
            _hits[remote_addr] = bucket
            return True
        bucket.append(now)
        _hits[remote_addr] = bucket
        if len(_hits) > 2048:                      # bound memory
            for k in [k for k, v in _hits.items() if not v or max(v) < cutoff]:
                _hits.pop(k, None)
    return False


# ============================================
# METRICS  (B-16)
# ============================================

_metrics = {
    'predictions_total': 0,
    'predictions_failed': 0,
    'stage2_invocations': 0,
    'abstentions': 0,
    'rate_limited': 0,
    'by_class': Counter(),
    'started_at': datetime.now().isoformat(),
}
_metrics_lock = threading.Lock()


# ============================================
# ROUTES
# ============================================

@app.before_request
def _assign_request_id():
    g.request_id = uuid.uuid4().hex[:8]


@app.errorhandler(413)
def _too_large(_):
    return jsonify({
        'error': f'File too large. Maximum {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.',
        'ref': getattr(g, 'request_id', None),
    }), 413


@app.route('/')
def index():
    """Render the page with the disease library baked in.

    The reference section is generated from DISEASE_INFO rather than written
    into the template, so the guidance a user browses and the guidance returned
    by /predict cannot drift apart — there is only one copy of it.
    """
    library = [
        {'key': k, **v} for k, v in DISEASE_INFO.items()
        if k in DISEASE_CLASSES
    ]
    return render_template(
        'index.html',
        library=library,
        support=SUPPORT_RESOURCES,
        guidance=INPUT_GUIDANCE,
    )


@app.route('/predict', methods=['POST'])
def predict():
    rid = g.request_id

    if rate_limited(request.remote_addr or 'unknown'):
        with _metrics_lock:
            _metrics['rate_limited'] += 1
        logger.warning(f"[{rid}] rate limited {request.remote_addr}")
        return jsonify({
            'error': f'Too many requests. Limit is {RATE_LIMIT_REQUESTS} '
                     f'per {RATE_LIMIT_WINDOW_S}s.',
            'ref': rid,
        }), 429

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded', 'ref': rid}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected', 'ref': rid}), 400

    predictor = get_predictor()
    if predictor is None:
        logger.error(f"[{rid}] predictor unavailable: {_predictor_error}")
        return jsonify({'error': 'Model not available', 'ref': rid}), 503

    try:
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({'error': 'Empty file', 'ref': rid}), 400
        image_tensor, _ = preprocess_image(image_bytes)
    except Exception:
        # B-01: never echo str(e) - PIL and torch messages routinely contain
        # absolute filesystem paths and library internals.
        logger.exception(f"[{rid}] image decode failed")
        with _metrics_lock:
            _metrics['predictions_failed'] += 1
        return jsonify({
            'error': 'Could not read that image. Please upload a valid JPG or PNG.',
            'ref': rid,
        }), 400

    try:
        result = predictor.predict(image_tensor)
    except Exception:
        logger.exception(f"[{rid}] inference failed")
        with _metrics_lock:
            _metrics['predictions_failed'] += 1
        return jsonify({'error': 'Analysis failed', 'ref': rid}), 500

    diagnosis = result['final_diagnosis']
    info = DISEASE_INFO.get(diagnosis, DISEASE_INFO['uncertain'])

    with _metrics_lock:
        _metrics['predictions_total'] += 1
        _metrics['by_class'][diagnosis] += 1
        if result['stage2_executed']:
            _metrics['stage2_invocations'] += 1
        if result['abstained']:
            _metrics['abstentions'] += 1

    response = {
        'success': True,
        'diagnosis': info['name'],
        'confidence': f"{result['final_confidence'] * 100:.2f}%",
        'confidence_value': round(result['final_confidence'], 4),
        'pathogen': info['pathogen'],
        'severity': info['severity'],
        'description': info['description'],
        'recommendation': info['recommendation'],
        'icon': info['icon'],
        'abstained': result['abstained'],
        'stage2_used': result['stage2_executed'],
        'input_guidance': INPUT_GUIDANCE,
        # Structured management guidance for the results panel. `recommendation`
        # above is kept as the one-line form so existing consumers (and
        # diagnostics/test_pipeline.py) do not break.
        'care': info.get('care'),
        'support': SUPPORT_RESOURCES,
        # A-12: the two stages can pick different diseases, and until now the
        # response gave no way to know. Surfaced so the UI can hedge instead of
        # presenting a contradicted label as settled fact.
        'stages_agree': (
            None if not result['stage2_executed']
            else result['stage1']['class'] == result['stage2']['disease_type']
        ),
        'runner_up': _runner_up(result),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ref': rid,
        'details': {
            'stage1_prediction': result['stage1']['class'].replace('_', ' ').title(),
            'stage1_confidence': f"{result['stage1']['confidence'] * 100:.2f}%",
            'stage2_prediction': (
                result.get('stage2', {}).get('disease_type', 'N/A')
                .replace('_', ' ').title()
                if result['stage2_executed'] else 'N/A'
            ),
            'stage2_confidence': (
                f"{result['stage2']['confidence'] * 100:.2f}%"
                if result['stage2_executed'] else 'N/A'
            ),
            'models_used': 5 if result['stage2_executed'] else 3,
        },
    }

    logger.info(
        f"[{rid}] {diagnosis} {result['final_confidence'] * 100:.2f}% "
        f"stage2={result['stage2_executed']} abstained={result['abstained']}"
    )
    return jsonify(response)


@app.route('/health')
def health():
    """B-05: v1 always returned 200, so an uptime monitor reported a healthy
    service while every /predict call failed. Readiness now reflects reality.
    """
    ready = get_predictor() is not None
    payload = {
        'status': 'healthy' if ready else 'degraded',
        'model_loaded': ready,
        'device': str(device),
        'timestamp': datetime.now().isoformat(),
    }
    if not ready:
        payload['detail'] = _predictor_error
    return jsonify(payload), (200 if ready else 503)


@app.route('/live')
def live():
    """Liveness, separate from readiness: the process is up and serving."""
    return jsonify({'status': 'alive'}), 200


@app.route('/metrics')
def metrics():
    with _metrics_lock:
        payload = dict(_metrics)
        payload['by_class'] = dict(_metrics['by_class'])
    return jsonify(payload)


if __name__ == '__main__':
    get_predictor()
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
