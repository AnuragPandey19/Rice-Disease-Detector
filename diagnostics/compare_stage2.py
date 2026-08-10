"""
A/B the candidate Stage 2 against the shipping Stage 2, before promoting anything.

WHY THIS EXISTS
---------------
v2/scripts/09_train_stage2.py reports a held-out test number (95.29%, 86.67% on
field photographs). That is the right number to trust for the model in
isolation, but it says nothing about the SYSTEM: Stage 1 routing, the abstain
rule, and the final response are all downstream of it.

This runs the real pipeline twice over the same images - once with the shipping
Stage 2 weights, once with the v2 candidates - and diffs the two. Stage 1 is
held constant in both runs, so any difference is attributable to Stage 2 alone.

Nothing is copied or promoted. The candidate weights are reached through the
STAGE2_MODEL_DIR override in app.py; saved_models/ is never written to.

USAGE
-----
    python diagnostics/compare_stage2.py                    # diagnostics/images/
    python diagnostics/compare_stage2.py path/to/photo.jpg
    python diagnostics/compare_stage2.py --probs            # class distributions

READING THE OUTPUT
------------------
Only images that reach Stage 2 can differ. If Stage 1 is confident (>= the
routing threshold) or returns a non-disease class, Stage 2 never runs and both
columns will be identical by construction - that is not evidence of anything.
The line to look for is "reached Stage 2".
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

CANDIDATE_DIR = PROJECT_ROOT / 'v2' / 'models' / 'stage2_models'
IMG_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def collect(argv_images):
    if argv_images:
        return [Path(p) for p in argv_images if Path(p).exists()]
    d = PROJECT_ROOT / 'diagnostics' / 'images'
    return sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT) \
        if d.is_dir() else []


def run_all(app, paths):
    """Predict every image with whatever predictor is currently loaded."""
    predictor = app.get_predictor(force_reload=True)
    if predictor is None:
        sys.exit(f"MODEL FAILED TO LOAD: {app._predictor_error}")
    out = {}
    for p in paths:
        tensor, _ = app.preprocess_image(p.read_bytes())
        r = predictor.predict(tensor)
        out[p.name] = {
            'final': r['final_diagnosis'],
            'conf': r['final_confidence'],
            's1': r['stage1']['class'],
            's1_conf': r['stage1']['confidence'],
            's2_ran': r['stage2_executed'],
            's2': r['stage2']['disease_type'] if r['stage2_executed'] else None,
            's2_conf': r['stage2']['confidence'] if r['stage2_executed'] else None,
            's2_probs': (r['stage2']['probabilities']
                         if r['stage2_executed'] else None),
            'abstained': r['abstained'],
        }
    return out


def label(app, rec):
    return app.DISEASE_INFO.get(rec['final'],
                                app.DISEASE_INFO['uncertain'])['name']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='*', help='defaults to diagnostics/images/')
    ap.add_argument('--probs', action='store_true',
                    help='print Stage 2 class distributions for both runs')
    args = ap.parse_args()

    paths = collect(args.images)
    if not paths:
        sys.exit("No images found. Put some in diagnostics/images/ or pass paths.")
    if not CANDIDATE_DIR.is_dir():
        sys.exit(f"No candidate weights at {CANDIDATE_DIR}. "
                 f"Run v2/scripts/09_train_stage2.py first.")

    # ASCII only from here down. PowerShell renders this console in cp1252 and
    # mangled the em dashes in the Stage 2 training log into 'u' characters;
    # this output is meant to be copied and pasted, so it must survive that.
    print("=" * 78)
    print("STAGE 2 A/B  --  shipping vs candidate, through the real pipeline")
    print("=" * 78)
    print(f"  images     : {len(paths)}")
    print("  stage 1    : unchanged in both runs (saved_models/stage1_models)")
    print(f"  candidate  : {CANDIDATE_DIR.relative_to(PROJECT_ROOT)}")
    print("  promotion  : none - saved_models/ is not written to")
    print()

    os.environ.pop('STAGE2_MODEL_DIR', None)
    import app

    print("-- run A: shipping Stage 2 " + "-" * 50)
    before = run_all(app, paths)

    print("\n-- run B: candidate Stage 2 " + "-" * 49)
    os.environ['STAGE2_MODEL_DIR'] = str(CANDIDATE_DIR)
    after = run_all(app, paths)
    os.environ.pop('STAGE2_MODEL_DIR', None)

    print()
    print("=" * 78)
    print("RESULT  (paste this)")
    print("=" * 78)

    changed, reached = 0, 0
    for p in paths:
        n = p.name
        a, b = before[n], after[n]
        if b['s2_ran']:
            reached += 1
        diff = (a['final'] != b['final']) or (a['abstained'] != b['abstained'])
        changed += diff

        print(f"\n  {n}")
        print(f"    stage 1        : {a['s1']} @ {a['s1_conf'] * 100:.2f}%"
              f"   ({'routed to Stage 2' if a['s2_ran'] else 'Stage 2 skipped'})")
        if not a['s2_ran'] and not b['s2_ran']:
            print(f"    verdict        : {label(app, a)} - Stage 2 not involved")
            continue
        print(f"    shipping       : {label(app, a):24s} {a['conf'] * 100:6.2f}%"
              f"{'   ABSTAINED' if a['abstained'] else ''}")
        print(f"    candidate      : {label(app, b):24s} {b['conf'] * 100:6.2f}%"
              f"{'   ABSTAINED' if b['abstained'] else ''}")
        if a['s2'] and b['s2']:
            delta = (b['s2_conf'] - a['s2_conf']) * 100
            print(f"    stage 2 conf   : {a['s2_conf'] * 100:.2f}% -> "
                  f"{b['s2_conf'] * 100:.2f}%  ({delta:+.2f} pts)")
        print(f"    changed        : {'YES' if diff else 'no'}")

        if args.probs and b['s2_probs']:
            print("    stage 2 distribution (shipping | candidate):")
            for cls in sorted(b['s2_probs'], key=lambda c: -b['s2_probs'][c]):
                pa = a['s2_probs'][cls] if a['s2_probs'] else float('nan')
                print(f"      {cls:22s} {pa * 100:6.2f}%  |  "
                      f"{b['s2_probs'][cls] * 100:6.2f}%")

    print()
    print("-" * 78)
    print(f"  reached Stage 2          : {reached}/{len(paths)}"
          "   (only these can differ)")
    print(f"  verdict changed          : {changed}/{len(paths)}")
    print(f"  rejected as not-a-leaf   : "
          f"{sum(1 for v in after.values() if v['final'] == 'not_rice_leaf')}"
          f"/{len(paths)}   (Stage 1's job - unchanged by this test)")
    print("-" * 78)
    if reached == 0:
        print("  Stage 1 answered every image on its own, so this run says nothing")
        print("  about Stage 2 either way. Test with images Stage 1 is unsure about.")

    out = PROJECT_ROOT / 'diagnostics' / 'stage2_ab_results.json'
    out.write_text(json.dumps({'shipping': before, 'candidate': after}, indent=2),
                   encoding='utf-8')
    print(f"\nsaved: {out.relative_to(PROJECT_ROOT)}")


if __name__ == '__main__':
    main()
