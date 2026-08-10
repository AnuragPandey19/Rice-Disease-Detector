"""
Run the FULL prediction pipeline on local images and print what the website
would show — so results can be pasted as text instead of screenshotted.

WHY THIS IMPORTS app.py
-----------------------
It does not reimplement inference. It imports the real `app` module and calls
the same predictor the Flask endpoint uses, so the output cannot drift from what
the website actually returns. If app.py changes, this changes with it.

That matters: v2/scripts/06_test_real_photos.py only runs the Stage 1 ensemble.
It says nothing about Stage 2 routing, the abstain rule, or the final response.
This script exercises all of it — Stage 1 → routing decision → Stage 2 → abstain
check → the JSON the browser receives.

USAGE
-----
    python diagnostics/test_pipeline.py                    # everything in diagnostics/images/
    python diagnostics/test_pipeline.py path/to/photo.jpg  # specific files
    python diagnostics/test_pipeline.py --json             # raw API response
    python diagnostics/test_pipeline.py --probs            # full class distribution

The default output is written to be pasted into a chat or an issue.
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

IMG_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def collect(args_images):
    if args_images:
        return [Path(p) for p in args_images if Path(p).exists()]
    d = PROJECT_ROOT / 'diagnostics' / 'images'
    if not d.is_dir():
        return []
    return sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='*', help='defaults to diagnostics/images/')
    ap.add_argument('--json', action='store_true', help='print the raw API response')
    ap.add_argument('--probs', action='store_true', help='print the full class distribution')
    args = ap.parse_args()

    paths = collect(args.images)
    if not paths:
        sys.exit("No images found. Put some in diagnostics/images/ or pass paths.")

    print("=" * 78)
    print("FULL PIPELINE TEST — identical to what the website returns")
    print("=" * 78)
    print("Loading app.py and its models (first run takes a moment)...\n")

    try:
        import app  # noqa: E402  — the real application module
    except Exception as exc:
        sys.exit(f"Could not import app.py: {type(exc).__name__}: {exc}")

    predictor = app.get_predictor()
    if predictor is None:
        print("MODEL FAILED TO LOAD")
        print(f"  {app._predictor_error}")
        sys.exit(1)

    print(f"  device               : {app.device}")
    print(f"  routing threshold    : {app.CONFIDENCE_THRESHOLD}  "
          f"(Stage 2 runs below this)")
    print(f"  abstain threshold    : {app.STAGE2_ABSTAIN_THRESHOLD}  "
          f"(returns 'uncertain' below this)")
    print(f"  weighted ensemble    : {app.USE_ACCURACY_WEIGHTED_ENSEMBLE}")
    print(f"  images to test       : {len(paths)}")
    print()

    results = []
    for p in paths:
        try:
            image_bytes = p.read_bytes()
            tensor, _ = app.preprocess_image(image_bytes)
            raw = predictor.predict(tensor)
        except Exception as exc:
            print(f"{p.name}: FAILED — {type(exc).__name__}: {exc}\n")
            continue

        diagnosis = raw['final_diagnosis']
        info = app.DISEASE_INFO.get(diagnosis, app.DISEASE_INFO['uncertain'])

        print("─" * 78)
        print(f"  {p.name}   ({p.stat().st_size // 1024} KB)")
        print("─" * 78)
        print(f"  RESULT        : {info['name']}")
        print(f"  Confidence    : {raw['final_confidence'] * 100:.2f}%")
        if info['pathogen']:
            print(f"  Pathogen      : {info['pathogen']}")
        print(f"  Severity      : {info['severity']}")
        if raw['abstained']:
            print(f"  ABSTAINED     : yes — Stage 2 fell below "
                  f"{app.STAGE2_ABSTAIN_THRESHOLD}")
        print()
        print(f"  Stage 1       : {raw['stage1']['class']} "
              f"@ {raw['stage1']['confidence'] * 100:.2f}%")
        if raw['stage2_executed']:
            print(f"  Stage 2       : {raw['stage2']['disease_type']} "
                  f"@ {raw['stage2']['confidence'] * 100:.2f}%   (5 models used)")
        else:
            reason = ('not a disease class' if raw['stage1']['class']
                      not in app.DISEASE_CLASSES
                      else f"Stage 1 confident (>= {app.CONFIDENCE_THRESHOLD})")
            print(f"  Stage 2       : skipped — {reason}   (3 models used)")

        if args.probs:
            print()
            print("  Stage 1 distribution:")
            for cls, prob in sorted(raw['stage1']['probabilities'].items(),
                                    key=lambda kv: -kv[1]):
                bar = '█' * int(prob * 40)
                print(f"    {cls:22s} {prob * 100:6.2f}%  {bar}")
            if raw['stage2_executed']:
                print("  Stage 2 distribution:")
                for cls, prob in sorted(raw['stage2']['probabilities'].items(),
                                        key=lambda kv: -kv[1]):
                    bar = '█' * int(prob * 40)
                    print(f"    {cls:22s} {prob * 100:6.2f}%  {bar}")
        print()

        results.append({
            'image': p.name,
            'diagnosis': info['name'],
            'raw_class': diagnosis,
            'confidence': round(raw['final_confidence'] * 100, 2),
            'stage1': raw['stage1']['class'],
            'stage1_conf': round(raw['stage1']['confidence'] * 100, 2),
            'stage2_used': raw['stage2_executed'],
            'stage2': (raw.get('stage2', {}).get('disease_type')
                       if raw['stage2_executed'] else None),
            'stage2_conf': (round(raw['stage2']['confidence'] * 100, 2)
                            if raw['stage2_executed'] else None),
            'abstained': raw['abstained'],
        })

    # ---- compact summary, easy to paste ----
    print("=" * 78)
    print("SUMMARY  (paste this)")
    print("=" * 78)
    print(f"{'image':16s} {'result':24s} {'conf':>7s} {'stage2':>8s} {'abstain':>8s}")
    print("-" * 68)
    for r in results:
        print(f"{r['image']:16s} {r['diagnosis']:24s} {r['confidence']:6.2f}% "
              f"{'yes' if r['stage2_used'] else 'no':>8s} "
              f"{'YES' if r['abstained'] else '-':>8s}")
    print("-" * 68)

    rejected = [r for r in results if r['raw_class'] == 'not_rice_leaf']
    abstained = [r for r in results if r['abstained']]
    print(f"  rejected as not-a-rice-leaf : {len(rejected)}/{len(results)}"
          + (f"  ({', '.join(r['image'] for r in rejected)})" if rejected else ""))
    print(f"  abstained as uncertain      : {len(abstained)}/{len(results)}"
          + (f"  ({', '.join(r['image'] for r in abstained)})" if abstained else ""))
    print(f"  stage 2 invoked             : "
          f"{sum(1 for r in results if r['stage2_used'])}/{len(results)}")

    if args.json:
        print()
        print("=" * 78)
        print("RAW")
        print("=" * 78)
        print(json.dumps(results, indent=2))

    out = PROJECT_ROOT / 'diagnostics' / 'pipeline_results.json'
    out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f"\nsaved: {out.relative_to(PROJECT_ROOT)}")


if __name__ == '__main__':
    main()
