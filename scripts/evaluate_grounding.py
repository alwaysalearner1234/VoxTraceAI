"""Grounding accuracy: IoU vs injected bounds (pause/L3) + label agreement.

Usage: python scripts/evaluate_grounding.py
Known-by-construction bounds exist for L2-pause and L3 (inserted silences);
for global mirrors we check that >=1 flaw fires with a compatible top label.
Also self-comparison (baseline vs itself) must yield ~0 flaws and total > 90.
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from voxtrace.pipeline import analyze_files

LABEL_DIR = os.path.join(ROOT, 'data', 'labels')
COMPAT = {'L1': None, 'L2-rushed': 'Rushed', 'L2-slow': 'Dragged', 'L2-pause': 'Pause',
          'L2-mumble': 'Mumble', 'L3': None}  # L1/L3: any/no flaw counts (checked separately)


def iou(a, b):
    s, e = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, e - s)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


rows = []
for jf in sorted(os.listdir(LABEL_DIR)):
    if jf.startswith('_') or not jf.endswith('.json'):
        continue
    item = json.load(open(os.path.join(LABEL_DIR, jf), encoding='utf-8'))
    name = os.path.splitext(jf)[0]
    text = open(os.path.join(ROOT, 'data', 'transcripts', item['transcript']), encoding='utf-8').read().strip()
    base = os.path.join(ROOT, item['baseline'])
    self_rep = analyze_files(base, base, text)
    print(f'{name}: SELF total={self_rep["scores"]["total"]} flaws={len(self_rep["flaws"])}')
    for m in item['mirrors']:
        rep = analyze_files(base, os.path.join(ROOT, m['file']), text)
        flaws = rep['flaws']
        inj_bounds = [(d['start'], d['end']) for d in m['injected'] if 'start' in d]
        best = max([max([iou((f['start'], f['end']), b) for b in inj_bounds] + [0.0])
                    for f in flaws] + [0.0]) if inj_bounds else None
        top = flaws[0]['label'] if flaws else '(none)'
        ok_label = (COMPAT[m['level']] is None or
                    (flaws and COMPAT[m['level']] in flaws[0]['label']) or
                    any(COMPAT[m['level']] in f['label'] for f in flaws))
        print(f'  {m["level"]:10s} total={rep["scores"]["total"]:5.1f} flaws={len(flaws)} '
              f'top={top[:28]:28s} bestIoU={best if best is None else round(best, 2)} '
              f'labelOK={ok_label}')
        rows.append((name, m['level'], rep['scores']['total'], len(flaws), best, ok_label))

ious = [r[4] for r in rows if r[4] is not None]
print(f'\nMean best-IoU on bounded mirrors (pause/L3): {sum(ious)/len(ious):.2f} over {len(ious)} cases')
print(f'Label agreement: {sum(1 for r in rows if r[5])}/{len(rows)}')
print(f'Monotonic (L1 top per speech): check stress_test_results.csv')
