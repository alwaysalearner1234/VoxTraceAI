"""Calibrate grounding thresholds: peak/mean composite per mirror level."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import json
import numpy as np
from voxtrace.pipeline import analyze_files

LABEL_DIR = os.path.join(ROOT, 'data', 'labels')
for jf in sorted(os.listdir(LABEL_DIR)):
    if jf.startswith('_') or not jf.endswith('.json'):
        continue
    item = json.load(open(os.path.join(LABEL_DIR, jf), encoding='utf-8'))
    name = os.path.splitext(jf)[0]
    text = open(os.path.join(ROOT, 'data', 'transcripts', item['transcript']), encoding='utf-8').read().strip()
    base = os.path.join(ROOT, item['baseline'])
    for m in item['mirrors']:
        rep = analyze_files(base, os.path.join(ROOT, m['file']), text)
        comp = np.asarray(rep['features']['composite'])
        cov = sum(f['end'] - f['start'] for f in rep['flaws']) / max(rep['test']['duration'], 1e-6)
        print(f"{name[:12]:12s} {m['level']:10s} peak={comp.max():.2f} p90={np.percentile(comp,90):.2f} "
              f"mean={comp.mean():.2f} flaws={len(rep['flaws'])} cov={cov:.2f} total={rep['scores']['total']}")
