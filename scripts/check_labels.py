"""Verify flaw labels match the injected flaw type (no inversions)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import json
from voxtrace.pipeline import analyze_files

for name in ['dream_excerpt', 'persuasive_original']:
    item = json.load(open(os.path.join(ROOT, 'data/labels', name + '.json'), encoding='utf-8'))
    text = open(os.path.join(ROOT, 'data/transcripts', item['transcript']), encoding='utf-8').read().strip()
    base = os.path.join(ROOT, item['baseline'])
    for lv in ['L2-rushed', 'L2-slow', 'L2-pause', 'L2-mumble']:
        m = [x for x in item['mirrors'] if x['level'] == lv][0]
        rep = analyze_files(base, os.path.join(ROOT, m['file']), text)
        labs = [(f['start'], f['end'], f['label']) for f in rep['flaws'][:4]]
        print(f"{name[:12]:12s} {lv:10s} total={rep['scores']['total']}")
        for l in labs:
            print('   ', l)
