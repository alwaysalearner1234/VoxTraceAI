"""Stress test: score every mirror against its baseline, print + save a table.

Usage: python scripts/demo_stress_test.py
Expects data/ built by scripts/generate_dataset.py.
"""
import csv, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from voxtrace.pipeline import analyze_files

LABEL_DIR = os.path.join(ROOT, "data", "labels")
OUT_CSV = os.path.join(ROOT, "data", "labels", "stress_test_results.csv")


def main():
    rows = []
    for jf in sorted(os.listdir(LABEL_DIR)):
        if jf.startswith("_") or not jf.endswith(".json"):
            continue
        with open(os.path.join(LABEL_DIR, jf), encoding="utf-8") as f:
            item = json.load(f)
        name = os.path.splitext(jf)[0]
        with open(os.path.join(ROOT, "data", "transcripts", item["transcript"]), encoding="utf-8") as f:
            text = f.read().strip()
        base = os.path.join(ROOT, item["baseline"])
        for m in item["mirrors"]:
            rep = analyze_files(base, os.path.join(ROOT, m["file"]), text)
            rows.append({"speech": name, "level": m["level"],
                         "total": rep["scores"]["total"], "grade": rep["scores"]["grade"],
                         "pace": rep["scores"]["dimensions"]["pace"],
                         "variety": rep["scores"]["dimensions"]["variety"],
                         "pauses": rep["scores"]["dimensions"]["pauses"],
                         "clarity": rep["scores"]["dimensions"]["clarity"],
                         "n_flaws": len(rep["flaws"]),
                         "dtw_cost": rep["dtw_cost"]})
            print(f'{name:22s} {m["level"]:10s} total={rep["scores"]["total"]:5.1f} '
                  f'grade={rep["scores"]["grade"]} flaws={len(rep["flaws"])}')
    # monotonicity check: L1 should outscore L3 on average
    l1 = [r["total"] for r in rows if r["level"] == "L1"]
    l3 = [r["total"] for r in rows if r["level"] == "L3"]
    if l1 and l3:
        print(f"\nSTRESS CHECK: mean L1={sum(l1)/len(l1):.1f} vs mean L3={sum(l3)/len(l3):.1f} "
              f"-> {'PASS (gradient holds)' if sum(l1)/len(l1) > sum(l3)/len(l3) else 'FAIL'}")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"saved {OUT_CSV}")


if __name__ == "__main__":
    main()
