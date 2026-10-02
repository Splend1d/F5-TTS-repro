"""Summarize WER/SIM/UTMOS over seeds: python repro/summarize.py results/<MODEL>_<STEP> [...]"""
import glob, os, re, sys
import numpy as np

# paper / README reference numbers (F5-TTS v1 Base from SWivid/F5-TTS, v0 Base from arXiv 2410.06885)
for root in sys.argv[1:]:
    print(f"== {root}")
    for task in sorted(os.listdir(root)):
        row = {}
        for m in ("wer", "sim", "utmos"):
            vals = []
            for f in sorted(glob.glob(f"{root}/{task}/seed*/_{m}_results.jsonl")):
                last = [l for l in open(f).read().strip().splitlines() if l.startswith(m.upper())]
                if last:
                    vals.append(float(re.findall(r"[-\d.]+", last[-1])[-1]))
            if vals:
                v = np.array(vals) * (100 if m == "wer" else 1)
                row[m] = f"{v.mean():.3f}±{v.std():.3f} (n={len(v)})"
        print(f"  {task:18s} " + "  ".join(f"{k.upper()}={v}" for k, v in row.items()))
