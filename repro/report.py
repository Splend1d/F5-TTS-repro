"""Seed-TTS test-en results of every run, EMA and online weights side by side.

Usage: python repro/report.py            # markdown table of the scheduled full evals (every 50k)
       python repro/report.py --all      # also list loss-only checkpoints

Reads results/<run>/summary.jsonl (EMA weights) and results/<run>_online/summary.jsonl (online weights).
Train time = update x the run's mean step time on 4 GPUs, measured from the training logs (tqdm s/update, weighted
by updates per session), so it excludes loading, startup, queueing and re-done work after preemption.
"""

import glob
import json
import re
import statistics
import sys
from pathlib import Path


RESULTS = Path(__file__).resolve().parent.parent / "results"
RUNS = [  # (label, description)
    ("main", "Emilia Base (F5TTS_v1_Base)"),
    ("emilia_small", "Emilia Small (F5TTS_v1_Small)"),
    ("libritts", "LibriTTS Small (F5TTS_v1_Small)"),
]


LOGS = RESULTS.parent / "repro" / "logs"


def step_time(label):
    """Update-weighted mean s/update over all training sessions of a run."""
    tot_u = tot_t = 0.0
    for f in glob.glob(str(LOGS / f"f5-train-{label}-*.out")):
        t = open(f, errors="ignore").read().replace("\r", "\n")
        us = [int(x) for x in re.findall(r"update=(\d+)\]", t)]
        rs = [float(x) for x in re.findall(r"([0-9.]+)s/update", t)]
        if len(us) < 20 or len(rs) < 20 or max(us) - min(us) < 50:
            continue
        n = max(us) - min(us)
        tot_u += n
        tot_t += n * statistics.median(rs[len(rs) // 5 :])  # skip the first readings of a session (warm-up)
    return tot_t / tot_u if tot_u else None


def load(label):
    f = RESULTS / label / "summary.jsonl"
    rows = {}
    if f.exists():
        for ln in f.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                rows[r["update"]] = {**rows.get(r["update"], {}), **{k: v for k, v in r.items() if v is not None}}
    return rows


def q(r):
    if not r or r.get("wer") is None:
        return "—", "—", "—"
    return f"{r['wer'] * 100:.1f}%", f"{r['sim']:.3f}", f"{r['utmos']:.2f}"


def main():
    show_all = "--all" in sys.argv
    print(
        "| Run | Update | Train time (4 GPU) | WER EMA | WER online | SIM EMA | SIM online | UTMOS EMA | UTMOS online "
        "| Eval loss EMA | Eval loss online | Train loss |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    speeds = {}
    for label, desc in RUNS:
        ema, onl = load(label), load(f"{label}_online")
        sp = speeds[label] = step_time(label)
        for u in sorted(set(ema) | set(onl)):
            e, o = ema.get(u, {}), onl.get(u, {})
            if not show_all and (u % 50000 or (e.get("wer") is None and o.get("wer") is None)):
                continue
            (we, se, ue), (wo, so, uo) = q(e), q(o)
            r = e or o
            le = f"{r['eval_loss_ema']:.3f}" if r.get("eval_loss_ema") is not None else "—"
            lo = f"{r['eval_loss']:.3f}" if r.get("eval_loss") is not None else "—"
            tl = f"{r['train_loss_at_ckpt']:.3f}" if r.get("train_loss_at_ckpt") is not None else "—"
            tt = f"{u * sp / 3600:.1f} h" if sp else "—"
            print(f"| {desc} | {u:,} | {tt} | {we} | {wo} | {se} | {so} | {ue} | {uo} | {le} | {lo} | {tl} |")
    print()
    for label, desc in RUNS:
        if speeds.get(label):
            print(f"- {desc}: {speeds[label]:.2f} s/update (mean step time on 4 GPUs, from training logs)")


if __name__ == "__main__":
    main()
