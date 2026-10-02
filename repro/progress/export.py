"""Export training progress for the GitHub Pages dashboard: python repro/progress/export.py [out.json]

Reads the W&B run (downsampled history), the SLURM queue, the latest checkpoint step from the training logs,
and any eval results, and writes docs/metrics.json. Cheap enough to run every few hours from a login node.
"""

import glob
import json
import os
import re
import subprocess
import sys
import time

import wandb


F5 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{F5}/docs/metrics.json"
TAG = os.environ.get("TAG", "main")
ENTITY = os.environ.get("WANDB_ENTITY", "splend1dchan")
PROJECT = os.environ.get("WANDB_PROJECT", "CFM-TTS")
RUN_ID = os.environ.get("WANDB_RUN_ID", "f5tts-main-20260930")
TOTAL_UPDATES = 1_250_000
UPDATES_PER_EPOCH = 115_551
SAMPLES = 4000  # history points served to the page


def wandb_history():
    run = wandb.Api(timeout=60).run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
    rows = run.history(samples=SAMPLES, keys=["loss", "lr", "_timestamp"], pandas=False)
    rows = sorted((r for r in rows if r.get("loss") is not None), key=lambda r: r["_step"])
    hist = {
        "step": [int(r["_step"]) for r in rows],
        "loss": [round(float(r["loss"]), 5) for r in rows],
        "lr": [float(r["lr"]) for r in rows],
    }
    # throughput from live points in the last 3 h of wall-clock (backfilled points share one timestamp)
    now, recent = time.time(), [r for r in rows if r.get("_timestamp", 0) > time.time() - 3 * 3600]
    rate = None
    if len(recent) >= 2:
        dt, ds = recent[-1]["_timestamp"] - recent[0]["_timestamp"], recent[-1]["_step"] - recent[0]["_step"]
        if dt > 1800 and ds > 0:
            rate = dt / ds
    last_ts = rows[-1].get("_timestamp") if rows else None
    return hist, run, rate, (now - last_ts) if last_ts else None


def slurm_status():
    try:
        out = (
            subprocess.run(
                ["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", f"f5-train-{TAG}", "-o", "%T|%N|%b|%M|%r"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            .stdout.strip()
            .splitlines()
        )
    except Exception:
        return {"state": "unknown"}
    if not out:
        done = os.path.exists(f"{F5}/repro/state/{TAG}.done")
        return {"state": "COMPLETED" if done else "NOT QUEUED"}
    jobs = [dict(zip(["state", "node", "gres", "elapsed", "reason"], line.split("|"))) for line in out]
    running = [j for j in jobs if j["state"] == "RUNNING"]
    j = running[0] if running else jobs[0]
    gres = re.sub(r"^gres/|^gres:", "", j["gres"]).replace("gpu:", "")
    return {
        "state": j["state"],
        "node": j["node"] if running else None,
        "gpus": gres or None,
        "elapsed": j["elapsed"] if running else None,
        "reason": None if running else j["reason"],
    }


def last_checkpoint():
    logs = sorted(glob.glob(f"{F5}/repro/logs/f5-train-{TAG}-*.out"), key=os.path.getmtime)[-3:]
    best = None
    for f in logs:
        for m in re.finditer(rb"Saved last checkpoint at update (\d+)", open(f, "rb").read()):
            best = max(best or 0, int(m.group(1)))
    return best


def evals():
    res = []
    for root in sorted(glob.glob(f"{F5}/results/F5TTS_v1_Base_[0-9]*")):
        step = int(root.rsplit("_", 1)[1])
        for task_dir in sorted(glob.glob(f"{root}/*")):
            row = {"step": step, "task": os.path.basename(task_dir)}
            for m in ("wer", "sim", "utmos"):
                vals = []
                for f in sorted(glob.glob(f"{task_dir}/seed*/_{m}_results.jsonl")):
                    last = [ln for ln in open(f).read().strip().splitlines() if ln.startswith(m.upper())]
                    if last:
                        vals.append(float(re.findall(r"[-\d.]+", last[-1])[-1]))
                if vals:
                    row[m] = round(sum(vals) / len(vals), 4)
                    row[f"{m}_n"] = len(vals)
            if len(row) > 2:
                res.append(row)
    return sorted(res, key=lambda r: (r["step"], r["task"]))


def main():
    hist, run, sec_per_update, since_last_log = wandb_history()
    step = hist["step"][-1] if hist["step"] else 0
    status = slurm_status()
    ckpt = last_checkpoint()
    data = {
        "generated_at": int(time.time()),
        "run": {"entity": ENTITY, "project": PROJECT, "id": RUN_ID, "url": run.url, "wandb_state": run.state},
        "progress": {
            "update": step,
            "total_updates": TOTAL_UPDATES,
            "epoch": step // UPDATES_PER_EPOCH + 1,
            "epochs": 11,
            "checkpoint_update": ckpt,
            "sec_per_update": round(sec_per_update, 3) if sec_per_update else None,
            "eta_days": round((TOTAL_UPDATES - step) * sec_per_update / 86400, 1) if sec_per_update else None,
            "seconds_since_last_log": int(since_last_log) if since_last_log else None,
        },
        "job": status,
        "history": hist,
        "evals": evals(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, OUT)
    print(f"wrote {OUT}: update {step}, {len(hist['step'])} points, job {status.get('state')}")


if __name__ == "__main__":
    main()
