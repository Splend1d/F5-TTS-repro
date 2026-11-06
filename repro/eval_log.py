"""Record one checkpoint's Seed-TTS test-en scores: results/<label>/summary.jsonl + W&B.

Usage: python repro/eval_log.py <LABEL> <CFG> <UPDATE> <CKPT> <RESULT_DIR>
       python repro/eval_log.py --rebuild <LABEL>      # rebuild the W&B eval run from summary.jsonl

W&B: one eval run per training run (name f5-eval-<label>, id in results/<label>/.wandb_run_id) in the same project
as training. Its W&B step is the checkpoint update, so every default chart is checkpoint vs performance. It also
holds checkpoint-vs-performance line charts, a train-vs-held-out loss chart and listening samples. Several eval jobs
can finish at once and W&B allows one writer per run, so writes are serialized with a lock file.
"""

import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path


F5 = Path("/data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS-sem")
METRICS = [("wer", "WER (%)"), ("sim", "SIM"), ("utmos", "UTMOS")]
LOSS_KEYS = [
    ("eval_loss", "loss/eval"),
    ("eval_loss_ema", "loss/eval_ema"),
    ("train_loss_at_ckpt", "loss/train_at_ckpt"),
]


def final_score(d, name):
    if not (d / f"_{name}_results.jsonl").exists():  # loss-only eval: no generation
        return None
    lines = [ln for ln in (d / f"_{name}_results.jsonl").read_text().splitlines() if ln.startswith(name.upper())]
    return float(re.findall(r"[-\d.]+", lines[-1])[-1])


TRAIN_RUNS = {
    "main": "f5tts-main-20260930",
    "libritts": "f5tts-libritts-20261007",
    "emilia_small": "f5tts-emilia-small-20261011",
}  # training W&B run ids


def eval_losses(d):
    f = d / "_loss_results.json"
    if not f.exists():
        return {}
    r = json.loads(f.read_text())
    return {"eval_loss": r["eval_loss_online"], "eval_loss_ema": r["eval_loss_ema"]}


def train_loss_near(label, update, window=500):
    """Mean logged training loss over the `window` updates up to this checkpoint (None if not in W&B)."""
    try:
        import wandb

        run = wandb.Api().run(
            f"{os.environ.get('WANDB_ENTITY', 'splend1dchan')}/CFM-TTS/{TRAIN_RUNS[label.removesuffix('_online')]}"
        )
        vals = [
            h["loss"]
            for h in run.scan_history(keys=["loss"], min_step=update - window, max_step=update + 1)
            if h.get("loss") is not None
        ]
        return sum(vals) / len(vals) if vals else None
    except Exception as e:
        print(f"train loss lookup failed: {e}")
        return None


def record(label, cfg, update, ckpt, d, mode="full"):
    row = {
        "mode": mode,
        "weights": "online" if label.endswith("_online") else "ema",
        "label": label,
        "config": cfg,
        "update": update,
        "checkpoint": ckpt,
        "testset": "seedtts_test_en",
        "seed": 0,
        "nfe": 32,
        "wer": final_score(d, "wer"),
        "sim": final_score(d, "sim"),
        "utmos": final_score(d, "utmos"),
        **eval_losses(d),
        "train_loss_at_ckpt": train_loss_near(label, update),
        "result_dir": str(d),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(F5 / "results" / label / "summary.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    return row


def load_summary(label):
    rows = {}
    for ln in (F5 / "results" / label / "summary.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            # merge evaluations of the same checkpoint; later non-null values win
            rows[r["update"]] = {**rows.get(r["update"], {}), **{k: v for k, v in r.items() if v is not None}}
    return [rows[u] for u in sorted(rows)]


SAMPLE_EVERY = 180  # 6 fixed Seed-TTS test-en items (lines 0, 180, ..., 900 of meta.lst), same for every checkpoint


def sample_rows(update, d):
    """Fixed listening samples for one checkpoint: prompt audio, generated audio, target text, ASR, WER."""
    import wandb

    if not (d / "_wer_results.jsonl").exists():  # loss-only eval: nothing to listen to
        return []
    meta = (F5 / "data/seedtts_testset/en/meta.lst").read_text().splitlines()[::SAMPLE_EVERY]
    asr = {}
    for ln in (d / "_wer_results.jsonl").read_text().splitlines():
        if ln.startswith("{"):
            r = json.loads(ln)
            asr[r["wav"]] = r
    out = []
    for ln in meta:
        utt, prompt_text, prompt_wav, target_text = ln.split("|")
        gen = d / f"{utt}.wav"
        if not gen.exists():
            continue
        a = asr.get(utt, {})
        out.append(
            [
                update,
                utt,
                target_text,
                a.get("hypo", "").strip(),
                a.get("wer"),
                wandb.Audio(str(F5 / "data/seedtts_testset/en" / prompt_wav), caption=prompt_text),
                wandb.Audio(str(gen), caption=target_text),
            ]
        )
    return out


def scalars(r):
    out = {k2: r[k1] for k1, k2 in LOSS_KEYS if r.get(k1) is not None}
    if r.get("wer") is not None:  # full eval only
        out.update({"seedtts_en/wer": r["wer"] * 100, "seedtts_en/sim": r["sim"], "seedtts_en/utmos": r["utmos"]})
    return out


def charts(wandb, label, rows):
    """Checkpoint-vs-performance tables/charts over all evaluated checkpoints (sorted by update)."""
    log = {}
    full = [r for r in rows if r.get("wer") is not None]
    table = wandb.Table(
        columns=["checkpoint_update", "wer", "sim", "utmos", "eval_loss", "eval_loss_ema", "train_loss_at_ckpt"],
        data=[
            [r["update"], r["wer"] * 100, r["sim"], r["utmos"]]
            + [r.get(k) for k in ("eval_loss", "eval_loss_ema", "train_loss_at_ckpt")]
            for r in full
        ],
    )
    log["seedtts_en_by_checkpoint/table"] = table
    for key, title in METRICS:
        log[f"seedtts_en_by_checkpoint/{key}"] = wandb.plot.line(
            table, "checkpoint_update", key, title=f"[{label}] Seed-TTS test-en {title} vs checkpoint"
        )
    lr = [r for r in rows if r.get("eval_loss") is not None]
    if lr:  # training loss and held-out loss on one chart, x = checkpoint update
        series = [
            ("train_loss_at_ckpt", "train loss (mean of 500 updates before ckpt)"),
            ("eval_loss", "eval loss (held-out, online weights)"),
            ("eval_loss_ema", "eval loss (held-out, EMA weights)"),
        ]
        keep = [(k, n) for k, n in series if all(r.get(k) is not None for r in lr)]
        log["loss/train_vs_eval"] = wandb.plot.line_series(
            xs=[r["update"] for r in lr],
            ys=[[r[k] for r in lr] for k, _ in keep],
            keys=[n for _, n in keep],
            title=f"[{label}] train vs held-out loss by checkpoint",
            xname="checkpoint update",
        )
    return log


SAMPLE_COLS = ["checkpoint_update", "utt", "target_text", "asr_hypothesis", "wer", "prompt_audio", "generated_audio"]


def log_wandb(label, cfg=None, new_row=None, rebuild=False):
    """The W&B step IS the checkpoint update, so every default chart is checkpoint vs performance.

    W&B steps must increase. A new checkpoint newer than everything logged is appended to the current eval run.
    Otherwise (an out-of-order eval, or --rebuild) a fresh eval run is built from summary.jsonl, one point per
    checkpoint in update order, and the previous eval run is deleted. The current run id is in
    results/<label>/.wandb_run_id.
    """
    if os.environ.get("F5_EVAL_WANDB", "1") != "1":
        return
    import wandb

    rows = load_summary(label)
    cfg = cfg or rows[-1]["config"]
    project = os.environ.get("WANDB_PROJECT", "CFM-TTS")
    entity = os.environ.get("WANDB_ENTITY", "splend1dchan")
    idfile = F5 / "results" / label / ".wandb_run_id"
    with open(F5 / "results" / f".wandb_{label}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old_id = idfile.read_text().strip() if idfile.exists() else None
        # last update logged to the current run, tracked locally (the W&B API summary can lag by minutes)
        lastfile = F5 / "results" / label / ".wandb_last_update"
        last = int(lastfile.read_text()) if (old_id and not rebuild and lastfile.exists()) else None
        append = new_row is not None and last is not None and new_row["update"] > last
        run_id = old_id if append else f"f5-eval-{label}-{time.strftime('%Y%m%d-%H%M%S')}"
        run = wandb.init(
            project=project,
            id=run_id,
            name=f"f5-eval-{label}",
            group=label,
            job_type="eval",
            resume="allow",
            config={"label": label, "model_config": cfg, "testset": "seedtts_test_en", "nfe": 32, "seed": 0},
            settings=wandb.Settings(init_timeout=300),
        )
        todo = [new_row] if append else rows
        for i, r in enumerate(todo):
            log = scalars(r)
            log["samples/seedtts_en"] = wandb.Table(
                columns=SAMPLE_COLS, data=sample_rows(r["update"], Path(r["result_dir"]))
            )
            if i == len(todo) - 1:
                log.update(charts(wandb, label, rows))
            run.log(log, step=r["update"])
        lf = [r for r in rows if r.get("wer") is not None]
        if lf:
            run.summary.update({f"latest/{k}": (lf[-1][k] * 100 if k == "wer" else lf[-1][k]) for k, _ in METRICS})
        run.summary["latest/checkpoint_update"] = rows[-1]["update"]
        run.finish()
        lastfile.write_text(f"{rows[-1]['update'] if not append else new_row['update']}\n")
        if not append:
            idfile.write_text(run_id + "\n")
            for rid in {old_id, f"f5-eval-{label}-v2"} - {None, run_id}:  # drop the superseded eval run
                try:
                    wandb.Api().run(f"{entity}/{project}/{rid}").delete()
                    print(f"deleted superseded W&B run {rid}")
                except Exception:
                    pass
        print(f"W&B eval run {run_id} ({'appended' if append else 'rebuilt'}; x = checkpoint update)")


if __name__ == "__main__":
    if sys.argv[1] in ("--refresh", "--rebuild", "--relog"):  # rebuild the eval run from summary.jsonl
        log_wandb(sys.argv[2], rebuild=True)
    else:
        label, cfg, update, ckpt, d = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
        mode = sys.argv[6] if len(sys.argv) > 6 else "full"
        row = record(label, cfg, update, ckpt, d, mode)
        log_wandb(label, cfg, row)
