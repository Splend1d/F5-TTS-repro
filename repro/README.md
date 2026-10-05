# F5-TTS v1 Base reproduction (Emilia ZH+EN)

This folder reproduces `F5TTS_v1_Base` (DiT, 1.25M updates) from scratch on Emilia: ZH+EN for the first 5,500
updates, then EN only (see [Training data](#training-data)). It is a fork of
[SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) at `2832525` (v1.1.22). The training scripts are set up for
a preemptible SLURM cluster (CMU babel).

- **Live dashboard:** https://splend1d.github.io/F5-TTS-repro/ (refreshed every 6 h)
- **Checkpoint:** https://huggingface.co/SpeechGenCourse/F5TTS_v1_Base_Emilia_ZH_EN_repro
- **W&B:** project `splend1dchan/CFM-TTS`, run id `f5tts-main-20260930`
  (https://wandb.ai/splend1dchan/CFM-TTS/runs/f5tts-main-20260930). This run is created on the first resume
  after 2026-10-01. Updates 0–5,631 were logged to TensorBoard; those event files are in the HF repo under `runs/`.

## Training data

The training data is not the same throughout the run:

- **Updates 0–5,500:** Emilia ZH+EN (37.8M utterances, about 95k h).
- **From update 5,500 on:** Emilia EN only (18.1M utterances, 46,643 h), a subset of the same data.
- The LR schedule is still pinned to the original ZH+EN end point.

Details: the EN set is `data/Emilia_EN_pinyin`, cut from the ZH+EN set by `repro/make_en_subset.py`. Its checkpoints
go to `ckpts/F5TTS_v1_Base_vocos_pinyin_Emilia_EN`. `optim.total_updates=1271061` keeps the LR schedule and the end of
training identical to the ZH+EN run (11 × 115,551 updates). These overrides live in `repro/state/main.env`.

## Current progress (2026-10-05)

| | |
|---|---|
| Latest checkpoint on HF | `model_last.pt` @ **update 40,000** / 1,271,061 (3.1%) |
| Train loss | ~0.8–1.1 at update ~40k |
| Throughput | ~2.44 s/update on 4× L40S, one node (accum 4) |
| LR | past warmup (20k updates, peak 7.5e-5), now in linear decay |
| Jobs | one uninterrupted 2-day job on `general` (any 4 GPUs), resubmitting itself at the time limit |
| Eval | none yet; the first real comparison is planned at 50k |

The checkpoint is the full resumable state: model, EMA, optimizer, scheduler and `update`.

## Config

The base config is `src/f5_tts/configs/F5TTS_v1_Base.yaml`. Overrides are passed by `repro/train.sbatch`:

- **Model:** DiT, dim 1024, depth 22, 16 heads, ff_mult 2, text_dim 512, conv_layers 4, pinyin tokenizer, vocos mel
  (24 kHz, 100 mels, hop 256, win/n_fft 1024).
- **Data:** `Emilia_ZH_EN` from Emilia pinned at rev `fc71e07` (the old folder format). That is 37.84M utterances,
  ~95,282 h, at `data/Emilia_ZH_EN_pinyin/{raw.arrow,duration.json,vocab.txt}`. `vocab.txt` is the **official**
  vocab from the released model.
- **Batch:** frame-based, 19,200 frames/GPU, `max_samples=32` (official uses 64), with NGPU × accum = 16. The
  effective batch is 307,200 frames/update, matching the official 8 × 38,400. Accum is derived from the GPU count,
  so the run can move between 4- and 8-GPU nodes.
- **Optimizer:** AdamW, LR 7.5e-5, 20k warmup, grad clip 1.0, 11 epochs, bf16.
- **Checkpoints:** `model_last.pt` every 500 updates, plus `model_<N>.pt` every 50k.

## Code changes vs upstream

- `src/f5_tts/model/trainer.py`: with `F5_SAVE_ON_SIGTERM=1`, all ranks agree to stop at the next micro-step on
  SIGTERM, then save `model_last.pt` and exit 143. Dataloader workers ignore SIGTERM. `model_last.pt` is written
  to a temp file and renamed, so a preemption mid-save can't corrupt it.
- `src/f5_tts/model/trainer.py`, resume state:
  - **LR schedule:** the same warmup + linear decay curve, but computed from the update count and stepped once per
    update. Upstream lets accelerate step it once per GPU per update, so a saved scheduler was only valid on the
    same GPU count; moving between 4 and 8 GPUs would have changed the schedule. On resume it is rebuilt from `update`.
  - **RNG:** each rank saves its Python/NumPy/torch/CUDA RNG state (taken at the last update boundary) to
    `rng_last/rank<r>.pt` next to `model_last.pt`, and restores it on resume. If the GPU count changed, or the
    checkpoint predates this (like the 5,500 one), ranks are seeded deterministically from (seed, update, rank).
  - **Sessions:** with `F5_SAVES_PER_SESSION=N`, the process exits with code 3 after its N-th periodic save.
- `src/f5_tts/train/train.py`: the env vars `F5_LOGGER`, `WANDB_PROJECT`, `WANDB_NAME` and `WANDB_RUN_ID`
  override the logger and W&B identity, so a job that is already queued can switch loggers and every resubmit
  continues the same W&B run.
- `src/f5_tts/configs/F5TTS_v1_Base_official.yaml`: a config for evaluating the released checkpoint.

## Environment (uv)

```bash
git clone git@github.com:Splend1d/F5-TTS-repro.git F5-TTS && cd F5-TTS
uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install -r repro/requirements.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install -e . --no-deps
python -c "import pypinyin; assert pypinyin.__version__ == '0.54.0'"
```

**Keep `pypinyin==0.54.0`.** Version 0.55 adds neutral tones (e.g. `yi4 si`), which are missing from the official
`vocab.txt`. Missing tokens map silently to id 0, which breaks both data prep and inference.

`repro/env.sh` sets the cache locations and activates the venv. It has the cluster paths hardcoded
(`R=/data/group_data/UTD-NAS/chanjanh/speechgen`). On another machine, edit `R`, the `HF_TOKEN` source, and the
`F5=` and `#SBATCH --output` paths in `train.sbatch`, `submit_train.sh`, `stop_train.sh` and `eval.sbatch`.

## Data

```bash
sbatch repro/download_emilia.sbatch   # Emilia ZH+EN @ fc71e07 -> $R/data_raw/Emilia/raw (idempotent)
sbatch repro/prepare.sbatch           # -> data/Emilia_ZH_EN_pinyin/{raw.arrow,duration.json,vocab.prepared.txt}
cp data/Emilia_ZH_EN_pinyin/vocab.official.txt data/Emilia_ZH_EN_pinyin/vocab.txt   # official vocab
```

## Resume from the current checkpoint

The trainer automatically resumes from `ckpts/F5TTS_v1_Base_vocos_pinyin_Emilia_ZH_EN/model_last.pt`:

```bash
source repro/env.sh
hf download SpeechGenCourse/F5TTS_v1_Base_Emilia_ZH_EN_repro model_last.pt \
  --local-dir ckpts/F5TTS_v1_Base_vocos_pinyin_Emilia_ZH_EN

# W&B: repro/state/main.env pins the run id. Every (re)submit appends to the same run.
wandb login

# current setup: preempt, 4 x L40S requested as 4 tasks x 1 GPU (may span nodes), 45 min sessions
SPREAD=1 GTYPE=L40S VARIANTS=4 bash repro/submit_train.sh
# all GPUs on one node instead (8- and 4-GPU variants queued; the first to start wins)
bash repro/submit_train.sh
```

`SPREAD=1` asks SLURM for `--ntasks=N --gpus-per-task=1` instead of N GPUs on one node, which fits into scattered
free GPUs. Ranks are started with `srun` and use DDP over InfiniBand (~1.35 GB of gradients per update). NGPU,
accumulation, data order and resume are the same as on one node. Keep `GTYPE` pinned so GPU types aren't mixed.

Training runs as a chain of short **sessions**. Each job has a 45 min limit (`TLIMIT`; a 500-update session takes ~27 min on 4× L40S). It resumes, trains to the
next `model_last.pt` save (every `LAST=500` updates, ~40 min on 4× A6000), saves once, exits, and queues the next
session. Short jobs start sooner through backfill. If a session hits its limit or is preempted first, it saves on
SIGTERM instead, so there is still at most one save per session. `SAVES_PER_SESSION=0` gives one long job instead.

What a resume restores: model, EMA, optimizer, LR position, update count, data order and position (seeded
sampler plus batch skip, consistent across 4↔8 GPUs), per-rank RNG (same GPU count) and the W&B run. Any partial
gradient accumulation at a SIGTERM save is redone.

Without SLURM, on one 8-GPU node:

```bash
source repro/env.sh
accelerate launch --mixed_precision bf16 src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml \
  ++datasets.batch_size_per_gpu=19200 ++datasets.max_samples=32 ++optim.grad_accumulation_steps=2 \
  ++ckpts.last_per_updates=500 ++ckpts.logger=wandb
```

Set accum so that NGPU × accum = 16, otherwise the effective batch and the resume data position change.

### Run control

- `bash repro/stop_train.sh [TAG]` stops a run cleanly. **Don't plain `scancel`**: the job's SIGTERM trap resubmits it.
  To restart after a stop: `rm repro/state/main.stop && bash repro/submit_train.sh`.
- State files are in `repro/state/<TAG>.{lock,siblings,fails,exclude,done,stop,env}` and logs in `repro/logs/`.
- A crash on a node with a GPU-looking error adds that node to `<TAG>.exclude`. After 3 crashes in a row with no
  progress, the run stops resubmitting.

## Evaluation

```bash
# the eval script takes an integer step, so use a numbered checkpoint copy (model_5500.pt, model_50000.pt, ...)
MODEL=F5TTS_v1_Base STEP=50000 SEEDS="0 1 2" sbatch repro/eval.sbatch     # seed-tts zh/en + LibriSpeech-PC
MODEL=F5TTS_v1_Base_official STEP=1250000 sbatch repro/eval.sbatch        # released model, for reference
python repro/summarize.py results/F5TTS_v1_Base_50000
```

`repro/ckpts_eval/` (not committed) holds the WavLM-large SIM model (`wavlm_large_finetune.pth`).

## Progress page

`docs/` is served by GitHub Pages. `repro/progress/update.sh` runs `export.py`, which writes `docs/metrics.json`
(W&B history, job state, checkpoint step, evals), and pushes when it changes. To refresh it every 6 h, run once on
a **login node**: `bash repro/progress/install_timer.sh`.

## License

The code is MIT, as upstream. Emilia is CC BY-NC 4.0, so the checkpoints are for non-commercial research only.
