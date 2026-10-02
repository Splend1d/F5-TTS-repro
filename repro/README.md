# F5-TTS v1 Base reproduction (Emilia ZH+EN)

This folder reproduces `F5TTS_v1_Base` (DiT, 1.25M updates) from scratch on Emilia ZH+EN. It is a fork of
[SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) at `2832525` (v1.1.22). The training scripts are set up for
a preemptible SLURM cluster (CMU babel).

- **Checkpoint (private):** https://huggingface.co/SpeechGenCourse/F5TTS_v1_Base_Emilia_ZH_EN_repro
- **W&B:** project `splend1dchan/CFM-TTS`, run id `f5tts-main-20260930`
  (https://wandb.ai/splend1dchan/CFM-TTS/runs/f5tts-main-20260930). This run is created on the first resume
  after 2026-10-01. Updates 0–5,631 were logged to TensorBoard; those event files are in the HF repo under `runs/`.

## Current progress (2026-10-01)

| | |
|---|---|
| Latest checkpoint | `model_last.pt` @ **update 5,500** / 1,250,000 (0.44%) |
| Epoch | 1 / 11 (115,551 updates per epoch) |
| Train loss | ~0.94–1.05 at update ~5.6k |
| Throughput | ~4.6 s/update at 16 × 19,200 frames (8×A100 or 4 GPUs × accum 4) |
| LR | still in warmup (20k updates, peak 7.5e-5) |
| Eval | none yet; a sanity check is planned, and the first real comparison at 50k |

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

# preempt partition, 8- and 4-GPU variants queued; the first to start wins and cancels the other
bash repro/submit_train.sh
# or the general partition on 8xA100-80GB (2-day limit, self-resubmits)
PART=general QOS=normal TLIMIT=2-00:00:00 GTYPE=A100_80GB VARIANTS=8 TAKEOVER=1 bash repro/submit_train.sh
```

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

## License

The code is MIT, as upstream. Emilia is CC BY-NC 4.0, so the checkpoints are for non-commercial research only.
