"""Held-out flow-matching loss of one checkpoint, computed exactly like the training loss.

Usage: python repro/eval_loss.py <CFG> <CKPT> <OUT_JSON>

Data: Seed-TTS test-en ground-truth audio (data/seedtts_testset/en/wavs/<utt>.wav) with its target text; never seen
in training. The loss is CFM.forward (random time, noise and infill span, as in training) under bf16 autocast, so
it is directly comparable to the logged training `loss`. Randomness is seeded per (utterance, draw), so every
checkpoint sees the same t / noise / masks. K draws per utterance reduce the variance.

Reports the online weights (what the training loss measures) and the EMA weights (what inference uses).
"""

import json
import sys
from importlib.resources import files

import torch
import torchaudio
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.model import CFM
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer


K = 4  # random draws per utterance
cfg_name, ckpt_path, out_path = sys.argv[1:4]
root = str(files("f5_tts").joinpath("../.."))
cfg = OmegaConf.load(f"{root}/src/f5_tts/configs/{cfg_name}.yaml")
tokenizer = cfg.model.tokenizer
vocab_char_map, vocab_size = get_tokenizer(cfg.datasets.name, tokenizer)
sr = cfg.model.mel_spec.target_sample_rate
dev = "cuda"

model = CFM(
    transformer=get_class(f"f5_tts.model.{cfg.model.backbone}")(
        **cfg.model.arch, text_num_embeds=vocab_size, mel_dim=cfg.model.mel_spec.n_mel_channels
    ),
    mel_spec_kwargs=cfg.model.mel_spec,
    vocab_char_map=vocab_char_map,
    sem_encoder=cfg.model.get("sem_encoder", None),
).to(dev)
ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
weights = {
    "online": ckpt["model_state_dict"],
    "ema": {
        k.replace("ema_model.", ""): v for k, v in ckpt["ema_model_state_dict"].items() if k not in ("initted", "step")
    },
}

base = f"{root}/data/seedtts_testset/en"
items = []
for ln in open(f"{base}/meta.lst").read().splitlines():
    utt, _, _, text = ln.split("|")
    wav, fs = torchaudio.load(f"{base}/wavs/{utt}.wav")
    wav = wav.mean(0)
    if fs != sr:
        wav = torchaudio.functional.resample(wav, fs, sr)
    if not 0.3 <= wav.shape[-1] / sr <= 30:  # same duration filter as the training dataset
        continue
    toks = convert_char_to_pinyin([text], polyphone=True)[0] if tokenizer == "pinyin" else text
    items.append((wav, toks))

res = {"checkpoint": ckpt_path, "update": ckpt.get("update"), "n_utts": len(items), "draws": K}
for name, sd in weights.items():
    model.load_state_dict(sd)
    model.eval()
    losses = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, (wav, toks) in enumerate(items):
            for k in range(K):
                torch.manual_seed(1000 * i + k)
                loss, _, _ = model(wav.unsqueeze(0).to(dev), text=[toks])
                losses.append(loss.float().item())
    res[f"eval_loss_{name}"] = sum(losses) / len(losses)
    print(f"{name}: eval loss {res[f'eval_loss_{name}']:.4f} over {len(items)} utts x {K} draws", flush=True)
json.dump(res, open(out_path, "w"), indent=2)
