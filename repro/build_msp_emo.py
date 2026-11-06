"""Build two emotional zero-shot TTS test sets from MSP-IMPROV, in the Seed-TTS test-set layout.

Each target is a real recording; its prompt comes from the SAME speaker in the SAME emotion, saying something else.
Only clips whose consensus label (listeners) equals the emotion of the scene they were recorded in are used, as
targets and as prompts.

  fixed   lexically controlled: targets are the 20 fixed target sentences (T = said inside the improvised scene,
          R = read). Every sentence exists in all four emotions, so the text alone does not give the emotion away.
          Prompt clips are the same recording type as the target (R -> R, T -> T) and never the target's sentence.
  improv  improvised content: targets are the other lines the actors made up inside the emotional scenes (S),
          transcribed with Whisper large-v3. Prompt clips are the speaker's read target sentences (R) of that
          emotion, topped up with T clips when the speaker has too little R speech.

Prompts: clips drawn in a fixed pseudo-random order (seeded by the target id) and joined with 0.2 s of silence
until they hold >= 3 s of speech (at most 3 clips); targets whose prompt stays under 2 s are dropped, and so are
pairs for which F5-TTS's length estimate (prompt length scaled by the text ratio) exceeds 30 s, the longest it
trains on. All audio is the corpus' 16 kHz mono.

Trimming, in two steps: a loose energy trim (frames within 40 dB of the clip peak) before ASR, then each kept clip
is cut to its transcript's words, force-aligned with MMS_FA as in prosody_corr.py, plus 0.20 s before the first
word and 0.20 s after the last. The energy trim alone leaves seconds of room noise and the partner's crosstalk in
many clips (the noise floor sits 35-40 dB below the peak), which inflates F5-TTS's prompt-rate length estimate.

Text: MSP-IMPROV ships no transcripts. Every candidate clip is transcribed with faster-whisper large-v3. A fixed-
sentence clip is kept only if its transcript equals the canonical sentence after lowercasing, removing
punctuation and expanding contractions; its text is then that transcript (so "I'm" vs "I am" follows the speaker).
Improvised clips keep their transcript when Whisper is confident (see ASR_OK) and it has no digits (number
normalization is not what this set tests, and the aligner has no digits).

Labels: the class comes from the official consensus (MSP_IMPROV_labels.zip); arousal/valence/dominance from
Labels/labels_consensus_attributes.csv, whose arousal runs 1 = calm .. 5 = active (the official file has it reversed).

Output (OUT = data/msp_improv_emo):
  clips/<clip>.wav             energy-trimmed candidate clips (16 kHz), the ASR input
  asr.jsonl                    Whisper transcript + confidence per clip (cache; requeue-safe)
  align.jsonl                  first-word start / last-word end per transcribed clip (cache)
  cut/<clip>.wav               kept clips cut to their words: the audio every set uses
  clips.tsv                    every candidate clip with labels, transcript and why it was kept or dropped
                               (duration = the cut clip's, empty when it was not cut)
  <set>/meta.lst               utt|prompt_text|prompt-wavs/<utt>.wav|target_text (Seed-TTS format)
  <set>/wavs/<utt>.wav         ground-truth target (symlink into cut/)
  <set>/prompt-wavs/<utt>.wav  the assembled prompt
  <set>/info.tsv               per target: speaker, gender, emotion, type, sentence, labels, prompt clips

Usage: python repro/build_msp_emo.py [--out data/msp_improv_emo]
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import zipfile
import zlib
from collections import defaultdict

import numpy as np
import soundfile as sf
from prosody_corr import Aligner, words_of


DB = "/data/group_data/UTD-NAS/Databases/MSP-IMPROV"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
EMOTIONS = {"A": "angry", "H": "happy", "S": "sad", "N": "neutral"}
PROMPT_SPEECH, PROMPT_MIN, PROMPT_MAX_CLIPS, GAP = 3.0, 2.0, 3, 0.2
CUT_BEFORE, CUT_AFTER = 0.20, 0.20  # margins around the aligned words (MMS_FA word edges run late/early)
MAX_TOTAL = 30.0  # F5-TTS length estimate (prompt + target), seconds

# The 20 target sentences (S01..S20). MSP-IMPROV does not distribute them as text; these are the Whisper large-v3
# transcripts of one read and one improvised clip per sentence, which agreed for every sentence (2026-10-27).
CANONICAL = {
    "S01": "How can I not?",
    "S02": "I am quite sure that we will find some way or another.",
    "S03": "Ella Jorgensen made the pudding.",
    "S04": "The floor was completely covered.",
    "S05": "They are just going to go ahead regardless.",
    "S06": "It has all been scheduled since Wednesday.",
    "S07": "I am going shopping.",
    "S08": "A preliminary study shows rats to be more inquisitive than once thought.",
    "S09": "That's it. The meeting is finished.",
    "S10": "I don't know how she could miss this opportunity.",
    "S11": "It is raining outside.",
    "S12": "Your dog is insane.",
    "S13": "She told me what you did.",
    "S14": "Your grandmother is on the phone.",
    "S15": "Only I joined her in the ceremony.",
    "S16": "Everyone is looking for it.",
    "S17": "This hat makes me look like an aardvark.",
    "S18": "Finally, the mail carrier let me in.",
    "S19": "The way you posed with him is the same way I posed with the kitten.",
    "S20": "I hear the echo of voices and the sound of shoes.",
}
CONTRACTIONS = {
    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
    "don't": "do not",
    "they're": "they are",
    "we'll": "we will",
    "you're": "you are",
    "she's": "she is",
    "he's": "he is",
    "can't": "cannot",
    "isn't": "is not",
    "we're": "we are",
    "i'll": "i will",
    "there's": "there is",
    "it'll": "it will",
    "everyone's": "everyone is",
    "they'll": "they will",
    "what's": "what is",
    "doesn't": "does not",
    "didn't": "did not",
    "wasn't": "was not",
    "i've": "i have",
    "we've": "we have",
    "you've": "you have",
}
TEXT_OK = re.compile(r"^[A-Za-z0-9 ,.?!'\-:;\"]+$")


def norm_words(text):
    words = re.sub(r"[^a-z' ]+", " ", text.lower().replace("’", "'")).split()
    return [x for w in words for x in CONTRACTIONS.get(w, w).split()]


def ASR_OK(a):
    """Improvised lines: keep the transcript only when Whisper is confident and it is plain text."""
    return (
        a["avg_logprob"] >= -0.8
        and a["no_speech_prob"] <= 0.5
        and a["compression_ratio"] <= 2.4
        and len(norm_words(a["text"])) >= 2
        and bool(TEXT_OK.match(a["text"]))
    )


def parse(clip):
    """MSP-IMPROV-S09A-M06-S-MF09.wav -> sentence S09, scene emotion A, speaker M06, type S."""
    _, _, se, spk, typ, _ = clip[:-4].split("-")
    return se[:-1], se[-1], spk, typ


def load_labels():
    z = zipfile.ZipFile(f"{DB}/MSP_IMPROV_labels.zip")
    official = {
        r["FileName"]: r
        for r in csv.DictReader(io.StringIO(z.read("MSP_IMPROV_labels/labels_concensus_1.csv").decode()))
    }
    attrs = {"MSP-" + r[""]: r for r in csv.DictReader(open(f"{DB}/Labels/labels_consensus_attributes.csv"))}
    probs = {"MSP-" + r[""]: r for r in csv.DictReader(open(f"{DB}/Labels/labels_consensus_4class_D.csv"))}
    return official, attrs, probs


def trim(y):
    """Cut leading/trailing silence: frames within 40 dB of the peak, 3 in a row; keep 0.10 s / 0.15 s margins."""
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160, center=True)[0]
    db = 20 * np.log10(rms + 1e-10)
    active = np.convolve(db > db.max() - 40, np.ones(3), "same") >= 3
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return y
    a = max(0, idx[0] * 160 - int(0.10 * SR))
    b = min(len(y), idx[-1] * 160 + int(0.15 * SR))
    return y[a:b]


def transcribe(paths, cache):
    done = {}
    if os.path.exists(cache):
        for ln in open(cache):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:  # last line cut off by a preemption: that clip is redone
                continue
            done[r["clip"]] = r
        with open(cache, "w") as f:  # drop the cut-off line so appends start on a fresh line
            f.writelines(json.dumps(r) + "\n" for r in done.values())
    todo = [p for p in paths if os.path.basename(p) not in done]
    if todo:
        from concurrent.futures import ThreadPoolExecutor

        import torch
        from faster_whisper import WhisperModel

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # on CPU: several 4-thread workers over the allocated cores (faster-whisper runs them concurrently)
        workers = 1 if dev == "cuda" else max(1, len(os.sched_getaffinity(0)) // 4)
        model = WhisperModel(
            "large-v3",
            device=dev,
            compute_type="float16" if dev == "cuda" else "int8",
            cpu_threads=4,
            num_workers=workers,
        )

        def run(p):
            segs = list(model.transcribe(p, language="en", beam_size=5, condition_on_previous_text=False)[0])
            return {
                "clip": os.path.basename(p),
                "text": " ".join(s.text.strip() for s in segs).strip(),
                "avg_logprob": min((s.avg_logprob for s in segs), default=-9.0),
                "no_speech_prob": max((s.no_speech_prob for s in segs), default=1.0),
                "compression_ratio": max((s.compression_ratio for s in segs), default=0.0),
            }

        print(f"asr on {dev}, {workers} worker(s), {len(todo)} clips", flush=True)
        with open(cache, "a") as f, ThreadPoolExecutor(workers) as ex:
            for i, r in enumerate(ex.map(run, todo)):
                f.write(json.dumps(r) + "\n")
                done[r["clip"]] = r
                if i % 200 == 0:
                    f.flush()
                    print(f"asr {i}/{len(todo)}", flush=True)
    return done


def cut_to_words(texts, out):
    """clips/<clip> -> cut/<clip>, from the first aligned word's start to the last one's end plus margins.
    texts: {clip: transcript}. Returns {clip: cut duration in s, or the aligner's error string}."""
    cache = f"{out}/align.jsonl"
    done = {}
    if os.path.exists(cache):
        for ln in open(cache):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:  # last line cut off by a preemption: that clip is redone
                continue
            done[r["clip"]] = r
    todo = [c for c in sorted(texts) if c not in done or done[c]["text"] != texts[c]]
    if todo:
        import torch

        aligner = Aligner("cuda" if torch.cuda.is_available() else "cpu")
        print(f"aligning {len(todo)} clips", flush=True)
        with open(cache, "a") as f:
            for c in todo:
                spans = aligner(f"{out}/clips/{c}", words_of(texts[c]))
                r = {"clip": c, "text": texts[c]}
                r.update({"error": spans} if isinstance(spans, str) else {"start": spans[0][0], "end": spans[-1][1]})
                f.write(json.dumps(r) + "\n")
                done[c] = r
        with open(cache, "w") as f:  # rewrite without cut-off lines or superseded entries
            f.writelines(json.dumps(r) + "\n" for r in done.values())

    os.makedirs(f"{out}/cut", exist_ok=True)
    res = {}
    for c in sorted(texts):
        r = done[c]
        if "error" in r:
            res[c] = r["error"]
            continue
        y = sf.read(f"{out}/clips/{c}", dtype="float32")[0]
        a, b = max(0, int((r["start"] - CUT_BEFORE) * SR)), min(len(y), int((r["end"] + CUT_AFTER) * SR))
        p = f"{out}/cut/{c}"
        sf.write(p + ".tmp.wav", y[a:b], SR, subtype="PCM_16")
        os.replace(p + ".tmp.wav", p)
        res[c] = (b - a) / SR
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/data/msp_improv_emo")
    args = ap.parse_args()
    out = args.out
    os.makedirs(f"{out}/clips", exist_ok=True)

    official, attrs, probs = load_labels()
    # candidates: R/T/S clips that listeners heard as the scene's emotion
    cand = sorted(c for c, r in official.items() if parse(c)[3] in "RTS" and r["EmoClass"] == parse(c)[1])
    print(f"{len(cand)} candidate clips (R/T/S, perceived = scene emotion)", flush=True)

    zf = zipfile.ZipFile(f"{DB}/MSP-IMPROV_auds_mono.zip")
    for c in cand:
        p = f"{out}/clips/{c}"
        if not os.path.exists(p):
            y, sr = sf.read(io.BytesIO(zf.read(f"AUDs_mono/{c}")), dtype="float32")
            assert sr == SR and y.ndim == 1, (c, sr, y.shape)
            sf.write(p + ".tmp.wav", trim(y), SR, subtype="PCM_16")
            os.replace(p + ".tmp.wav", p)
    asr = transcribe([f"{out}/clips/{c}" for c in cand], f"{out}/asr.jsonl")

    canon = {k: norm_words(v) for k, v in CANONICAL.items()}
    canon_set = {tuple(v) for v in canon.values()}
    status = {}
    for c in cand:
        sent, _, _, typ = parse(c)
        a = asr[c]
        if typ in "RT":
            status[c] = (
                "kept" if norm_words(a["text"]) == canon[sent] else "transcript differs from the target sentence"
            )
        elif not ASR_OK(a):
            status[c] = "low-confidence or non-text transcript"
        elif tuple(norm_words(a["text"])) in canon_set:
            status[c] = "is a target sentence"
        elif re.search(r"\d", a["text"]):
            status[c] = "text has digits"
        else:
            status[c] = "kept"
    cut = cut_to_words({c: asr[c]["text"] for c in cand if status[c] == "kept"}, out)
    for c, d in cut.items():
        if isinstance(d, str):
            status[c] = "alignment failed"
        elif parse(c)[3] == "S" and not 1.0 <= d <= 15.0:
            status[c] = "duration"
    cps = [len(asr[c]["text"]) / d for c, d in cut.items() if status[c] == "kept" and parse(c)[3] in "RT"]
    print(
        "fixed-sentence clips after cutting, chars/s: q01 %.1f, median %.1f, q99 %.1f"
        % tuple(np.quantile(cps, [0.01, 0.5, 0.99])),
        flush=True,
    )

    clips, rows = {}, []
    for c in cand:
        sent, emo, spk, typ = parse(c)
        dur = cut.get(c)
        info = {
            "clip": c,
            "speaker": spk,
            "gender": "female" if spk[0] == "F" else "male",
            "emotion": EMOTIONS[emo],
            "type": typ,
            "sentence": sent if typ in "RT" else "",
            "text": asr[c]["text"],
            "duration": round(dur, 3) if isinstance(dur, float) else "",
            "arousal": float(attrs[c]["EmoAct"]),
            "valence": float(attrs[c]["EmoVal"]),
            "dominance": float(attrs[c]["EmoDom"]),
            **{f"p_{k}": float(probs[c][k]) for k in ("angry", "happy", "sad", "neutral")},
            "status": status[c],
        }
        rows.append(info)
        if status[c] == "kept":
            clips[c] = info
    with open(f"{out}/clips.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print("clip status:", {k: sum(r["status"] == k for r in rows) for k in sorted({r["status"] for r in rows})})

    by = defaultdict(list)  # (speaker, emotion, type) -> clips
    for c, i in clips.items():
        by[(i["speaker"], i["emotion"], i["type"])].append(c)
    for v in by.values():
        v.sort()

    def assemble(target, pools):
        """Draw prompt clips pool by pool, each pool in a seeded order, until >= PROMPT_SPEECH s of speech."""
        rng = np.random.default_rng(zlib.crc32(target.encode()))
        chosen, speech = [], 0.0
        for pool in pools:
            for j in rng.permutation(len(pool)):
                if speech >= PROMPT_SPEECH or len(chosen) == PROMPT_MAX_CLIPS:
                    break
                chosen.append(pool[j])
                speech += clips[pool[j]]["duration"]
        return chosen, speech

    for name in ("fixed", "improv"):
        d = f"{out}/{name}"
        shutil.rmtree(d, ignore_errors=True)  # fully derived; no stale targets or links from an earlier build
        os.makedirs(f"{d}/wavs")
        os.makedirs(f"{d}/prompt-wavs")
        meta, info_rows, dropped, too_long = [], [], 0, 0
        for t, i in sorted(clips.items()):
            key = (i["speaker"], i["emotion"])
            if name == "fixed" and i["type"] in "RT":
                pools = [[c for c in by[(*key, i["type"])] if clips[c]["sentence"] != i["sentence"]]]
            elif name == "improv" and i["type"] == "S":
                pools = [by[(*key, "R")], by[(*key, "T")]]
            else:
                continue
            chosen, speech = assemble(t, pools)
            if speech < PROMPT_MIN:
                dropped += 1
                continue
            utt = t[:-4]
            gap = np.zeros(int(GAP * SR), dtype=np.float32)
            parts = []
            for c in chosen:
                parts += [sf.read(f"{out}/cut/{c}", dtype="float32")[0], gap]
            prompt = np.concatenate(parts[:-1])
            prompt_text = " ".join(clips[c]["text"] for c in chosen)
            # F5-TTS's duration (utils_eval.get_inference_prompt): prompt length * (1 + target bytes / prompt bytes),
            # a space appended to a prompt text ending in an ASCII character
            pt = prompt_text + " " if len(prompt_text[-1].encode()) == 1 else prompt_text
            if len(prompt) / SR * (1 + len(i["text"].encode()) / len(pt.encode())) > MAX_TOTAL:
                too_long += 1
                continue
            sf.write(f"{d}/prompt-wavs/{utt}.wav", prompt, SR, subtype="PCM_16")
            os.symlink(f"../../cut/{t}", f"{d}/wavs/{utt}.wav")
            meta.append(f"{utt}|{prompt_text}|prompt-wavs/{utt}.wav|{i['text']}")
            info_rows.append(
                {
                    **{k: v for k, v in i.items() if k != "status"},
                    "utt": utt,
                    "prompt_clips": ",".join(c[:-4] for c in chosen),
                    "prompt_types": ",".join(clips[c]["type"] for c in chosen),
                    "prompt_speech_s": round(speech, 3),
                }
            )
        open(f"{d}/meta.lst", "w").write("\n".join(meta) + "\n")
        with open(f"{d}/info.tsv", "w") as f:
            w = csv.DictWriter(f, fieldnames=list(info_rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(info_rows)
        cnt = defaultdict(int)
        for r in info_rows:
            cnt[r["emotion"]] += 1
        print(
            f"{name}: {len(meta)} targets {dict(cnt)}, {len({r['speaker'] for r in info_rows})} speakers, "
            f"{dropped} dropped (prompt < {PROMPT_MIN} s), {too_long} dropped (F5 length estimate > {MAX_TOTAL} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
