"""Listening page for the MSP-IMPROV test sets (repro/build_msp_emo.py): for a sample of targets, the prompt, the real
recording and every system's output, each with emotion2vec+'s label.

Rows are drawn at random (seed 0) under coverage rules, never by how they sound:
  fixed   "one sentence, four emotions": GROUPS (speaker, sentence, recording type) combinations that the set has in
          all four emotions, with distinct speakers and sentences, half female / half male and half R / half T as far
          as the set allows. One target per emotion in each group.
  improv  PER_EMOTION targets per emotion, distinct speakers within an emotion (each speaker at most 3 times
          overall), half female / half male, a third each short (< 2 s), medium (2-4 s) and long (4-10 s).

Needs repro/msp_emo_eval.sbatch outputs (results/msp_emo/<system>/<set>/) and repro/msp_emo_analysis.sbatch outputs
(results/_analysis/msp_emo/<set>/: emotion2vec+ cache, text-emotion groups). Audio is re-encoded to MP3 (mono,
64 kb/s) under OUT/audio/; OUT/index.html has the data inline, so it also opens from disk.

Usage: python repro/make_msp_demo.py [--out repro/demo/msp_improv] [--groups 8] [--per-emotion 6]
"""

import argparse
import json
import os
import pickle
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{ROOT}/data/msp_improv_emo"
GEN = f"{ROOT}/results/msp_emo"
ANA = f"{ROOT}/results/_analysis/msp_emo"
EMOS = ("neutral", "happy", "angry", "sad")  # row order inside a fixed-text group
E2V_IDX = {"angry": 0, "happy": 3, "neutral": 4, "sad": 6}  # emotion2vec+ 9-way label order (as msp_emo_eval.py)
SYSTEMS = [  # results/msp_emo/<name>, column label, description
    ("small250k", "Small", "F5-TTS v1 Small trained on Emilia EN, update 250k (run emilia_small): the baseline."),
    (
        "colbert250k",
        "+ColBERT",
        "Small with frozen GTE-ModernColBERT token vectors added to the text input, from Small's update 150k, "
        "update 250k (run emilia_small_sem).",
    ),
    (
        "bert250k",
        "+ModernBERT",
        "The same with ModernBERT-base token vectors, the control without retrieval training (run emilia_small_bert).",
    ),
    ("v1base_released", "v1 Base", "The released F5-TTS v1 Base (Emilia ZH+EN, 1.25M updates), for reference."),
]
TYPES = {"R": "read", "T": "said in the scene", "S": "improvised"}
BINS = (("short", 0, 2), ("medium", 2, 4), ("long", 4, 10))


def gender(spk):
    return "female" if spk.startswith("F") else "male"


def pick_fixed(info, rng, n):
    """n (speaker, sentence, type) groups present in all four emotions; one target per emotion each."""
    cands = [k for k, d in info.groupby(["speaker", "sentence", "type"]) if set(d.emotion) == set(EMOS)]
    cands = [cands[i] for i in rng.permutation(len(cands))]
    chosen = []
    # later passes drop the R/T balance, then the gender balance, where the set cannot meet them (few male
    # speakers have a sentence in all four emotions)
    for relax in (0, 1, 2):
        for spk, sent, typ in cands:
            if len(chosen) == n or (spk, sent, typ) in chosen:
                continue
            if spk in {c[0] for c in chosen} or sent in {c[1] for c in chosen}:
                continue
            if relax < 2 and sum(gender(c[0]) == gender(spk) for c in chosen) >= n // 2:
                continue
            if relax < 1 and sum(c[2] == typ for c in chosen) >= n // 2:
                continue
            chosen.append((spk, sent, typ))
    groups = []
    for spk, sent, typ in chosen:
        d = info[(info.speaker == spk) & (info.sentence == sent) & (info.type == typ)]
        rows = [d[d.emotion == e].iloc[rng.integers((d.emotion == e).sum())] for e in EMOS]
        groups.append((spk, typ, rows))
    return groups


def pick_improv(info, rng, per_emotion):
    """per_emotion targets per emotion, spread over speakers, genders and lengths."""
    quota = {b: per_emotion // len(BINS) + (i < per_emotion % len(BINS)) for i, (b, _, _) in enumerate(BINS)}
    overall = Counter()
    out = {}
    for emo in ("angry", "happy", "sad", "neutral"):
        d = info[info.emotion == emo]
        d = d.iloc[rng.permutation(len(d))]
        chosen, spk, g, b = [], set(), Counter(), Counter()
        for _, r in d.iterrows():
            bin_ = next((name for name, lo, hi in BINS if lo <= r.duration < hi), None)
            if bin_ is None or b[bin_] >= quota[bin_] or r.speaker in spk or overall[r.speaker] >= 3:
                continue
            if g[r.gender] >= per_emotion // 2:
                continue
            chosen.append(r)
            spk.add(r.speaker)
            g[r.gender] += 1
            b[bin_] += 1
            overall[r.speaker] += 1
            if len(chosen) == per_emotion:
                break
        out[emo] = sorted(chosen, key=lambda r: r.duration)
    return out


def e2v_label(scores, emo):
    """Top emotion among the four and its share of the four scores; whether it is the row's emotion."""
    p = {e: float(scores[i]) for e, i in E2V_IDX.items()}
    top = max(p, key=p.get)
    return {"top": top, "p": round(p[top] / sum(p.values()), 2), "ok": top == emo}


def encode(src, dst):
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            src,
            "-ac",
            "1",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "64k",
            dst,
        ],
        check=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/repro/demo/msp_improv")
    ap.add_argument("--groups", type=int, default=8, help="fixed set: (speaker, sentence) groups")
    ap.add_argument("--per-emotion", type=int, default=6, help="improv set: targets per emotion")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    jobs, sections = [], []
    for set_ in ("fixed", "improv"):
        info = pd.read_csv(f"{DATA}/{set_}/info.tsv", sep="\t", keep_default_na=False)
        meta = {
            u: (pt, tt)
            for u, pt, _, tt in (line.rstrip("\n").split("|")[:4] for line in open(f"{DATA}/{set_}/meta.lst"))
        }
        per = pickle.load(open(f"{ANA}/{set_}/per_utterance.pkl", "rb"))
        text_group = per["group"]
        ceiling = np.mean([v["acc4"] for v in per["per"]["ground_truth"].values()])
        emo = {n: pickle.load(open(f"{ANA}/{set_}/emotion/{n}.pkl", "rb")) for n, _, _ in SYSTEMS}
        emo["prompt"] = pickle.load(open(f"{ANA}/{set_}/emotion/prompt.pkl", "rb"))
        emo["real"] = pickle.load(open(f"{ANA}/{set_}/emotion/target.pkl", "rb"))

        def row(r):
            utt = r.utt
            srcs = [("prompt", f"{DATA}/{set_}/prompt-wavs/{utt}.wav"), ("real", f"{DATA}/{set_}/wavs/{utt}.wav")]
            srcs += [(n, f"{GEN}/{n}/{set_}/{utt}.wav") for n, _, _ in SYSTEMS]
            clips = []
            for role, src in srcs:
                rel = f"audio/{set_}/{utt.removeprefix('MSP-IMPROV-')}-{role}.mp3"
                jobs.append((src, f"{args.out}/{rel}"))
                clips.append(
                    {"src": rel, "dur": round(sf.info(src).duration, 1), **e2v_label(emo[role][utt][0], r.emotion)}
                )
            prompt_text, target_text = meta[utt]
            return {
                "utt": utt,
                "emotion": r.emotion,
                "speaker": r.speaker,
                "gender": r.gender,
                "type": TYPES[r.type],
                "text": target_text,
                "prompt_text": prompt_text,
                "text_group": text_group[utt],
                "clips": clips,
            }

        if set_ == "fixed":
            groups = []
            for spk, typ, rows in pick_fixed(info, rng, args.groups):
                texts = Counter(meta[r.utt][1] for r in rows)
                groups.append(
                    {
                        "title": texts.most_common(1)[0][0],
                        "meta": f"{spk} · {gender(spk)} · {TYPES[typ]} ({typ})",
                        "rows": [row(r) for r in rows],
                    }
                )
            sections.append(
                {
                    "id": "fixed",
                    "title": "Fixed text: one sentence, four emotions",
                    "intro": "One of the 20 fixed target sentences, spoken by one speaker in all four emotions. The words "
                    "are the same in every row, so any difference between rows comes from the prompt. Each row's "
                    "prompt is the same speaker in that emotion saying other fixed sentences.",
                    "groups": groups,
                    "ceiling": ceiling,
                    "n": len(info),
                }
            )
        else:
            picked = pick_improv(info, rng, args.per_emotion)
            sections.append(
                {
                    "id": "improv",
                    "title": "Improvised content",
                    "intro": "Lines the actors made up inside the emotional scenes, transcribed with Whisper large-v3. The "
                    "prompt is the same speaker in the same emotion reading fixed sentences, so the target's words "
                    "often carry the emotion too (text group: congruent, neutral text, or conflicting).",
                    "groups": [
                        {"title": e.capitalize(), "emotion": e, "rows": [row(r) for r in rs]}
                        for e, rs in picked.items()
                    ],
                    "ceiling": ceiling,
                    "n": len(info),
                }
            )

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: encode(*j), jobs))
    data = {
        "systems": [{"name": n, "label": lab, "desc": desc} for n, lab, desc in SYSTEMS],
        "sections": sections,
        "seed": args.seed,
    }
    os.makedirs(args.out, exist_ok=True)
    # script content is raw text (no entity decoding): only "</" must not appear
    page = PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    open(f"{args.out}/index.html", "w").write(page)
    n_rows = sum(len(g["rows"]) for s in sections for g in s["groups"])
    print(f"{args.out}/index.html: {n_rows} targets, {len(jobs)} clips")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MSP-IMPROV listening samples</title>
<style>
:root {
  --bg: #f7f6f3; --surface: #ffffff; --ink: #1c1b19; --ink2: #504e49; --muted: #85827a; --line: #e4e1da;
  --btn: #f1efea; --btn-hover: #e8e5de; --accent: #2563c9; --ok: #1f7a46;
  --angry: #c8433a; --happy: #c48a0c; --sad: #3a6fbf; --neutral: #7d7a72;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #141413; --surface: #1c1c1a; --ink: #edebe6; --ink2: #bdb9b0; --muted: #8d897f; --line: #302f2b;
    --btn: #262522; --btn-hover: #2f2e2a; --accent: #7aa7f0; --ok: #5cc48a;
    --angry: #ec6f66; --happy: #e2b03f; --sad: #74a2ea; --neutral: #a8a49b;
  }
}
:root[data-theme="dark"] {
  --bg: #141413; --surface: #1c1c1a; --ink: #edebe6; --ink2: #bdb9b0; --muted: #8d897f; --line: #302f2b;
  --btn: #262522; --btn-hover: #2f2e2a; --accent: #7aa7f0; --ok: #5cc48a;
  --angry: #ec6f66; --happy: #e2b03f; --sad: #74a2ea; --neutral: #a8a49b;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 1.7rem; line-height: 1.2; margin: 0 0 8px; letter-spacing: -0.01em; }
h2 { font-size: 1.25rem; margin: 48px 0 6px; }
h3 { font-size: 1.05rem; margin: 0; font-weight: 600; }
p { margin: 0 0 10px; color: var(--ink2); max-width: 72ch; }
.lede { font-size: 1.02rem; }
dl.cols { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px; margin: 18px 0; max-width: 820px; }
dl.cols dt { font-weight: 600; }
dl.cols dd { margin: 0; color: var(--ink2); }
nav { display: flex; gap: 16px; flex-wrap: wrap; margin: 14px 0 0; }
nav a { color: var(--accent); text-decoration: none; }
nav a:hover { text-decoration: underline; }
.group { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; margin: 18px 0; padding: 16px 16px 8px; }
.group > header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 14px; margin-bottom: 10px; }
.group > header .meta { color: var(--muted); font-size: 0.9rem; }
.scroll { overflow-x: auto; }
.grid { min-width: 680px; }
.row { display: grid; grid-template-columns: 9.5rem repeat(6, minmax(5.4rem, 1fr)); gap: 8px; align-items: start; padding: 10px 0 4px; }
.row.head { padding: 0 0 6px; border-bottom: 1px solid var(--line); color: var(--muted); font-size: 0.82rem; font-weight: 600; }
.row.head span[title] { cursor: help; text-decoration: underline dotted; text-underline-offset: 3px; }
.item { border-bottom: 1px solid var(--line); padding-bottom: 8px; }
.item:last-child { border-bottom: 0; }
.lab { font-size: 0.85rem; color: var(--ink2); line-height: 1.35; }
.lab .sub { display: block; color: var(--muted); font-size: 0.8rem; margin-top: 2px; }
.chip { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; color: var(--ink); text-transform: capitalize; }
.chip::before { content: ""; width: 9px; height: 9px; border-radius: 50%; background: var(--c); }
.chip.angry { --c: var(--angry); } .chip.happy { --c: var(--happy); } .chip.sad { --c: var(--sad); } .chip.neutral { --c: var(--neutral); }
.texts { font-size: 0.88rem; margin: 2px 0 0 calc(9.5rem + 8px); color: var(--ink2); }
.texts q { color: var(--ink); }
.texts .p { color: var(--muted); }
button.clip { position: relative; display: flex; align-items: center; gap: 7px; width: 100%; height: 36px; padding: 0 10px; border: 1px solid var(--line); border-radius: 7px; background: var(--btn); color: var(--ink); font: inherit; font-size: 0.85rem; cursor: pointer; overflow: hidden; }
button.clip:hover { background: var(--btn-hover); }
button.clip:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
button.clip::after { content: ""; position: absolute; left: 0; bottom: 0; height: 3px; width: calc(var(--p, 0) * 100%); background: var(--accent); }
button.clip svg { width: 12px; height: 12px; fill: currentColor; flex: none; }
button.clip .pause { display: none; }
button.clip.playing { border-color: var(--accent); }
button.clip.playing .play { display: none; }
button.clip.playing .pause { display: block; }
.e2v { display: block; font-size: 0.76rem; color: var(--muted); margin-top: 3px; white-space: nowrap; }
.e2v.ok { color: var(--ok); font-weight: 600; }
.note { font-size: 0.85rem; color: var(--muted); }
footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--line); font-size: 0.85rem; color: var(--muted); }
code { font-size: 0.85em; }
@media (max-width: 720px) {
  h1 { font-size: 1.4rem; }
  /* the grid scrolls sideways inside its card; keep the row label and the texts in view */
  .row > :first-child { position: sticky; left: 0; z-index: 1; align-self: stretch; background: var(--surface); }
  .texts { position: sticky; left: 0; margin-left: 0; max-width: calc(100vw - 66px); }
}
</style>
</head>
<body>
<main>
<h1>MSP-IMPROV listening samples</h1>
<p class="lede">Emotional zero-shot TTS on MSP-IMPROV. Each target is a real recording by an actor. The model gets a
prompt from the <em>same speaker in the same emotion</em> saying something else, plus the target text. The question is
whether the output carries that emotion.</p>
<dl class="cols" id="legend"></dl>
<p class="note">Generation: EMA weights, seed 0, 32 Euler steps, CFG 2, sway sampling −1. The output length comes from
F5-TTS's prompt-rate rule (no duration model). Under each clip: emotion2vec+ large's top class among angry, happy, sad
and neutral, with its share of those four. It is green when it matches the row's emotion. <span id="ceil"></span></p>
<p class="note">Rows were drawn at random (seed <span id="seed"></span>) under coverage rules: distinct speakers and
sentences, both genders, read and in-scene clips, short to long lines. They were not chosen by how they sound.</p>
<nav id="nav"></nav>
<div id="sections"></div>
<footer>Built by <code>repro/make_msp_demo.py</code> from the test sets of <code>repro/build_msp_emo.py</code> and the
outputs of <code>repro/msp_emo_eval.sbatch</code>. Audio from the MSP-IMPROV corpus (Busso et al., 2017) and voices
cloned from it, used under the corpus license.</footer>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
(() => {
  const D = JSON.parse(document.getElementById("data").textContent);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
    }
    for (const k of kids) if (k != null) n.append(k);
    return n;
  };
  const ICON = '<svg class="play" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1l9 5-9 5z"/></svg>' +
               '<svg class="pause" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1h3v10H2zM7 1h3v10H7z"/></svg>';
  const COLS = [{ label: "Prompt", desc: "Same speaker, same emotion, other words (1-3 clips joined, at least 3 s of speech). The model hears this." },
                { label: "Real", desc: "The actual target recording." },
                ...D.systems];

  const legend = document.getElementById("legend");
  for (const c of COLS) legend.append(el("dt", { text: c.label }), el("dd", { text: c.desc }));
  document.getElementById("seed").textContent = D.seed;
  document.getElementById("ceil").textContent = "On the real recordings it matches " +
    D.sections.map(s => `${Math.round(s.ceiling)}% of the ${s.id} set (${s.n} targets)`).join(" and ") +
    ", so it is a rough guide.";

  const nav = document.getElementById("nav");
  const root = document.getElementById("sections");
  for (const s of D.sections) {
    nav.append(el("a", { href: "#" + s.id, text: s.title }));
    const sec = el("section", { id: s.id }, el("h2", { text: s.title }), el("p", { text: s.intro }));
    for (const g of s.groups) {
      const head = el("header", {}, s.id === "fixed" ? el("h3", { text: "“" + g.title + "”" })
                                                     : el("h3", {}, el("span", { class: "chip " + g.emotion, text: g.title })));
      if (g.meta) head.append(el("span", { class: "meta", text: g.meta }));
      const grid = el("div", { class: "grid", role: "table" });
      const hr = el("div", { class: "row head", role: "row" }, el("span", { role: "columnheader", text: s.id === "fixed" ? "Emotion" : "Target" }));
      for (const c of COLS) hr.append(el("span", { role: "columnheader", title: c.desc, text: c.label }));
      grid.append(hr);
      for (const r of g.rows) {
        const lab = s.id === "fixed"
          ? el("div", { class: "lab", role: "cell" }, el("span", { class: "chip " + r.emotion, text: r.emotion }),
               el("span", { class: "sub", text: "text: " + r.text_group }))
          : el("div", { class: "lab", role: "cell" }, `${r.speaker} · ${r.gender}`,
               el("span", { class: "sub", text: `${r.clips[1].dur} s · text: ${r.text_group}` }));
        const row = el("div", { class: "row", role: "row" }, lab);
        r.clips.forEach((c, i) => {
          const b = el("button", { class: "clip", type: "button", "data-src": c.src, "aria-pressed": "false",
                                   "aria-label": `Play ${COLS[i].label}, ${r.emotion}, ${c.dur} seconds` });
          b.innerHTML = ICON;
          b.append(el("span", { text: c.dur + " s" }));
          row.append(el("div", { role: "cell" }, b,
            el("span", { class: "e2v" + (c.ok ? " ok" : ""), title: "emotion2vec+ top class among the four emotions",
                         text: `${c.top} ${Math.round(c.p * 100)}%` })));
        });
        const texts = el("div", { class: "texts" });
        if (s.id === "fixed") {
          if (r.text !== g.title) texts.append(el("div", {}, "says: ", el("q", { text: r.text })));
          texts.append(el("div", { class: "p" }, "prompt: ", el("q", { text: r.prompt_text })));
        } else {
          texts.append(el("div", {}, el("q", { text: r.text })),
                       el("div", { class: "p" }, "prompt: ", el("q", { text: r.prompt_text })));
        }
        grid.append(el("div", { class: "item" }, row, texts));
      }
      sec.append(el("div", { class: "group" }, head, el("div", { class: "scroll" }, grid)));
    }
    root.append(sec);
  }

  const audio = new Audio();
  let cur = null;
  const mark = on => { if (cur) { cur.classList.toggle("playing", on); cur.setAttribute("aria-pressed", String(on)); } };
  document.addEventListener("click", e => {
    const b = e.target.closest("button.clip");
    if (!b) return;
    if (b === cur) { audio.paused ? audio.play() : audio.pause(); return; }
    if (cur) { mark(false); cur.style.setProperty("--p", 0); }
    cur = b;
    audio.src = b.dataset.src;
    audio.play();
  });
  audio.addEventListener("play", () => mark(true));
  audio.addEventListener("pause", () => mark(false));
  audio.addEventListener("ended", () => { mark(false); if (cur) cur.style.setProperty("--p", 0); });
  audio.addEventListener("timeupdate", () => { if (cur && audio.duration) cur.style.setProperty("--p", audio.currentTime / audio.duration); });
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
