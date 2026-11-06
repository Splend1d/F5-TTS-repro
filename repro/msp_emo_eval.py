"""Emotion metrics and the summary report for the MSP-IMPROV test sets (repro/build_msp_emo.py).

Per target utterance and system:
- wer, sim, utmos       from the F5-TTS evaluators (_wer/_sim/_utmos_results.jsonl in the system's output dir)
- f0, energy            word-level prosody correlation with the ground truth (repro/prosody_corr.py, run first
                        with --out OUT/prosody --speaker-field 3)
- acc4                  emotion2vec+ large: argmax over {angry, happy, sad, neutral} equals the target emotion
                        (the scene emotion, which listeners also heard). Its value on the ground truth is the
                        classifier's ceiling on this set.
- emo_sim_gt/_prompt    cosine of emotion2vec+ utterance embeddings, output vs the ground-truth target / the prompt
- margin_f0/_energy     fixed set only. The same speaker said the same sentence in the other three emotions too, so
                        margin = rho(output, target-emotion recording) - mean rho(output, other-emotion recordings):
                        > 0 when the output's contour is specific to the target emotion, not just to the words.
                        ident_f0/_energy: the target-emotion recording correlates best (chance = 1 / (1 + #others)).
Text-emotion group of each target (j-hartmann/emotion-english-distilroberta-base on the target text, restricted to
anger/joy/sadness/neutral): congruent (text emotion = target emotion), neutral text (target emotional, text
neutral), conflicting (text says another emotion).

Means with 95% CIs from a bootstrap over speakers; paired system differences on the same utterances.

Usage:
  python repro/msp_emo_eval.py --set fixed NAME=GEN_DIR [...] --compare A:B [...]
  (a NAME called ground_truth is the real recording: it gets sim/utmos/acc4/emo_sim_prompt as ceilings)
"""

import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prosody_corr import Aligner, cluster_boot, features, spearman, words_of  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{ROOT}/data/msp_improv_emo"
E2V_IDX = {"angry": 0, "happy": 3, "neutral": 4, "sad": 6}  # emotion2vec+ 9-way label order
TEXT_MAP = {"anger": "angry", "joy": "happy", "sadness": "sad", "neutral": "neutral"}
EMOS = ("angry", "happy", "sad", "neutral")


def speaker(utt):
    return utt.split("-")[3]


def emotion_features(name, paths, cache):
    """utt -> (9 emotion2vec+ scores, utterance embedding)."""
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    import torch
    from funasr import AutoModel

    global _E2V
    if "_E2V" not in globals():
        _E2V = AutoModel(
            model="emotion2vec/emotion2vec_plus_large",
            hub="hf",
            disable_update=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    out = {}
    for utt, p in paths.items():
        r = _E2V.generate(p, granularity="utterance", extract_embedding=True, disable_pbar=True)[0]
        out[utt] = (np.asarray(r["scores"], dtype=np.float32), np.asarray(r["feats"], dtype=np.float32))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    pickle.dump(out, open(cache + ".tmp", "wb"))
    os.replace(cache + ".tmp", cache)
    print(f"emotion2vec {name}: {len(out)} files", flush=True)
    return out


def text_emotions(texts, cache):
    """text -> (emotion among EMOS, probabilities over the four)."""
    if os.path.exists(cache):
        return json.load(open(cache))
    from transformers import pipeline

    clf = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    out = {}
    uniq = sorted(set(texts))
    for t, scores in zip(uniq, clf(uniq, batch_size=64)):
        p = {TEXT_MAP[s["label"]]: s["score"] for s in scores if s["label"] in TEXT_MAP}
        z = sum(p.values())
        p = {k: v / z for k, v in p.items()}
        out[t] = [max(p, key=p.get), p]
    json.dump(out, open(cache, "w"), indent=0)
    return out


def margins(info, systems, prosody_dir, out_dir, workers):
    """Fixed set: emotion-specific prosody (see module doc). Returns {system: {utt: {...}}}."""
    clips = [
        r
        for r in csv.DictReader(open(f"{DATA}/clips.tsv"), delimiter="\t")
        if r["status"] == "kept" and r["type"] in "RT"
    ]
    others = {}
    for utt, i in info.items():
        w = words_of(i["text"])
        others[utt] = [
            c["clip"][:-4]
            for c in clips
            if (c["speaker"], c["sentence"], c["type"]) == (i["speaker"], i["sentence"], i["type"])
            and c["emotion"] != i["emotion"]
            and words_of(c["text"]) == w
        ]
    need = sorted({o for v in others.values() for o in v})
    text_of = {c["clip"][:-4]: c["text"] for c in clips}
    import torch

    aligner = Aligner("cuda" if torch.cuda.is_available() else "cpu")
    oth = features(
        "other_emotion_clips", f"{DATA}/cut", [(u, words_of(text_of[u])) for u in need], out_dir, aligner, workers
    )
    gt = pickle.load(open(f"{prosody_dir}/cache/ground_truth.pkl", "rb"))
    res = {}
    for name in systems:
        syn = pickle.load(open(f"{prosody_dir}/cache/{name}.pkl", "rb"))
        res[name] = {}
        for utt, os_ in others.items():
            s, g = syn.get(utt), gt.get(utt)
            os_ = [o for o in os_ if not isinstance(oth.get(o), str) and oth.get(o) is not None]
            if isinstance(s, str) or isinstance(g, str) or s is None or g is None or not os_:
                continue
            r = {}
            for k, key in ((0, "f0"), (1, "energy")):
                same = spearman(g[k].ravel(), s[k].ravel())
                diff = [spearman(oth[o][k].ravel(), s[k].ravel()) for o in os_]
                if np.isnan(same) or any(np.isnan(diff)):
                    break
                r[f"margin_{key}"] = same - float(np.mean(diff))
                r[f"ident_{key}"] = float(same > max(diff))
                r["n_others"] = len(diff)
            else:
                res[name][utt] = r
    # how far apart the real recordings of one sentence are across emotions (context for the margins)
    ref = []
    for utt, os_ in others.items():
        g = gt.get(utt)
        if isinstance(g, str) or g is None:
            continue
        for o in os_:
            if not isinstance(oth.get(o), str) and oth.get(o) is not None:
                ref.append(spearman(g[0].ravel(), oth[o][0].ravel()))
    return res, float(np.nanmean(ref)) if ref else float("nan")


def read_jsonl(path, key):
    out = {}
    if os.path.exists(path):
        for ln in open(path):
            if ln.startswith("{"):
                r = json.loads(ln)
                out[r["wav"]] = r[key]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+", help="NAME=GEN_DIR")
    ap.add_argument("--set", required=True, choices=["fixed", "improv"])
    ap.add_argument("--out", default=None, help="default results/_analysis/msp_emo/<set>")
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    out = args.out or f"{ROOT}/results/_analysis/msp_emo/{args.set}"
    d = f"{DATA}/{args.set}"
    info = {r["utt"]: r for r in csv.DictReader(open(f"{d}/info.tsv"), delimiter="\t")}
    systems = dict(s.split("=", 1) for s in args.systems)

    prompt_e = emotion_features("prompt", {u: f"{d}/prompt-wavs/{u}.wav" for u in info}, f"{out}/emotion/prompt.pkl")
    gt_e = emotion_features("target", {u: f"{d}/wavs/{u}.wav" for u in info}, f"{out}/emotion/target.pkl")
    sys_e = {
        n: emotion_features(
            n, {u: f"{g}/{u}.wav" for u in info if os.path.exists(f"{g}/{u}.wav")}, f"{out}/emotion/{n}.pkl"
        )
        for n, g in systems.items()
        if n != "ground_truth"
    }
    text_e = text_emotions([i["text"] for i in info.values()], f"{out}/text_emotion.json")
    group = {}
    for u, i in info.items():
        te = text_e[i["text"]][0]
        group[u] = "congruent" if te == i["emotion"] else "neutral text" if te == "neutral" else "conflicting"

    pros = pickle.load(open(f"{out}/prosody/per_utterance.pkl", "rb"))
    synth = [n for n in systems if n != "ground_truth"]
    marg, gt_cross = (
        margins(info, synth, f"{out}/prosody", f"{out}/margin", args.workers)
        if args.set == "fixed"
        else ({}, float("nan"))
    )

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    per = {}  # system -> utt -> metric -> value
    for n, g in systems.items():
        wer, sim, ut = (read_jsonl(f"{g}/_{k}_results.jsonl", k) for k in ("wer", "sim", "utmos"))
        per[n] = {}
        for u, i in info.items():
            m = {}
            if u in wer and n != "ground_truth":
                m["wer"] = 100 * wer[u]
            if u in sim:
                m["sim"] = sim[u]
            if u in ut:
                m["utmos"] = ut[u]
            if u in pros.get(n, {}):
                m["f0"], m["energy"] = pros[n][u]["f0"], pros[n][u]["energy"]
            e = gt_e[u] if n == "ground_truth" else sys_e[n].get(u)
            if e is not None:
                four = {k: e[0][j] for k, j in E2V_IDX.items()}
                m["acc4"] = 100.0 * (max(four, key=four.get) == i["emotion"])
                m["emo_sim_prompt"] = cos(e[1], prompt_e[u][1])
                if n != "ground_truth":
                    m["emo_sim_gt"] = cos(e[1], gt_e[u][1])
            m.update(marg.get(n, {}).get(u, {}))
            m.pop("n_others", None)
            if "ident_f0" in m:
                m["ident_f0"] *= 100
                m["ident_energy"] *= 100
            per[n][u] = m
    pickle.dump({"per": per, "group": group, "info": info}, open(f"{out}/per_utterance.pkl", "wb"))

    cols = ["wer", "sim", "utmos", "f0", "energy", "acc4", "emo_sim_gt", "emo_sim_prompt"]
    if args.set == "fixed":
        cols += ["margin_f0", "ident_f0", "margin_energy", "ident_energy"]
    fmt = {"wer": "{:.2f}", "acc4": "{:.1f}", "ident_f0": "{:.1f}", "ident_energy": "{:.1f}"}

    def agg(n, utts, k, ci=True):
        utts = [u for u in utts if k in per[n][u]]
        if not utts:
            return "-"
        f = fmt.get(k, "{:.3f}")
        if not ci:
            return f.format(np.mean([per[n][u][k] for u in utts]))
        m, lo, hi = cluster_boot([per[n][u][k] for u in utts], [speaker(u) for u in utts])
        return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]"

    emo_n = {e: sum(i["emotion"] == e for i in info.values()) for e in EMOS}
    grp_n = {g: sum(v == g for v in group.values()) for g in ("congruent", "neutral text", "conflicting")}
    L = [
        f"# MSP-IMPROV emotional zero-shot TTS: {args.set} set",
        "",
        f"{len(info)} targets, {len({i['speaker'] for i in info.values()})} speakers; per emotion {emo_n}; "
        f"text-emotion groups {grp_n}.",
        "wer in %, acc4 / ident in %. 95% CIs: bootstrap over speakers. See repro/msp_emo_eval.py for definitions.",
        "",
    ]
    L += ["## Overall", "", "| System | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    L += [f"| {n} | " + " | ".join(agg(n, list(info), k) for k in cols) + " |" for n in systems]
    if args.set == "fixed":
        L += [
            "",
            f"Mean F0 rho between the real recordings of the same sentence in two different emotions (same "
            f"speaker): {gt_cross:.3f}.",
        ]
    for title, key, cats in (
        ("By target emotion", "emotion", EMOS),
        ("By text-emotion group", "group", ("congruent", "neutral text", "conflicting")),
    ):
        L += ["", f"## {title} (means)", ""]
        for k in ("wer", "f0", "acc4", "emo_sim_gt") + (("margin_f0",) if args.set == "fixed" else ()):
            L += [f"**{k}**", "", "| System | " + " | ".join(cats) + " |", "|---" * (len(cats) + 1) + "|"]
            for n in systems:
                cells = []
                for c in cats:
                    us = [u for u in info if (info[u]["emotion"] if key == "emotion" else group[u]) == c]
                    cells.append(agg(n, us, k, ci=False))
                L.append(f"| {n} | " + " | ".join(cells) + " |")
            L.append("")
    if args.compare:
        L += [
            "## Paired differences (A - B), same utterances",
            "",
            "| A - B | " + " | ".join(cols) + " |",
            "|---" * (len(cols) + 1) + "|",
        ]
        for pair in args.compare:
            a, b = pair.split(":")
            cells = []
            for k in cols:
                us = [u for u in info if k in per[a][u] and k in per[b][u]]
                if not us:
                    cells.append("-")
                    continue
                f = fmt.get(k, "{:.3f}")
                m, lo, hi = cluster_boot([per[a][u][k] - per[b][u][k] for u in us], [speaker(u) for u in us])
                cells.append(f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]")
            L.append(f"| {a} - {b} | " + " | ".join(cells) + " |")
    open(f"{out}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
