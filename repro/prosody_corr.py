"""Word-level prosody correlation between synthesized speech and the human recording of the same sentence.

Adapts the metric of Xie et al., "Is Prosody Lost in Translation? Fine-Grained Cross-Lingual Prosody Similarity
Across Languages" (Findings of EMNLP 2026, arXiv 2608.27848) from source/translation pairs to TTS output vs. ground
truth on Seed-TTS test-en (the ground truth is the same speaker reading the same sentence):

- F0: pYAAPT (35 ms frames, 10 ms hop, 60-400 Hz); unvoiced frames linearly interpolated (pYAAPT samp_interp), as in
  the paper. Utterances whose F0 track is constant are excluded, as in the paper.
- Energy: librosa RMS on the same 35 ms / 10 ms frames.
- Both normalized by their utterance mean (as in the paper; Spearman is unaffected).
- Units: words. The paper word-aligns two languages (whisper_timestamped + FastAlign); here both sides say the same
  words, so the normalized target transcript is force-aligned (torchaudio MMS_FA) on both recordings and word i
  pairs with word i. Each word is resampled to 20 points (frames_per_word = 20) and the words are concatenated;
  pauses between words are not part of any unit.
- Score: zero-lag Spearman rho between the two sequences, per utterance, then the mean over utterances.
- Shuffled baseline: the synthesized side's word contours (20-point blocks) are randomly reordered within the
  utterance (20 permutations, mean rho).
- Uncertainty: bootstrap over utterances clustered by prompt speaker (the analogue of the paper's episode
  clustering), 10,000 resamples; paired system comparisons use per-utterance differences + Wilcoxon.

Usage:
  python repro/prosody_corr.py --out results/_analysis/prosody NAME=WAV_DIR [NAME=WAV_DIR ...]
Per-system features are cached in OUT/cache/NAME.pkl, so adding systems later only computes the new ones.
"""

import argparse
import os
import pickle
import re
import sys
import zlib
from multiprocessing import get_context

import numpy as np
import soundfile as sf


SR = 16000
FRAMES_PER_WORD = 20
N_SHUFFLE = 20
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTSET = f"{ROOT}/data/seedtts_testset/en"


def words_of(text):
    """Target transcript -> MMS_FA word list: lowercase, curly quotes straightened, only a-z and ' kept."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    words = re.sub(r"[^a-z']+", " ", text).split()
    return [w for w in (w.strip("'") for w in words) if w]


def load16k(path):
    import librosa

    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)
    return librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y


def contours(path):
    """(f0_times, f0, en_times, energy), each normalized by its utterance mean; None if F0 is constant/unusable."""
    import amfm_decompy.basic_tools as basic
    import amfm_decompy.pYAAPT as pYAAPT
    import librosa

    try:
        y = load16k(path)
        p = pYAAPT.yaapt(basic.SignalObj(y, SR), frame_length=35.0, frame_space=10.0, f0_min=60.0, f0_max=400.0)
        f0 = np.asarray(p.samp_interp, dtype=np.float64)
        f0_t = np.asarray(p.frames_pos, dtype=np.float64) / SR
        if len(f0) < 2 or np.allclose(f0, f0[0]) or f0.mean() <= 0:
            return None
        en = librosa.feature.rms(y=y, frame_length=560, hop_length=160, center=True)[0].astype(np.float64)
        en_t = librosa.frames_to_time(np.arange(len(en)), sr=SR, hop_length=160)
        if en.mean() <= 0:
            return None
        return f0_t, f0 / f0.mean(), en_t, en / en.mean()
    except Exception as e:  # noqa: BLE001 - reported per utterance, never silently substituted
        return f"error: {type(e).__name__}: {e}"


class Aligner:
    def __init__(self, device):
        import torch
        import torchaudio

        self.torch, self.device = torch, device
        bundle = torchaudio.pipelines.MMS_FA
        self.model = bundle.get_model(with_star=False).to(device).eval()
        self.tokenizer, self.aligner = bundle.get_tokenizer(), bundle.get_aligner()

    def __call__(self, path, words):
        """[(start_s, end_s)] per word, or an error string."""
        try:
            y = self.torch.from_numpy(load16k(path))[None].to(self.device)
            with self.torch.inference_mode():
                emission, _ = self.model(y)
                spans = self.aligner(emission[0].cpu(), self.tokenizer(words))
            ratio = y.shape[1] / emission.shape[1] / SR
            return [(s[0].start * ratio, s[-1].end * ratio) for s in spans]
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {e}"


def word_sequences(cont, spans):
    """Each word resampled to FRAMES_PER_WORD points: arrays [n_words, FRAMES_PER_WORD] for F0 and energy."""
    f0_t, f0, en_t, en = cont
    pts = np.stack([np.linspace(a, b, FRAMES_PER_WORD) for a, b in spans])
    return np.interp(pts, f0_t, f0), np.interp(pts, en_t, en)


def features(name, wav_dir, items, out, aligner, workers):
    cache = f"{out}/cache/{name}.pkl"
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    paths = [f"{wav_dir}/{utt}.wav" for utt, _ in items]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"{name}: {len(missing)} of {len(paths)} wavs missing in {wav_dir}, e.g. {missing[0]}")
    with get_context("spawn").Pool(workers) as pool:  # no fork after CUDA init
        conts = pool.map(contours, paths, chunksize=8)
    feats = {}
    for (utt, words), path, cont in zip(items, paths, conts):
        if cont is None or isinstance(cont, str):
            feats[utt] = cont or "constant F0"
            continue
        spans = aligner(path, words)
        feats[utt] = spans if isinstance(spans, str) else word_sequences(cont, spans)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    pickle.dump(feats, open(cache + ".tmp", "wb"))
    os.replace(cache + ".tmp", cache)
    bad = sum(isinstance(v, str) for v in feats.values())
    print(f"{name}: {len(feats) - bad} utterances with features, {bad} excluded", flush=True)
    return feats


def spearman(a, b):
    from scipy.stats import spearmanr

    return spearmanr(a, b).statistic


def score(ref, syn, utt):
    """Aligned and shuffled-baseline Spearman rho for F0 and energy."""
    rng = np.random.default_rng(zlib.crc32(utt.encode()))  # same permutations in every run and system
    out = {}
    for k, (r, s) in (("f0", (ref[0], syn[0])), ("energy", (ref[1], syn[1]))):
        out[k] = spearman(r.ravel(), s.ravel())
        perms = [rng.permutation(len(s)) for _ in range(N_SHUFFLE)]
        out[f"{k}_shuffled"] = float(np.mean([spearman(r.ravel(), s[p].ravel()) for p in perms]))
    return out


def cluster_boot(values, clusters, n=10000, seed=0):
    """Mean and 95% CI, resampling prompt speakers (clusters) with replacement."""
    rng = np.random.default_rng(seed)
    ids = sorted(set(clusters))
    groups = {c: [] for c in ids}
    for v, c in zip(values, clusters):
        groups[c].append(v)
    sums = np.array([np.sum(groups[c]) for c in ids])
    cnts = np.array([len(groups[c]) for c in ids])
    pick = rng.integers(0, len(ids), (n, len(ids)))
    means = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(np.mean(values)), *np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+", help="NAME=WAV_DIR")
    ap.add_argument("--out", default=f"{ROOT}/results/_analysis/prosody")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--compare", nargs="*", default=[], help="A:B pairs for paired comparisons")
    ap.add_argument("--limit", type=int, default=0, help="first N utterances only (testing; uses no cache)")
    ap.add_argument("--testset", default=TESTSET, help="Seed-TTS-layout test set: meta.lst + wavs/<utt>.wav")
    ap.add_argument(
        "--speaker-field",
        type=int,
        default=0,
        help="'-'-separated field of the utt id that names the speaker (bootstrap cluster); "
        "0 for Seed-TTS test-en (prompt speaker), 3 for MSP-IMPROV",
    )
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import torch
    from scipy.stats import wilcoxon

    items, utt2spk = [], {}
    for ln in open(f"{args.testset}/meta.lst", encoding="utf-8").read().splitlines():
        utt, _, _, text = ln.split("|")
        items.append((utt, words_of(text)))
        utt2spk[utt] = utt.split("-")[args.speaker_field]
    systems = dict(s.split("=", 1) for s in args.systems)
    first = next(iter(systems.values()))
    items = [(u, w) for u, w in items if w and os.path.exists(f"{first}/{u}.wav")]  # the 1,088 evaluated pairs
    if args.limit:
        items, args.out = items[: args.limit], args.out + f"_test{args.limit}"

    aligner = Aligner("cuda" if torch.cuda.is_available() else "cpu")
    ref = features("ground_truth", f"{args.testset}/wavs", items, args.out, aligner, args.workers)

    per_utt, rows = {}, []
    for name, wav_dir in systems.items():
        syn = features(name, wav_dir, items, args.out, aligner, args.workers)
        scores = {}
        for utt, _ in items:
            r, s = ref.get(utt), syn.get(utt)
            if isinstance(r, str) or isinstance(s, str) or r is None or s is None:
                continue
            sc = score(r, s, utt)
            if not any(np.isnan(v) for v in sc.values()):  # rho undefined for a constant sequence
                scores[utt] = sc
        per_utt[name] = scores
        utts = sorted(scores)
        spk = [utt2spk[u] for u in utts]
        row = {"system": name, "n": len(utts)}
        for k in ("f0", "f0_shuffled", "energy", "energy_shuffled"):
            row[k] = cluster_boot([scores[u][k] for u in utts], spk)
        for k in ("f0", "energy"):
            row[f"{k}_wilcoxon_p"] = float(
                wilcoxon([scores[u][k] for u in utts], [scores[u][f"{k}_shuffled"] for u in utts]).pvalue
            )
        rows.append(row)
        print(
            f"{name}: F0 rho {row['f0'][0]:.3f} (shuffled {row['f0_shuffled'][0]:.3f}), "
            f"energy rho {row['energy'][0]:.3f} (shuffled {row['energy_shuffled'][0]:.3f}), n={len(utts)}",
            flush=True,
        )

    os.makedirs(args.out, exist_ok=True)
    pickle.dump(per_utt, open(f"{args.out}/per_utterance.pkl", "wb"))
    fmt = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"  # noqa: E731
    lines = [
        "# Word-level prosody correlation with the ground-truth recording (Seed-TTS test-en)",
        "",
        "Mean per-utterance Spearman rho, 95% CI from a prompt-speaker-clustered bootstrap. Shuffled = synthesized word",
        "contours randomly reordered within the utterance. See repro/prosody_corr.py for the method.",
        "",
        "| System | n | F0 rho | F0 shuffled | Energy rho | Energy shuffled |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['system']} | {r['n']} | {fmt(r['f0'])} | {fmt(r['f0_shuffled'])} | {fmt(r['energy'])} | "
        f"{fmt(r['energy_shuffled'])} |"
        for r in rows
    ]
    if args.compare:
        lines += [
            "",
            "## Paired differences (A - B), same utterances",
            "",
            "| A - B | n | dF0 rho [95% CI] | Wilcoxon p | dEnergy rho [95% CI] | Wilcoxon p |",
            "|---|---|---|---|---|---|",
        ]
        for pair in args.compare:
            a, b = pair.split(":")
            utts = sorted(set(per_utt[a]) & set(per_utt[b]))
            spk = [utt2spk[u] for u in utts]
            cells = []
            for k in ("f0", "energy"):
                d = [per_utt[a][u][k] - per_utt[b][u][k] for u in utts]
                cells += [fmt(cluster_boot(d, spk)), f"{wilcoxon(d).pvalue:.3g}"]
            lines.append(f"| {a} - {b} | {len(utts)} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    open(f"{args.out}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
