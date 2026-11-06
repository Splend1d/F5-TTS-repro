"""Plot the word-level F0 and energy contours that repro/prosody_corr.py correlates, for chosen utterances.

Each panel shows the ground-truth recording and one system's output on the metric's own axis: every word resampled
to 20 points, words in transcript order. Points where pYAAPT found the frame unvoiced (F0 filled in by its
interpolation) are drawn faint. The contours come from the metric's cache, so the plot shows exactly what was scored;
alignment and voicing are recomputed for the faint/solid split and checked against the cache.

Usage:
  python repro/prosody_examples.py --system bert300k=results/emilia_small_bert/update_300000/seedtts_test_en \
      --out results/_analysis/prosody/examples.png UTT [UTT ...]
"""

import argparse
import os
import pickle

import numpy as np
from prosody_corr import FRAMES_PER_WORD, ROOT, SR, TESTSET, Aligner, contours, load16k, word_sequences, words_of


GT_COLOR, SYN_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def voicing(path):
    """(frame times, voiced flag) from pYAAPT, with the settings prosody_corr.contours uses."""
    import amfm_decompy.basic_tools as basic
    import amfm_decompy.pYAAPT as pYAAPT

    p = pYAAPT.yaapt(basic.SignalObj(load16k(path), SR), frame_length=35.0, frame_space=10.0, f0_min=60.0, f0_max=400.0)
    return np.asarray(p.frames_pos, dtype=np.float64) / SR, np.asarray(p.samp_values) > 0


def utterance(path, words, aligner):
    cont = contours(path)
    spans = aligner(path, words)
    f0, en = word_sequences(cont, spans)
    t, v = voicing(path)
    pts = np.stack([np.linspace(a, b, FRAMES_PER_WORD) for a, b in spans])
    voiced = np.interp(pts, t, v.astype(float)) >= 0.5
    return f0, en, voiced, spans


def panel(ax, gt, syn, gt_voiced, syn_voiced, words, ylabel):
    n = len(words)
    x = np.arange(n * FRAMES_PER_WORD)
    for k in range(1, n):
        ax.axvline(k * FRAMES_PER_WORD - 0.5, color=GRID, lw=0.8, zorder=0)
    for seq, voiced, color, label in (
        (gt, gt_voiced, GT_COLOR, "Ground truth"),
        (syn, syn_voiced, SYN_COLOR, "Synthesized"),
    ):
        seq, voiced = seq.ravel(), voiced.ravel() if voiced is not None else None
        if voiced is None:
            ax.plot(x, seq, color=color, lw=1.6, label=label)
            continue
        ax.plot(x, seq, color=color, lw=1.0, alpha=0.35)
        ax.plot(x, np.where(voiced, seq, np.nan), color=color, lw=1.6, label=label)
    ax.axhline(1.0, color=INK2, lw=0.6, ls=(0, (2, 3)), zorder=0)
    ax.set_xticks(np.arange(n) * FRAMES_PER_WORD + (FRAMES_PER_WORD - 1) / 2, words, fontsize=9, color=INK)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=8, colors=INK2)
    ax.set_xlim(-0.5, n * FRAMES_PER_WORD - 0.5)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_facecolor(SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("utts", nargs="+")
    ap.add_argument("--system", required=True, help="NAME=WAV_DIR (NAME as in the prosody_corr cache)")
    ap.add_argument("--label", default=None, help="system name shown in the title")
    ap.add_argument("--prosody", default=f"{ROOT}/results/_analysis/prosody")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name, wav_dir = args.system.split("=", 1)
    meta = {
        ln.split("|")[0]: ln.split("|")[3] for ln in open(f"{TESTSET}/meta.lst", encoding="utf-8").read().splitlines()
    }
    scores = pickle.load(open(f"{args.prosody}/per_utterance.pkl", "rb"))[name]
    cache = {k: pickle.load(open(f"{args.prosody}/cache/{k}.pkl", "rb")) for k in ("ground_truth", name)}
    aligner = Aligner("cpu")

    fig, axes = plt.subplots(2 * len(args.utts), 1, figsize=(11, 3.1 * 2 * len(args.utts)), facecolor=SURFACE)
    for i, utt in enumerate(args.utts):
        words = words_of(meta[utt])
        gt = utterance(f"{TESTSET}/wavs/{utt}.wav", words, aligner)
        syn = utterance(f"{wav_dir}/{utt}.wav", words, aligner)
        for key, got in (("ground_truth", gt), (name, syn)):
            # pYAAPT differs by ~1e-3 between machines; a different alignment would differ far more
            for a, b in zip(cache[key][utt], got[:2]):
                if not np.allclose(a, b, atol=0.02):
                    raise SystemExit(f"{utt}: recomputed {key} contour differs from the prosody_corr cache")
        gt, syn = (*cache["ground_truth"][utt], *gt[2:]), (*cache[name][utt], *syn[2:])
        sc = scores[utt]
        dur = lambda spans: spans[-1][1] - spans[0][0]  # noqa: E731
        head = (
            f"“{meta[utt]}”\n{args.label or name} vs ground truth   •   "
            f"F0 ρ = {sc['f0']:.2f} (shuffled {sc['f0_shuffled']:.2f})   •   "
            f"energy ρ = {sc['energy']:.2f} (shuffled {sc['energy_shuffled']:.2f})   •   "
            f"speech {dur(gt[3]):.1f} s GT / {dur(syn[3]):.1f} s synth"
        )
        ax_f0, ax_en = axes[2 * i], axes[2 * i + 1]
        ax_f0.set_title(head, fontsize=10, color=INK, loc="left", pad=8)
        panel(ax_f0, gt[0], syn[0], gt[2], syn[2], words, "F0 / utterance mean")
        panel(ax_en, gt[1], syn[1], None, None, words, "RMS energy / utterance mean")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9, frameon=False, ncols=2, labelcolor=INK2)
    fig.text(
        0.01,
        0.003,
        "Each word resampled to 20 points (the metric's units). Faint F0 = unvoiced frames filled by "
        "pYAAPT interpolation. Dotted line = utterance mean.",
        fontsize=8,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.015, 1, 0.985), h_pad=1.2)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    print(f"wrote {args.out}")
    for utt in args.utts:
        print(f"  {utt}: GT {TESTSET}/wavs/{utt}.wav | synth {wav_dir}/{utt}.wav")


if __name__ == "__main__":
    main()
