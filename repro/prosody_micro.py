"""Macro and micro word-level prosody correlation from the feature cache of repro/prosody_corr.py.

- macro: one Spearman rho per utterance over its points (n_words x 20), then the mean over utterances (what
  prosody_corr.py reports).
- micro: the points of all utterances pooled into one Spearman rho per system. Contours are divided by their
  utterance mean, but the size of the swings around that mean still differs between utterances, so micro also
  rewards matching each utterance's range, while macro rewards only the shape.

Both use the utterances prosody_corr.py scored for every system (its per_utterance.pkl: features on both sides and
rho defined), so paired differences compare the same points. 95% CIs:
bootstrap over speakers. A resample repeats a speaker's points, so micro rho is computed with per-point weights
(tied average ranks and weighted Pearson), which equals Spearman on the resampled points.

Usage:
  python repro/prosody_micro.py --prosody-dir results/_analysis/msp_emo/fixed/prosody --speaker-field 3 \
    small250k colbert250k bert250k v1base_released --compare colbert250k:small250k bert250k:small250k
Writes <prosody-dir>/summary_micro.md.
"""

import argparse
import pickle

import numpy as np
from scipy.stats import spearmanr


def weighted_rho(order, x_sorted_ties, y_order, y_ties, w):
    """Spearman rho of points repeated w times (w=0 drops a point), from precomputed sort orders."""

    def ranks(order, tie_start, w):
        ws = w[order]
        c = np.cumsum(ws)
        # tied values share the average of their ranks: group by tie_start (index of each tie group's first member)
        group_sum = np.add.reduceat(ws, tie_start)
        group_end = c[np.r_[tie_start[1:] - 1, len(c) - 1]]
        avg = group_end - (group_sum - 1) / 2
        r = np.empty(len(order))
        r[order] = np.repeat(avg, np.diff(np.r_[tie_start, len(order)]))
        return r

    rx, ry = ranks(order, x_sorted_ties, w), ranks(y_order, y_ties, w)
    W = w.sum()
    mx, my = (w * rx).sum() / W, (w * ry).sum() / W
    dx, dy = rx - mx, ry - my
    return (w * dx * dy).sum() / np.sqrt((w * dx * dx).sum() * (w * dy * dy).sum())


def prep(v):
    """Sort order and tie-group starts of a 1-D array."""
    order = np.argsort(v, kind="stable")
    s = v[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    return order, starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prosody-dir", required=True)
    ap.add_argument("--speaker-field", type=int, default=0, help="speaker = utt.split('-')[field]")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--compare", nargs="*", default=[], help="A:B pairs, reported as A - B")
    ap.add_argument("systems", nargs="+")
    args = ap.parse_args()

    load = lambda n: pickle.load(open(f"{args.prosody_dir}/cache/{n}.pkl", "rb"))  # noqa: E731
    gt = load("ground_truth")
    syn = {n: load(n) for n in args.systems}
    # the utterances prosody_corr.py scored for every system (features on both sides, rho defined)
    scored = pickle.load(open(f"{args.prosody_dir}/per_utterance.pkl", "rb"))
    utts = sorted(set.intersection(*(set(scored[n]) for n in args.systems)))
    spk = np.array([u.split("-")[args.speaker_field] for u in utts])
    ids = sorted(set(spk))
    npts = np.array([gt[u][0].size for u in utts])
    pt_spk = np.repeat(np.searchsorted(ids, spk), npts)  # speaker index of every pooled point
    rng = np.random.default_rng(0)
    counts = np.stack(
        [np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)) for _ in range(args.n_boot)]
    )  # speaker multiplicities per resample, shared by all systems

    res = {}
    for n in args.systems:
        for k, feat in (("f0", 0), ("energy", 1)):
            per_utt = np.array([spearmanr(gt[u][feat].ravel(), syn[n][u][feat].ravel()).statistic for u in utts])
            x = np.concatenate([gt[u][feat].ravel() for u in utts])
            y = np.concatenate([syn[n][u][feat].ravel() for u in utts])
            micro = spearmanr(x, y).statistic
            ox, tx = prep(x)
            oy, ty = prep(y)
            spk_sum = np.bincount(np.searchsorted(ids, spk), weights=per_utt, minlength=len(ids))
            spk_cnt = np.bincount(np.searchsorted(ids, spk), minlength=len(ids))
            boot_macro = (counts * spk_sum).sum(1) / (counts * spk_cnt).sum(1)
            boot_micro = np.array([weighted_rho(ox, tx, oy, ty, c[pt_spk].astype(float)) for c in counts])
            res[n, k] = {"macro": per_utt.mean(), "micro": micro, "boot_macro": boot_macro, "boot_micro": boot_micro}
            print(f"{n} {k}: macro {per_utt.mean():.3f} micro {micro:.3f}", flush=True)

    ci = lambda b: np.percentile(b, [2.5, 97.5])  # noqa: E731
    fmt = lambda v, b: f"{v:.3f} [{ci(b)[0]:.3f}, {ci(b)[1]:.3f}]"  # noqa: E731
    sfmt = lambda v, b: f"{v:+.3f} [{ci(b)[0]:+.3f}, {ci(b)[1]:+.3f}]"  # noqa: E731
    lines = [
        f"# Macro vs micro prosody correlation: {args.prosody_dir}",
        "",
        f"{len(utts)} utterances scored for every system, {len(ids)} speakers, {int(npts.sum())} points "
        f"per system. Mean [95% CI], bootstrap over speakers ({args.n_boot} resamples). See repro/prosody_micro.py.",
        "",
        "| System | F0 macro | F0 micro | Energy macro | Energy micro |",
        "|---|---|---|---|---|",
    ]
    for n in args.systems:
        cells = [fmt(res[n, k][m], res[n, k][f"boot_{m}"]) for k in ("f0", "energy") for m in ("macro", "micro")]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    if args.compare:
        lines += [
            "",
            "Paired differences (A - B), same utterances and resamples:",
            "",
            "| A - B | F0 macro | F0 micro | Energy macro | Energy micro |",
            "|---|---|---|---|---|",
        ]
        for pair in args.compare:
            a, b = pair.split(":")
            cells = [
                sfmt(res[a, k][m] - res[b, k][m], res[a, k][f"boot_{m}"] - res[b, k][f"boot_{m}"])
                for k in ("f0", "energy")
                for m in ("macro", "micro")
            ]
            lines.append(f"| {a} - {b} | " + " | ".join(cells) + " |")
    open(f"{args.prosody_dir}/summary_micro.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
