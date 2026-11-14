#!/usr/bin/env python3
"""Temporal-localisation analysis for the restructured paper (2026-11-06).
  python localisation.py --events <events.jsonl> --out <dir> [--firing <firing.jsonl>] run1 run2 ...
Writes:
  timelines.json   six anomalous videos (two per tier), per model the window scores, plus event spans (for F1)
  loc_stats.md     per model: within-video AUC, median over videos of (mean in-event - mean out-event score),
                   share of unlabelled windows >= 0.5, video-level AUC
  decay.md         unlabelled windows of anomalous videos grouped by distance (windows) to the nearest event
  prior.md         across anomalous videos: correlation of each video's mean unlabelled score with its mean labelled
                   score, and with the zero-shot model's mean unlabelled score (scene prior)
  tgm.md           if --firing given: unlabelled-window score by TGM hot-area quartile, anomalous vs normal videos
"""
import argparse, json
from pathlib import Path
import numpy as np

R = Path("results/pro6000")


def latest(n):
    if (R / n / "scores.jsonl").exists():
        return R / n
    ds = sorted(d for d in (R / n).iterdir() if (d / "scores.jsonl").exists() or (d / "merged" / "scores.jsonl").exists())
    d = ds[-1]
    return d if (d / "scores.jsonl").exists() else d / "merged"


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s, kind="stable"); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    return float(np.trapezoid(tp / tp[-1], fp / fp[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--firing", type=Path, default=None)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    ev = [json.loads(l) for l in a.events.read_text().splitlines() if l.strip()]
    area = np.array([e["area_frac"] for e in ev]); t1, t2 = np.quantile(area, [1 / 3, 2 / 3])
    tier_of_video = {}
    for e in ev:
        e["tier"] = "small" if e["area_frac"] < t1 else ("medium" if e["area_frac"] < t2 else "large")
        tier_of_video.setdefault(e["video"], set()).add(e["tier"])
    data = {}
    for n in a.runs:
        d = latest(n)
        man = {m["name"]: m for m in json.loads((d / "manifest.json").read_text())}
        rows = {}
        for r in map(json.loads, (d / "scores.jsonl").read_text().splitlines()):
            if r:
                rows.setdefault(r["video"], []).append(r)
        for v in rows:
            rows[v].sort(key=lambda r: r["window"])
        data[n] = (man, rows)
    man0, rows0 = data[a.runs[0]]
    anom = [v for v in rows0 if man0[v].get("events")]
    normal = [v for v in rows0 if not man0[v].get("events")]

    # ---- timelines: two videos per tier, single-tier videos with >= 20 windows, most windows first
    chosen = []
    for t in ("small", "medium", "large"):
        cand = [v for v in anom if tier_of_video.get(v) == {t} and len(rows0[v]) >= 20]
        cand.sort(key=lambda v: -len(rows0[v]))
        chosen += cand[:2]
    tl = {}
    for v in chosen:
        m = man0[v]; fps = m["fps"]
        tl[v] = {"tier": sorted(tier_of_video[v])[0], "fps": fps, "events_s": [[e0 / fps, (e1 + 1) / fps] for e0, e1 in m["events"]],
                 "models": {n: [[r["start_s"], r["score"]] for r in data[n][1][v]] for n in a.runs}}
    (a.out / "timelines.json").write_text(json.dumps(tl))

    # ---- loc_stats, decay, prior
    loc = ["| run | video-AUC | within-video AUC | median(in - out) | unlabelled >= 0.5 | labelled >= 0.5 |", "|---|---|---|---|---|---|"]
    decay_bins = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 10 ** 9)]
    decay = ["| run | " + " | ".join(f"dist {lo}" + (f"-{hi}" if hi != lo and hi < 10 ** 8 else ("+" if hi > 10 ** 8 else "")) for lo, hi in decay_bins) + " | labelled |", "|---|" + "---|" * (len(decay_bins) + 1)]
    prior = ["| run | corr(mean unlabelled, mean labelled) over videos | corr(mean unlabelled, zero-shot mean unlabelled) |", "|---|---|---|"]
    zs_unl = None
    for n in a.runs:
        man, rows = data[n]
        vy, vs = [], []; wy, ws = [], []; diffs = []; unl_all = []; lab_all = []
        bins = {b: [] for b in decay_bins}; per_video_unl = {}; per_video_lab = {}
        for v, rs in rows.items():
            s = np.array([r["score"] for r in rs]); y = np.array([r["label"] for r in rs], bool)
            vy.append(1 if man[v].get("events") else 0); vs.append(s.max())
            if not man[v].get("events"):
                continue
            wy += list(y.astype(int)); ws += list(s)
            if y.any() and (~y).any():
                diffs.append(s[y].mean() - s[~y].mean())
                per_video_unl[v] = s[~y].mean(); per_video_lab[v] = s[y].mean()
            unl_all += list(s[~y]); lab_all += list(s[y])
            idx_lab = np.where(y)[0]
            for i in np.where(~y)[0]:
                dist = int(np.min(np.abs(idx_lab - i))) if len(idx_lab) else 10 ** 9
                for lo, hi in decay_bins:
                    if lo <= dist <= hi:
                        bins[(lo, hi)].append(s[i])
        unl_all = np.array(unl_all); lab_all = np.array(lab_all)
        loc.append(f"| {n} | {100*auc(vy, vs):.1f} | {100*auc(wy, ws):.1f} | {np.median(diffs):+.2f} | {100*(unl_all >= 0.5).mean():.0f}% | {100*(lab_all >= 0.5).mean():.0f}% |")
        decay.append(f"| {n} | " + " | ".join(f"{np.mean(bins[b]):.2f} (n={len(bins[b])})" for b in decay_bins) + f" | {lab_all.mean():.2f} |")
        vids = sorted(per_video_unl)
        u = np.array([per_video_unl[v] for v in vids]); l = np.array([per_video_lab[v] for v in vids])
        if zs_unl is None:
            zs_unl = {v: per_video_unl[v] for v in vids}
        z = np.array([zs_unl.get(v, np.nan) for v in vids]); ok = ~np.isnan(z)
        prior.append(f"| {n} | {np.corrcoef(u, l)[0, 1]:.2f} | {np.corrcoef(u[ok], z[ok])[0, 1]:.2f} |")
    (a.out / "loc_stats.md").write_text("\n".join(loc) + "\n"); (a.out / "decay.md").write_text("\n".join(decay) + "\n"); (a.out / "prior.md").write_text("\n".join(prior) + "\n")
    print("\n".join(loc)); print(); print("\n".join(decay)); print(); print("\n".join(prior))

    # ---- tgm
    if a.firing and a.firing.exists():
        fr = {(x["video"], x["window"]): x["area"] for x in map(json.loads, a.firing.read_text().splitlines()) if x}
        qs = np.quantile(list(fr.values()), [0.25, 0.5, 0.75])
        out = ["| run | set | " + " | ".join(f"area Q{i+1}" for i in range(4)) + " |", "|---|---|---|---|---|---|"]
        for n in a.runs:
            man, rows = data[n]
            sc = {(r["video"], r["window"]): r for v, rs in rows.items() for r in rs}
            for name, cond in (("anomalous videos, unlabelled", lambda k, r: man[k[0]].get("events") and not r["label"]), ("normal videos", lambda k, r: not man[k[0]].get("events"))):
                groups = [[] for _ in range(4)]
                for k, ar in fr.items():
                    r = sc.get(k)
                    if r is None or not cond(k, r):
                        continue
                    q = int(np.searchsorted(qs, ar)); groups[q].append(r["score"])
                out.append(f"| {n} | {name} | " + " | ".join(f"{np.mean(g):.2f} (n={len(g)})" if g else "-" for g in groups) + " |")
        (a.out / "tgm.md").write_text(f"area quartile cuts: {qs}\n" + "\n".join(out) + "\n"); print(); print("\n".join(out))


if __name__ == "__main__":
    main()
