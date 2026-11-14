#!/usr/bin/env python3
"""Threshold-free event-level metrics for jev-hard (2026-11-06), answering R1(b)/R2#2.
  python event-metrics.py --events <events.jsonl> <run> [<run> ...]
Per run and tier:
  evAP   event-level AP: positives = events scored by their max window, negatives = every window of the normal test
         videos (each a potential false alarm); threshold-free.
  evAUC  the matching AUC (event max vs normal windows).
  nLA    normalised area under the recall-vs-false-alarm curve for FA in [0.25, 10]/h on a log axis (0..1).
  split  event recall at 1 FA/h with tau chosen on HALF of the normal videos (odd index) and false alarms counted on
         the other half; two folds averaged. Removes the per-model test-set threshold selection.
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


def ap_auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    o = np.argsort(-s, kind="stable"); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    auc = float(np.trapezoid(tp / tp[-1], fp / fp[-1]))
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum()), auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    ev = [json.loads(l) for l in a.events.read_text().splitlines() if l.strip()]
    area = np.array([e["area_frac"] for e in ev]); t1, t2 = np.quantile(area, [1 / 3, 2 / 3])
    for e in ev:
        e["tier"] = "small" if e["area_frac"] < t1 else ("medium" if e["area_frac"] < t2 else "large")
    tiers = ("small", "medium", "large")
    print(f"{'run':26s} | " + " | ".join(f"{t}: evAP evAUC nLA split@1/h" for t in tiers))
    for n in a.runs:
        d = latest(n)
        man = {m["name"]: m for m in json.loads((d / "manifest.json").read_text())}
        rows = {}
        for r in map(json.loads, (d / "scores.jsonl").read_text().splitlines()):
            if r:
                rows.setdefault(r["video"], []).append(r)
        normal = sorted(v for v in rows if not man[v].get("events"))
        norm_scores = {v: np.array([r["score"] for r in rows[v]]) for v in normal}
        norm_hours = {v: sum(r["end_s"] - r["start_s"] for r in rows[v]) / 3600 for v in normal}
        all_norm = np.concatenate([norm_scores[v] for v in normal]); H = sum(norm_hours.values())
        evmax = {t: [] for t in tiers}
        for e in ev:
            m = man.get(e["video"])
            if m is None:
                continue
            fps = m["fps"]; e0, e1 = e["event"]
            ws = [r["score"] for r in rows.get(e["video"], []) if r["start_s"] * fps < e1 + 1 and r["end_s"] * fps > e0]
            if ws:
                evmax[e["tier"]].append(max(ws))
        line = f"{n[:26]:26s} |"
        srt = np.sort(all_norm)[::-1]
        for t in tiers:
            em = np.array(evmax[t])
            y = np.concatenate([np.ones(len(em)), np.zeros(len(all_norm))]); s = np.concatenate([em, all_norm])
            evap, evauc = ap_auc(y, s)
            fas = np.logspace(np.log10(0.25), np.log10(10), 25)
            rec = []
            for fa in fas:
                k = max(int(round(fa * H)), 1); tau = srt[k - 1]
                rec.append((em >= tau).mean())
            nla = float(np.trapezoid(rec, np.log10(fas)) / (np.log10(10) - np.log10(0.25)))
            # split protocol
            folds = []
            for parity in (0, 1):
                sel = [v for i, v in enumerate(normal) if i % 2 == parity]; oth = [v for i, v in enumerate(normal) if i % 2 != parity]
                ss = np.sort(np.concatenate([norm_scores[v] for v in sel]))[::-1]; hs = sum(norm_hours[v] for v in sel)
                k = max(int(round(1.0 * hs)), 1); tau = ss[k - 1]
                so = np.concatenate([norm_scores[v] for v in oth]); ho = sum(norm_hours[v] for v in oth)
                fa_other = (so >= tau).sum() / ho
                folds.append(((em >= tau).mean(), fa_other))
            line += f" {100*evap:5.1f} {100*evauc:5.1f} {nla:.3f} {100*np.mean([f[0] for f in folds]):4.0f}% (FA on held-out {np.mean([f[1] for f in folds]):.1f}/h) |"
        print(line)


if __name__ == "__main__":
    main()
