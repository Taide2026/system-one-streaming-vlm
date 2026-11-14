#!/usr/bin/env python3
"""A1-A3 (2026-11-06): event-level metrics for jev-hard, false-alarm suppression rules, and within-event score structure.

  python a1a3.py --events <events.jsonl> --out <md path> [--fa 0.5,1,2,5] [--rules] run1 run2 ...

A1  Per run and tier: window AP (as jev-hard.py) and event recall at k false alarms per hour of normal test video
    (alarm = window with score >= tau; tau set so that normal videos produce exactly k*hours alarms; tau is per run).
A2  --rules: the same for post-hoc causal rules on the scores: EMA (0.7 new / 0.3 old), two consecutive windows
    (score = min of this and previous window), per-video adaptive (score minus the causal median of the video so far).
A3  Within-event structure per tier: median over events of (max window score), (median window score), fraction of
    event windows >= 0.5, and the share of events whose max >= 0.5 while median < 0.3 ("one window fires").
"""
from __future__ import annotations

import argparse
import json
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
    o = np.argsort(-s, kind="stable"); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    auc = float(np.trapezoid(tp / max(tp[-1], 1), fp / max(fp[-1], 1)))
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1)), auc


def apply_rule(byv, rule):
    out = {}
    for v, rs in byv.items():
        rs = sorted(rs, key=lambda r: r["window"])
        s = np.array([r["score"] for r in rs], float)
        if rule == "raw":
            t = s
        elif rule == "ema":
            t = s.copy()
            for i in range(1, len(s)):
                t[i] = 0.7 * s[i] + 0.3 * t[i - 1]
        elif rule == "two":
            t = s.copy()
            t[1:] = np.minimum(s[1:], s[:-1])
        elif rule == "adapt":
            t = s.copy()
            for i in range(1, len(s)):
                t[i] = s[i] - float(np.median(s[:i]))
            t[0] = s[0] - 0.5
        else:
            raise ValueError(rule)
        out[v] = [dict(r, score=float(x)) for r, x in zip(rs, t)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fa", default="0.5,1,2,5")
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    fas = [float(x) for x in a.fa.split(",")]
    ev = [json.loads(l) for l in a.events.read_text().splitlines() if l.strip()]
    area = np.array([e["area_frac"] for e in ev]); t1, t2 = np.quantile(area, [1 / 3, 2 / 3])
    for e in ev:
        e["tier"] = "small" if e["area_frac"] < t1 else ("medium" if e["area_frac"] < t2 else "large")
    tiers = ("small", "medium", "large")
    lines = [f"# A1-A3 on {a.events.name} ({len(ev)} events; terciles {t1:.4f} / {t2:.4f})", ""]
    lines.append("## A1 window AP and event recall at k false alarms per hour" + (" (with A2 rules)" if a.rules else ""))
    lines.append("| run | rule | " + " | ".join(f"{t} AP" for t in tiers) + " | all AP | " +
                 " | ".join(f"recall@{fa:g}/h s/m/l" for fa in fas) + " |")
    lines.append("|---|---|" + "---|" * (len(tiers) + 1 + len(fas)))
    a3 = ["", "## A3 within-event structure (raw scores)", "| run | tier | n events | median of max | median of median | mean frac windows >= 0.5 | one-window-fires share |", "|---|---|---|---|---|---|---|"]
    for n in a.runs:
        d = latest(n)
        man = {m["name"]: m for m in json.loads((d / "manifest.json").read_text())}
        byv = {}
        for r in map(json.loads, (d / "scores.jsonl").read_text().splitlines()):
            if r:
                byv.setdefault(r["video"], []).append(r)
        for rule in (["raw", "ema", "two", "adapt"] if a.rules else ["raw"]):
            rows = apply_rule(byv, rule)
            # tier window sets, standard convention
            tier_of = {}
            for e in ev:
                tier_of.setdefault(e["video"], []).append((e["event"][0], e["event"][1], e["tier"]))
            sub = {k: ([], []) for k in tiers + ("all",)}
            normal_scores, hours = [], 0.0
            for v, rs in rows.items():
                m = man[v]
                if not m.get("events"):
                    for k in sub:
                        sub[k][0].extend([0] * len(rs)); sub[k][1].extend(r["score"] for r in rs)
                    normal_scores += [r["score"] for r in rs]; hours += sum(r["end_s"] - r["start_s"] for r in rs) / 3600
                    continue
                fps = m["fps"]
                for r in rs:
                    w0, w1 = r["start_s"] * fps, r["end_s"] * fps
                    here = {t for e0, e1, t in tier_of.get(v, []) if w0 < e1 + 1 and w1 > e0}
                    sub["all"][0].append(r["label"]); sub["all"][1].append(r["score"])
                    if r["label"]:
                        for t in here:
                            sub[t][0].append(1); sub[t][1].append(r["score"])
                    else:
                        for t in tiers:
                            sub[t][0].append(0); sub[t][1].append(r["score"])
            aps = {k: ap_auc(np.array(sub[k][0], float), np.array(sub[k][1], float))[0] for k in sub}
            # event max per tier
            ns = np.sort(np.array(normal_scores))[::-1]
            evmax = {t: [] for t in tiers}; evstats = {t: [] for t in tiers}
            for e in ev:
                m = man.get(e["video"])
                if m is None:
                    continue
                fps = m["fps"]; e0, e1 = e["event"]
                ws = [r["score"] for r in rows.get(e["video"], []) if r["start_s"] * fps < e1 + 1 and r["end_s"] * fps > e0]
                if not ws:
                    continue
                evmax[e["tier"]].append(max(ws)); evstats[e["tier"]].append(np.array(ws))
            rec = []
            for fa in fas:
                k = int(round(fa * hours)); tau = ns[k - 1] if k >= 1 else ns[0] + 1e-9
                rec.append("/".join(f"{sum(1 for x in evmax[t] if x >= tau)}" for t in tiers) + f" (τ {tau:.2f})")
            lines.append(f"| {n} | {rule} | " + " | ".join(f"{100*aps[t]:.1f}" for t in tiers) + f" | {100*aps['all']:.1f} | " + " | ".join(rec) + " |")
            if rule == "raw":
                for t in tiers:
                    arrs = evstats[t]
                    if not arrs:
                        continue
                    mx = np.array([x.max() for x in arrs]); md = np.array([np.median(x) for x in arrs])
                    frac = np.array([(x >= 0.5).mean() for x in arrs])
                    one = float(np.mean((mx >= 0.5) & (md < 0.3)))
                    a3.append(f"| {n} | {t} | {len(arrs)} | {np.median(mx):.2f} | {np.median(md):.2f} | {frac.mean():.2f} | {one:.2f} |")
    lines.append(f"\n(normal test video: {hours:.1f} h; events per tier: " + ", ".join(f"{t} {sum(1 for e in ev if e['tier']==t)}" for t in tiers) + ")")
    a.out.write_text("\n".join(lines + a3) + "\n")
    print("\n".join(lines + a3))


if __name__ == "__main__":
    main()
