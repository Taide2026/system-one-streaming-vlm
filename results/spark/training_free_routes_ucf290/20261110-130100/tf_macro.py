"""Copy of tf_auc.py that also reports macro AUC (all tier; per anomaly video, event vs label-0 windows, mean; paired bootstrap over videos, 1,000 draws, seed 20261022); writes tf_macro.json."""
import json, importlib.util, numpy as np
from pathlib import Path
R = Path("/home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/small-event-fov")
spec = importlib.util.spec_from_file_location("le", R/"scripts/layout-eval.py"); le = importlib.util.module_from_spec(spec); spec.loader.exec_module(le)
E = Path("/home/cctsai/Eric/VLM/system-one-streaming-vlm-wt/encoder-probing/results/spark/encoder_probe_ucf_test_rest_v1_f8/20261108-063724")
def load(p): return {(r["video"], r["window"]): r for r in map(json.loads, Path(p).read_text().splitlines())}
def ap(y, s):
    o = np.argsort(-s); y = y[o]; tp = np.cumsum(y); return float((tp/np.arange(1,len(y)+1)*y).sum()/y.sum())
def auc(y, s):
    o = np.argsort(-s); y = y[o]; tp, fp = np.cumsum(y), np.cumsum(1-y); return float(np.trapezoid(tp/tp[-1], fp/fp[-1]))
def logit(p): p = np.clip(p, 1e-6, 1-1e-6); return np.log(p/(1-p))
def sig(z): return 1/(1+np.exp(-z))
class D:
    def __init__(s, p1, p2, pv, vkey, hot=None):
        a, b, v = load(p1), load(p2), load(pv); s.keys = k = sorted(set(a)&set(b)&set(v))
        s.rows = [a[x] for x in k]
        s.y = np.array([a[x]["label"] for x in k], float); s.s1 = np.array([a[x]["score"] for x in k]); s.s2 = np.array([b[x]["score"] for x in k])
        s.v = np.array([v[x][vkey] for x in k]); s.vid = np.array([x[0] for x in k]); s.w = np.array([x[1] for x in k])
        h = load(hot) if hot else a; s.hot = np.array([h[x]["hot_area"] for x in k])
        va = {x[0] for x in k if a[x]["label"]}; s.grp = np.array(["ev" if a[x]["label"] else ("an" if x[0] in va else "nn") for x in k])
        s.d = s.s2 - s.s1
        # per-video ordered index lists
        s.vids = {}
        for i, (vv, ww) in enumerate(zip(s.vid, s.w)): s.vids.setdefault(vv, []).append((ww, i))
        for vv in s.vids: s.vids[vv] = [i for _, i in sorted(s.vids[vv])]
def causal(dset, f):
    """f(past_idx_list, i) -> score; past = earlier windows of the same video."""
    out = np.zeros(len(dset.y))
    for vv, idx in dset.vids.items():
        for n, i in enumerate(idx): out[i] = f(idx[:n], i)
    return out
# ---------- rules; each returns a score vector given dataset and params
def r_band(ds, lo, hi): z = (ds.s1>=lo)&(ds.s1<hi); return np.where(z, ds.s2, ds.s1), z
def r_dcorr(ds, alpha, incl):
    def f(past, i):
        p = past + ([i] if incl else [])
        return ds.s2[i] - alpha*ds.d[p].mean() if p else ds.s2[i]
    return causal(ds, f)
def r_qmatch(ds, nmin=5):
    def f(past, i):
        if len(past) < nmin: return ds.s2[i]
        q = (ds.s2[past] <= ds.s2[i]).mean(); return float(np.quantile(ds.s1[past], q))
    return causal(ds, f)
def r_vidroute(ds, tau, stat):
    def f(past, i):
        p = past + [i]; v = ds.s1[p].max() if stat=="max" else ds.s1[p].mean()
        return ds.s2[i] if v >= tau else ds.s1[i]
    return causal(ds, f)
def r_smooth(ds, base, k):
    s = ds.s1 if base=="1x" else ds.s2
    return causal(ds, lambda past, i: s[(past+[i])[-k:]].mean())
def r_ema(ds, base, a):
    s = ds.s1 if base=="1x" else ds.s2; out = np.zeros(len(s))
    for vv, idx in ds.vids.items():
        m = None
        for i in idx: m = s[i] if m is None else a*s[i] + (1-a)*m; out[i] = m
    return out
def r_vj(ds, base, w):
    s = ds.s1 if base=="1x" else ds.s2; return sig(logit(s) + w*ds.v)
def r_zs1(ds, alpha):  # causal per-video z-score of s1 (mean-shift only)
    return causal(ds, lambda past, i: ds.s1[i] - alpha*ds.s1[past+[i]].mean())
def r_hotband(ds, g, lo, hi): z = (ds.hot<g)&(ds.s1>=lo)&(ds.s1<hi); return np.where(z, ds.s2, ds.s1)

tr = D(R/"results/spark/train_zoom_scores_ucf390/20261108-131218/mixkl/1x/scores.jsonl", R/"results/spark/train_zoom_scores_ucf390/20261108-131218/g4/mixkl/2x/scores.jsonl", E/"train-scores/train-scores.jsonl", "crossfit", hot=R/"results/spark/train_zoom_scores_ucf390/20261108-131218/windows.jsonl")
te = D(R/"results/spark/sr_ucf290_d3nano_ft/20261106-142617/floor_d3nano_c/draft/scores.jsonl", R/"results/spark/sr_ucf290_d3nano_ft/20261106-142617/floor_d3nano_c/B/scores.jsonl", E/"refit-290/scores.jsonl", "score")
print(f"train n={len(tr.y)} test n={len(te.y)}; train AP 1x {ap(tr.y,tr.s1)*100:.1f} 2x {ap(tr.y,tr.s2)*100:.1f}; vjepa AUC train(crossfit) {auc(tr.y,tr.v):.3f} test {auc(te.y,te.v):.3f}, AP test {ap(te.y,te.v)*100:.1f}")
# within-bin AUC(d) ev vs an, train and test
for nm, ds in [("train", tr), ("test", te)]:
    out=[]
    for lo,hi in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.01)]:
        m = (ds.s1>=lo)&(ds.s1<hi)&(ds.grp!="nn"); out.append(f"[{lo:.1f},{hi:.1f}):{auc((ds.grp[m]=='ev').astype(float), ds.d[m]):.2f}")
    print(f"  within-bin AUC(d) ev vs an {nm}: " + " ".join(out))
# ---------- test eval with bootstrap
manifest = {m["name"]: m for m in json.loads((R/"results/spark/frames_ucf290_f8/20261105-180529/manifest.json").read_text())}
events = [json.loads(l) for l in (R/"results/spark/tgm_size_ucf290/20261105-200402/jev-hard-events.jsonl").read_text().splitlines()]
masks = le.tier_masks(te.rows, manifest, events)
rng = np.random.default_rng(le.SEED); draws = [le.bootstrap_index(te.rows, manifest, rng) for _ in range(le.DRAWS)]
every = np.arange(len(te.rows))
base1 = [le.metrics(te.s1, masks, d) for d in draws]; base2 = [le.metrics(te.s2, masks, d) for d in draws]
def ci(boot, base, t, j=1): x = np.array([b[t][j]-c[t][j] for b,c in zip(boot,base)])*100; return [round(float(np.percentile(x,2.5)),1), round(float(np.percentile(x,97.5)),1)]

VID = np.array([r["video"] for r in te.rows]); INC, YA = masks["all"]
MVIDS = sorted(v for v in np.unique(VID[INC & (YA == 1)]) if 0 < YA[INC & (VID == v)].sum() < (INC & (VID == v)).sum())
MIDX = [np.where(INC & (VID == v))[0] for v in MVIDS]
MRNG = np.random.default_rng(20261022); MDRAWS = [MRNG.integers(0, len(MVIDS), len(MVIDS)) for _ in range(1000)]
from s1stream import metrics as SM
def macro_vals(s): return np.array([SM.auc(list(s[i]), list(YA[i])) for i in MIDX])
MBASE = macro_vals(te.s1); MBASE2 = macro_vals(te.s2)
def macro_ci(vals, base): d = [vals[x].mean() - base[x].mean() for x in MDRAWS]; return [round(float(np.percentile(d, 2.5))*100, 1), round(float(np.percentile(d, 97.5))*100, 1)]
OUT = []
def report(name, s, note=""):
    pt = le.metrics(s, masks, every); boot = [le.metrics(s, masks, d) for d in draws]
    row = {"rule": name, "note": note}
    for t in ("small", "medium", "large", "all"):
        row[t] = {"auc": round(pt[t][0]*100, 1), "ap": round(pt[t][1]*100, 1),
                  "auc_vs1x": ci(boot, base1, t, 0), "ap_vs1x": ci(boot, base1, t, 1),
                  "auc_vs2x": ci(boot, base2, t, 0), "ap_vs2x": ci(boot, base2, t, 1)}
    mv = macro_vals(s); row["macro"] = {"auc": round(float(mv.mean())*100, 1), "vs1x": macro_ci(mv, MBASE), "vs2x": macro_ci(mv, MBASE2), "videos": len(MVIDS)}
    OUT.append(row)
    a = row["all"]
    print(f"{name:34s} AUC {a['auc']} vs1x{a['auc_vs1x']} vs2x{a['auc_vs2x']} | AP {a['ap']} vs1x{a['ap_vs1x']} vs2x{a['ap_vs2x']} | macro {row['macro']['auc']} vs1x{row['macro']['vs1x']} vs2x{row['macro']['vs2x']} | {note}", flush=True)
report("1x", te.s1); report("all 2x", te.s2)
# band router chosen on train
grid = np.arange(0, 1.0001, 0.05); best = max(((ap(tr.y, r_band(tr,lo,hi)[0]), lo, hi) for lo in grid for hi in grid if hi>lo))
s, z = r_band(te, best[1], best[2]); report(f"band s1 in [{best[1]:.2f},{best[2]:.2f})", s, f"train AP {best[0]*100:.1f}, zoom {z.mean()*100:.0f}%")
# delta correction
for incl in (True, False):
    bestA = max((ap(tr.y, r_dcorr(tr, a, incl)), a) for a in [0.25,0.5,0.75,1.0,1.5])
    report(f"s2 - a*cummean(d) a={bestA[1]} incl={incl}", r_dcorr(te, bestA[1], incl), f"train AP {bestA[0]*100:.1f}")
report("quantile match s2->s1 (past>=5)", r_qmatch(te), f"train AP {ap(tr.y, r_qmatch(tr))*100:.1f}")
# video-level routing
for stat in ("max","mean"):
    b = max((ap(tr.y, r_vidroute(tr, t, stat)), t) for t in np.arange(0.1,1.0,0.05))
    s = r_vidroute(te, b[1], stat); report(f"video route run-{stat} s1>={b[1]:.2f}", s, f"train AP {b[0]*100:.1f}, zoom {np.mean(s!=te.s1)*100:.0f}%(approx)")
# smoothing
for base in ("1x","2x"):
    b = max((ap(tr.y, r_smooth(tr, base, k)), k) for k in (1,2,3,4,6,8))
    report(f"causal mean k={b[1]} on {base}", r_smooth(te, base, b[1]), f"train AP {b[0]*100:.1f}")
    b = max((ap(tr.y, r_ema(tr, base, a)), a) for a in (0.3,0.5,0.7,0.85,1.0))
    report(f"EMA a={b[1]} on {base}", r_ema(te, base, b[1]), f"train AP {b[0]*100:.1f}")
# vjepa ensemble
for base in ("1x","2x"):
    b = max((ap(tr.y, r_vj(tr, base, w)), w) for w in (0,0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0))
    report(f"logit({base}) + w*vjepa w={b[1]}", r_vj(te, base, b[1]), f"train AP {b[0]*100:.1f}")
# causal per-video mean-shift of s1
b = max((ap(tr.y, r_zs1(tr, a)), a) for a in (0,0.25,0.5,0.75,1.0))
report(f"s1 - a*cummean(s1) a={b[1]}", r_zs1(te, b[1]), f"train AP {b[0]*100:.1f}")
# TGM gate on train + hot&band
print(f"  train: TGM gate hot<0.045 AP {ap(tr.y, np.where(tr.hot<0.045, tr.s2, tr.s1))*100:.1f}; zoom {np.mean(tr.hot<0.045)*100:.0f}%")
b = max((ap(tr.y, r_hotband(tr, 0.045, lo, hi)), lo, hi) for lo in grid for hi in grid if hi>lo)
report(f"hot<0.045 & s1 in [{b[1]:.2f},{b[2]:.2f})", r_hotband(te, 0.045, b[1], b[2]), f"train AP {b[0]*100:.1f}")
# combined: delta-correction (past only) then causal mean k, chosen jointly on train
def r_combo(ds, a, k):
    base = r_dcorr(ds, a, False); return causal(ds, lambda past, i: base[(past+[i])[-k:]].mean())
b = max((ap(tr.y, r_combo(tr, a, k)), a, k) for a in (0,0.25,0.5) for k in (1,2,3))
report(f"combo a={b[1]} k={b[2]}", r_combo(te, b[1], b[2]), f"train AP {b[0]*100:.1f}")
# smoothing k=2 of 2x, per-group mean shift check
s = r_smooth(te, "2x", 2); print("  k=2 on 2x: ev-vs-an AUC", round(auc((te.grp[te.grp!='nn']=='ev').astype(float), s[te.grp!='nn']),3), "vs raw 2x", round(auc((te.grp[te.grp!='nn']=='ev').astype(float), te.s2[te.grp!='nn']),3))

json.dump(OUT, open(Path(__file__).parent / "tf_macro.json", "w"), indent=1)
