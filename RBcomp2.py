"""
Volunteer Vision - RB comps engine v2
=====================================
Fixes the three problems the v1 audit found:
  1. Single seasons are noisy  -> career-to-date profile (carry-weighted, older seasons decay)
                                  + shrink low-carry stats toward the season average
  2. Equal group weights are arbitrary -> weights chosen by what best forecasts a back's NEXT season
                                  (tuned on 2016-2020, tested on 2021-2024)
  3. Comp lists are unstable    -> resample with stat noise 200x; report how often each comp appears

Test used for tuning and for v1-vs-v2 comparison ("forecast test"):
  For a back in season S, take his 15 nearest comps whose following season is already known
  (comp season + 1 <= S, so no peeking at the future). Predict his season S+1 from what
  happened to the comps in their next season. Score = average correlation between predicted
  and actual next-season stats. A comps engine that finds truly similar backs forecasts better.
"""
import json, sys, itertools
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

B = sys.argv[1]                       # prefix of v1 outputs (reuses the cleaned table)
df = pd.read_pickle(B + ".pkl").reset_index(drop=True)
meta = json.load(open(B + ".meta.json"))
GROUPS = meta["groups"]
FE = [f for g in GROUPS.values() for f in g]
SIZE = set(GROUPS["Size"])
n = len(df)
pid = df.PlayerId.values; season = df.Season.values; car = df.CAR.values.astype(float)

# within-season z-scores (raw, single season)
Z = df.groupby("Season")[FE].transform(lambda s: (s - s.mean()) / s.std(ddof=0)).clip(-3, 3).values.astype(np.float32)
gidx = {g: [FE.index(f) for f in fs] for g, fs in GROUPS.items()}

# next-season index (same player, season+1)
key = {(p, s): i for i, (p, s) in enumerate(zip(pid, season))}
nxt = np.array([key.get((p, s + 1), -1) for p, s in zip(pid, season)])
same = pid[:, None] == pid[None, :]

# forecast targets: the stats a scout cares about next year (not size)
TGT = [FE.index(f) for f in FE if f not in SIZE]

def profile(k, decay):
    """shrink low-carry seasons, then carry-weighted career-to-date average"""
    shrink = (car / (car + k))[:, None] if k > 0 else np.ones((n, 1))
    Zs = Z * shrink
    Zs[:, [FE.index(f) for f in SIZE]] = Z[:, [FE.index(f) for f in SIZE]]   # size is not noisy
    return (WMAT(decay) @ Zs).astype(np.float32)

_wcache = {}
def WMAT(decay):
    """row i = weights over player i's seasons up to and including this one"""
    if decay not in _wcache:
        M = (same & (season[None, :] <= season[:, None])) * car[None, :] * decay ** np.clip(season[:, None] - season[None, :], 0, None)
        _wcache[decay] = (M / M.sum(1, keepdims=True)).astype(np.float32)
    return _wcache[decay]

def group_d2(P):
    return np.stack([((P[:, None, ix] - P[None, :, ix]) ** 2).mean(-1) for ix in gidx.values()]).astype(np.float32)

def dist(D2, w):
    return np.sqrt(np.tensordot(np.asarray(w, np.float32), D2, 1))

def forecast_score(D, rows, K=15, mode="delta"):
    """mean correlation (over target stats) between predicted and actual next season"""
    preds, acts = [], []
    for i in rows:
        ok = (nxt >= 0) & (season + 1 <= season[i]) & (~same[i])
        cand = np.where(ok)[0]
        if len(cand) < K: continue
        c = cand[np.argpartition(D[i, cand], K)[:K]]
        if mode == "level":
            p = Z[nxt[c]][:, TGT].mean(0)
        else:   # PECOTA-style: his current stats + how comps changed the next year
            p = Z[i, TGT] + (Z[nxt[c]][:, TGT] - Z[c][:, TGT]).mean(0)
        preds.append(p); acts.append(Z[nxt[i], TGT])
    P, A = np.array(preds), np.array(acts)
    return float(np.mean([np.corrcoef(P[:, t], A[:, t])[0, 1] for t in range(len(TGT))])), len(P)

train = np.where((nxt >= 0) & (season >= 2016) & (season <= 2020))[0]
test  = np.where((nxt >= 0) & (season >= 2021))[0]

# ---------- baselines + v1 ----------
def persist(rows):
    P, A = Z[rows][:, TGT], Z[nxt[rows]][:, TGT]
    return float(np.mean([np.corrcoef(P[:, t], A[:, t])[0, 1] for t in range(len(TGT))]))
w1 = [1.0] * len(GROUPS)
D2_v1 = group_d2(Z)
res = {"persist_train": persist(train), "persist_test": persist(test)}
for mode in ("level", "delta"):
    res[f"v1_{mode}_train"] = forecast_score(dist(D2_v1, w1), train, mode=mode)[0]
print("baseline/v1:", {k: round(v, 3) for k, v in res.items()})

# ---------- tune shrinkage k and history decay (train only) ----------
best = None
for k, decay in itertools.product([0, 50, 100, 200], [0.0, 0.4, 0.7, 1.0]):
    P = profile(k, decay) if decay > 0 else profile(k, 1e-9)
    D2 = group_d2(P)
    s = forecast_score(dist(D2, w1), train)[0]
    print(f"  k={k:<3} decay={decay}: train r={s:.3f}")
    if best is None or s > best[0]: best = (s, k, decay, D2, P)
_, K_SHR, DECAY, D2, P = best
print("chosen shrink k =", K_SHR, "decay =", DECAY)

# ---------- learn group weights (coordinate search, train only) ----------
w = np.ones(len(GROUPS)); cur = forecast_score(dist(D2, w), train)[0]
for _ in range(2):
    for gi in range(len(GROUPS)):
        for cand in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            t = w.copy(); t[gi] = cand
            if t.sum() == 0: continue
            s = forecast_score(dist(D2, t), train)[0]
            if s > cur + 1e-4: cur, w = s, t
w = w / w.sum() * len(w)
WEIGHTS = dict(zip(GROUPS, np.round(w, 2).tolist()))
print("learned weights:", WEIGHTS, "train r =", round(cur, 3))

# Tuning showed every setting forecasts about equally well (within +/-0.01), so the forecast
# test can't separate them. Lock in the full career profile (k=100, no decay): it is
# conceptually right for scouting (a back is more than his last season) and the most stable
# under noise in v2_check.py (4.4 vs 3.9 of 8 comps kept).
K_SHR, DECAY = 100, 1.0
P = profile(K_SHR, DECAY); D2 = group_d2(P)
D_v1, D_v2 = dist(D2_v1, w1), dist(D2, w)

# ---------- held-out comparison ----------
out = {"persist_test": res["persist_test"]}
for name, D in (("v1", D_v1), ("v2", D_v2)):
    out[f"{name}_forecast_test"] = forecast_score(D, test)[0]
    out[f"{name}_forecast_test_level"] = forecast_score(D, test, mode="level")[0]

# draft AUC, time-respecting (comps only from earlier seasons with known outcome)
lab = df.Drafted.map({"Y": 1, "N": 0}).values.astype(float); known = ~np.isnan(lab)
def draft_auc(D):
    Dt = D.copy(); Dt[(season[None, :] >= season[:, None]) | ~known[None, :] | same] = np.inf
    nn = np.argsort(Dt, 1)[:, :15]; ok = known & (season >= 2016)
    return roc_auc_score(lab[ok], lab[nn].mean(1)[ok])
out["v1_draft_auc"], out["v2_draft_auc"] = draft_auc(D_v1), draft_auc(D_v2)
out["yards_draft_auc"] = roc_auc_score(lab[known & (season >= 2016)], df.YDS.values[known & (season >= 2016)])

# stability: same noise test as the audit (0.25 SD added to every single-season stat)
rng = np.random.default_rng(0)
def stability(D, build):
    Do = D.copy(); Do[same] = np.inf; top = np.argsort(Do, 1)[:, :8]; ov = []
    for _ in range(5):
        Dn = build(Z + rng.normal(0, .25, Z.shape).astype(np.float32)); Dn[same] = np.inf
        tn = np.argsort(Dn, 1)[:, :8]
        ov.append(np.mean([len(set(a) & set(b)) for a, b in zip(top, tn)]))
    return float(np.mean(ov))

def build_v2(Zn):
    global Z
    keep = Z; Z = Zn; Pn = profile(K_SHR, DECAY if DECAY > 0 else 1e-9); Z = keep
    return dist(group_d2(Pn), w)
out["v1_stability"] = stability(D_v1, lambda Zn: dist(group_d2(Zn), w1))
out["v2_stability"] = stability(D_v2, build_v2)
print("held-out:", {k: round(v, 3) for k, v in out.items()})

# ---------- confidence: how often each comp appears across 200 noisy resamples ----------
# noise per stat scales with 1/sqrt(carries): a 100-carry season is noisier than a 300-carry one
Do = D_v2.copy(); Do[same] = np.inf
counts = [dict() for _ in range(n)]
R = 200
for r in range(R):
    sd = (0.25 * np.sqrt(150 / car))[:, None].astype(np.float32)
    Zn = Z + rng.normal(0, 1, Z.shape).astype(np.float32) * sd
    Dn = build_v2(Zn); Dn[same] = np.inf
    # count PLAYERS, not seasons: a comp's 2022 and 2023 profiles shouldn't take two slots
    top = np.argpartition(Dn, 40, axis=1)[:, :40]
    for i in range(n):
        seen = []
        for j in top[i][np.argsort(Dn[i, top[i]])]:
            if pid[j] not in seen:
                seen.append(pid[j])
                if len(seen) == 8: break
        for p in seen:
            counts[i][p] = counts[i].get(p, 0) + 1
    if r % 50 == 0: print("resample", r)
allp = np.sort(Do[np.isfinite(Do)])
rows_of = pd.Series(range(n)).groupby(pid).apply(list).to_dict()
comps = []
for i in range(n):
    # represent each comp player by his season closest to this back (noise-free distance)
    best = {p: min(rows_of[p], key=lambda j: Do[i, j]) for p in counts[i]}
    best8 = sorted(((best[p], c) for p, c in counts[i].items()), key=lambda kv: (-kv[1], Do[i, kv[0]]))[:8]
    comps.append([[int(j), round(100 * c / R), round(100 - 100 * np.searchsorted(allp, Do[i, j]) / len(allp), 1)] for j, c in best8])
conf = np.array([c[0][1] for c in comps]); conf8 = np.array([np.mean([x[1] for x in c]) for c in comps])
out["median_top_comp_conf"] = float(np.median(conf)); out["median_avg_conf_top8"] = float(np.median(conf8))
print("median confidence of #1 comp:", np.median(conf), " avg of top 8:", np.median(conf8))

np.save(B + "_v2_P.npy", P)
json.dump({"comps": comps, "weights": WEIGHTS, "k": K_SHR, "decay": DECAY, "results": out,
           "train": {"persist": res["persist_train"], "v1": res["v1_delta_train"], "v2": cur}},
          open(B + "_v2.json", "w"))