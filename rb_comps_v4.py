"""
Volunteer Vision - RB comps engine v4: schedule + recency weighting
====================================================================
Same as v3, plus recency: each earlier calendar season counts DECAY times as much as the
next one (default 0.5, so a season 3 years back counts 1/8). Added after the Zach Charbonnet
miss, where a weak freshman year dragged down his career profile.
Run after RBcomps.py (v1) and RBcomp2.py (v2):
    python rb_comps_v4.py comps/out          (decay 0.5)
    python rb_comps_v4.py comps/out 0.7      (try another setting)

Held-out tests vs v3 (decay 1.0) came out within noise:
  forecast 0.327 vs 0.333 | draft AUC 2021+ 0.881 vs 0.871 | stability 4.91 vs 4.98
Case studies: Charbonnet and Sampson improve; Jadarian Price gets worse (his last season was weaker).

--- original v3 notes ---

v2 compared every back to all of FBS that season but ignored who he played.
v3 tests two ways to use the workbook's 'strength of schedule' column (0-1, higher = tougher):

  A  FEATURE  - add schedule as its own stat group, so comps prefer backs who faced
                similar defenses. Its weight is chosen on 2016-2020 seasons only.
  B  ADJUST   - raise or lower each stat for schedule difficulty. How much a tougher schedule
                moves each stat is estimated WITHIN players: backs whose schedule changed from
                one season to the next (transfers, conference moves, scheduling swings).
                Comparing a player to himself avoids the trap that SEC backs are also better
                players, which would make tough schedules look like they help stats.
  A+B both.

Every variant is scored on the same held-out tests as v2 (2021+ forecast, time-respecting
draft AUC, stability under noise). The best one gets the 200-run confidence pass.
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

B = sys.argv[1] if len(sys.argv) > 1 else "comps/out"
df = pd.read_pickle(B + ".pkl").reset_index(drop=True)
meta = json.load(open(B + ".meta.json")); v2 = json.load(open(B + "_v2.json"))
GROUPS = meta["groups"]; W2 = v2["weights"]
FE = [f for g in GROUPS.values() for f in g]
SIZE = [FE.index(f) for f in GROUPS["Size"]]
NONSIZE = [i for i in range(len(FE)) if i not in SIZE]
n = len(df)
pid = df.PlayerId.values; season = df.Season.values; car = df.CAR.values.astype(float)
same = pid[:, None] == pid[None, :]

zs = lambda cols: df.groupby("Season")[cols].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
Z = zs(FE).clip(-3, 3).values.astype(np.float32)                       # raw, as in v2
SOS = zs(["strength of schedule"]).values[:, 0].astype(np.float32)    # schedule, z within season

key = {(p, s): i for i, (p, s) in enumerate(zip(pid, season))}
nxt = np.array([key.get((p, s + 1), -1) for p, s in zip(pid, season)])
TGT = NONSIZE
train = np.where((nxt >= 0) & (season >= 2016) & (season <= 2020))[0]
test = np.where((nxt >= 0) & (season >= 2021))[0]
lab = df.Drafted.map({"Y": 1, "N": 0}).values.astype(float); known = ~np.isnan(lab)

# ---------- B: within-player schedule effect ----------
# For consecutive seasons of the same back (training years only):  dStat = a + beta * dSOS
pairs = np.where((nxt >= 0) & (season <= 2020))[0]
dS = SOS[nxt[pairs]] - SOS[pairs]
beta = np.zeros(len(FE), np.float32); beta_se = np.zeros(len(FE))
Xd = np.column_stack([np.ones_like(dS), dS])
for f in NONSIZE:
    y = Z[nxt[pairs], f] - Z[pairs, f]
    coef, res, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    sigma2 = res[0] / (len(y) - 2) if len(res) else np.var(y - Xd @ coef)
    beta[f] = coef[1]; beta_se[f] = np.sqrt(sigma2 * np.linalg.inv(Xd.T @ Xd)[1, 1])
print(f"Schedule effect from {len(pairs)} back-to-back seasons (stat change per +1 SD of schedule):")
for f in NONSIZE:
    flag = "  *" if abs(beta[f]) > 2 * beta_se[f] else ""
    print(f"  {FE[f]:<34} {beta[f]:+.3f}  (se {beta_se[f]:.3f}){flag}")
# only adjust where the effect is clearly real (|beta| > 2 se); others left alone
beta_use = np.where(np.abs(beta) > 2 * beta_se, beta, 0).astype(np.float32)
Z_adj = (Z - beta_use[None, :] * SOS[:, None]).clip(-3, 3).astype(np.float32)

# ---------- profile / distance machinery (same as v2) ----------
K_SHR = v2["k"]
DECAY = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
M = (same & (season[None, :] <= season[:, None])) * car[None, :] * DECAY ** np.clip(season[:, None] - season[None, :], 0, None)
WM = (M / M.sum(1, keepdims=True)).astype(np.float32)
def profile(Zsrc, sos_col):
    Zs = Zsrc * (car / (car + K_SHR))[:, None]
    Zs[:, SIZE] = Zsrc[:, SIZE]
    P = WM @ Zs
    return np.column_stack([P, WM @ sos_col[:, None]]).astype(np.float32)  # last column = schedule
GIDX = {g: [FE.index(f) for f in fs] for g, fs in GROUPS.items()}
GIDX["Schedule"] = [len(FE)]
def group_d2(P):
    return {g: ((P[:, None, ix] - P[None, :, ix]) ** 2).mean(-1) for g, ix in GIDX.items()}
def dist(D2, w):
    return np.sqrt(sum(w.get(g, 0) * D2[g] for g in D2))

def forecast_score(D, rows, K=15):
    preds, acts = [], []
    for i in rows:
        cand = np.where((nxt >= 0) & (season + 1 <= season[i]) & (~same[i]))[0]
        if len(cand) < K: continue
        c = cand[np.argpartition(D[i, cand], K)[:K]]
        preds.append(Z[i, TGT] + (Z[nxt[c]][:, TGT] - Z[c][:, TGT]).mean(0)); acts.append(Z[nxt[i], TGT])
    P, A = np.array(preds), np.array(acts)
    return float(np.mean([np.corrcoef(P[:, t], A[:, t])[0, 1] for t in range(len(TGT))]))
def draft_auc(D, rows_mask):
    Dt = D.copy(); Dt[(season[None, :] >= season[:, None]) | ~known[None, :] | same] = np.inf
    nn = np.argsort(Dt, 1)[:, :15]; ok = known & rows_mask
    return roc_auc_score(lab[ok], lab[nn].mean(1)[ok])
rng = np.random.default_rng(0)
def stability(Zsrc, w):
    D = dist(group_d2(profile(Zsrc, SOS)), w); D[same] = np.inf; top = np.argsort(D, 1)[:, :8]; ov = []
    for _ in range(3):
        noise = rng.normal(0, .25, Z.shape).astype(np.float32)
        Dn = dist(group_d2(profile(Zsrc + noise, SOS)), w); Dn[same] = np.inf
        ov.append(np.mean([len(set(a) & set(b)) for a, b in zip(top, np.argsort(Dn, 1)[:, :8])]))
    return float(np.mean(ov))

# ---------- tune schedule weight on TRAIN seasons only ----------
D2_raw, D2_adj = group_d2(profile(Z, SOS)), group_d2(profile(Z_adj, SOS))
train_mask = (season >= 2016) & (season <= 2020)
def tune(D2):
    best = None
    for ws in (0.0, 0.25, 0.5, 1.0, 2.0):
        w = dict(W2, Schedule=ws); D = dist(D2, w)
        s = (forecast_score(D, train), draft_auc(D, train_mask))
        print(f"    schedule weight {ws:<4}: train forecast r={s[0]:.3f}  train draft AUC={s[1]:.3f}")
        if best is None or s[1] + s[0] > best[0]: best = (s[1] + s[0], ws)
    return best[1]
print("\nA (schedule as a stat), raw stats:"); wA = tune(D2_raw)
print("A+B (schedule as a stat), adjusted stats:"); wAB = tune(D2_adj)

variants = {
    "v2 (no schedule)": (Z, D2_raw, dict(W2, Schedule=0.0)),
    "A: schedule as a stat": (Z, D2_raw, dict(W2, Schedule=wA)),
    "B: schedule-adjusted stats": (Z_adj, D2_adj, dict(W2, Schedule=0.0)),
    "A+B": (Z_adj, D2_adj, dict(W2, Schedule=wAB)),
}
test_mask = season >= 2021
res = {}
print("\nHELD-OUT comparison (2021+ for forecast and draft; stability on all rows)")
print(f"  {'variant':<28} {'forecast r':>10} {'draft AUC':>10} {'draft AUC 2016+':>16} {'stability':>10}")
for name, (Zsrc, D2, w) in variants.items():
    D = dist(D2, w)
    r = dict(forecast=forecast_score(D, test), auc_test=draft_auc(D, test_mask),
             auc_all=draft_auc(D, season >= 2016), stab=stability(Zsrc, w), w_sched=w["Schedule"])
    res[name] = r
    print(f"  {name:<28} {r['forecast']:>10.3f} {r['auc_test']:>10.3f} {r['auc_all']:>16.3f} {r['stab']:>10.2f}   (schedule weight {w['Schedule']})")
json.dump({"beta": dict(zip(FE, beta.round(4).tolist())), "beta_se": dict(zip(FE, beta_se.round(4).tolist())),
           "beta_used": dict(zip(FE, beta_use.round(4).tolist())), "results": res, "wA": wA, "wAB": wAB},
          open(B + "_v4_tests.json", "w"), indent=1)

# ---------- FINAL v3 ----------
# Decision (see printed tests + v3_check.py):
#  * A (schedule as a stat) at weight 1.0. Weight 2.0 scored about the same but lets schedule dominate.
#  * B dropped: within players, schedule moved no efficiency stat by a reliable amount; it only cut
#    carries/game and rec yds/game, which looks like transfers taking smaller roles, not tougher defenses.
#  * Honest note: most of the draft-AUC gain is schedule itself (schedule alone ~ v2 comps; a 2-variable
#    logistic of yards + schedule matches comps + schedule). Schedule's real value for COMPS is that a
#    back's comps now faced similar competition, and the lists are more stable.
W3 = dict(W2, Schedule=1.0)
P3 = profile(Z, SOS); D3 = dist(group_d2(P3), W3)
final = dict(forecast=forecast_score(D3, test), auc_test=draft_auc(D3, test_mask),
             auc_all=draft_auc(D3, season >= 2016), stab=stability(Z, W3))
print(f"\nFINAL v4 (schedule weight 1.0, decay {DECAY}):", {k: round(v, 3) for k, v in final.items()})

# 200-run confidence, counting players not seasons (as in v2). Schedule is not noisy; stats are.
Do = D3.copy(); Do[same] = np.inf
counts = [dict() for _ in range(n)]
R = 200
for r in range(R):
    sd = (0.25 * np.sqrt(150 / car))[:, None].astype(np.float32)
    Dn = dist(group_d2(profile(Z + rng.normal(0, 1, Z.shape).astype(np.float32) * sd, SOS)), W3); Dn[same] = np.inf
    top = np.argpartition(Dn, 40, axis=1)[:, :40]
    for i in range(n):
        seen = []
        for j in top[i][np.argsort(Dn[i, top[i]])]:
            if pid[j] not in seen:
                seen.append(pid[j])
                if len(seen) == 8: break
        for p in seen: counts[i][p] = counts[i].get(p, 0) + 1
    if r % 50 == 0: print("resample", r)
allp = np.sort(Do[np.isfinite(Do)])
rows_of = pd.Series(range(n)).groupby(pid).apply(list).to_dict()
comps = []
for i in range(n):
    best = {p: min(rows_of[p], key=lambda j: Do[i, j]) for p in counts[i]}
    best8 = sorted(((best[p], c) for p, c in counts[i].items()), key=lambda kv: (-kv[1], Do[i, kv[0]]))[:8]
    comps.append([[int(j), round(100 * c / R), round(100 - 100 * np.searchsorted(allp, Do[i, j]) / len(allp), 1)] for j, c in best8])
conf = np.array([c[0][1] for c in comps])
final.update(median_top_comp_conf=float(np.median(conf)), strong_share=float(np.mean(conf >= 70)))
print("median confidence of #1 comp:", np.median(conf), " share with a strong comp:", round(float(np.mean(conf >= 70)), 3))
np.save(B + "_v4_P.npy", P3)
json.dump({"comps": comps, "weights": W3, "k": K_SHR, "decay": DECAY, "version": "v4", "results": final,
           "tests": res, "beta": dict(zip(FE, beta.round(4).tolist()))}, open(B + "_v4.json", "w"))
