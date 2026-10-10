"""
Volunteer Vision - full evaluation of the comps engine on every back with a known draft result
=============================================================================================
    python evaluate.py comps/out v5            (any of v3 v4 v5 that you have run)
    python evaluate.py comps/out v3 v4 v5      (compare versions on the same test)

For each back's FINAL college season (2017-2025, outcome known: drafted or undrafted):
  1. take his 8 nearest comps using that version's profiles and weights, but ONLY comps from
     earlier seasons whose draft result was already known (no peeking at the future)
  2. predicted pick = distance-weighted median of the comps' picks (undrafted = pick 260)
  3. compare with his actual pick
Baselines get the exact same treatment, using only rushing yards (and yards + schedule)
to find "comps", so the only difference is how similarity is measured.

Scores
  Spearman  rank correlation of predicted vs actual pick (1 = perfect order, 0 = random)
  MAE       average miss in rounds (32 picks = 1 round)
  within 1  share of backs whose predicted pick was within one round of the actual pick
  AUC       drafted vs undrafted, using the comps' draft rate
Output: printed report + eval_<version>.csv with every back's prediction and error.
"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

B = sys.argv[1] if len(sys.argv) > 1 else "comps/out"
VERS = sys.argv[2:] or ["v5"]
UDFA, K = 260, 8

df = pd.read_pickle(B + ".pkl").reset_index(drop=True)
groups0 = json.load(open(B + ".meta.json"))["groups"]
n = len(df); pid = df.PlayerId.values; season = df.Season.values
lab = df.Drafted.map({"Y": 1, "N": 0}).values.astype(float); known = ~np.isnan(lab)
pick = np.where(lab == 1, df["NFL Draft Pick (overall)"].fillna(UDFA).values, UDFA).astype(float)
final = df.groupby("PlayerId").Season.transform("max").values == season
test = np.where(final & known & (season >= 2017))[0]
same = pid[:, None] == pid[None, :]

def wmedian(v, w):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w)
    return v[np.searchsorted(c, c[-1] / 2)]

def predict(D, extra_block=None):
    """time-respecting KNN prediction for every test row"""
    pred, rate = np.full(n, np.nan), np.full(n, np.nan)
    for i in test:
        ok = known & (season < season[i]) & ~same[i]
        if extra_block is not None: ok &= ~extra_block[i]
        cand = np.where(ok)[0]
        if len(cand) < K: continue
        c = cand[np.argsort(D[i, cand])[:K]]
        w = 1 / (D[i, c] + 1e-6)
        pred[i] = wmedian(pick[c], w); rate[i] = np.average(lab[c], weights=w)
    return pred, rate

def score(pred, rate, rows=None):
    rows = test if rows is None else rows
    rows = rows[~np.isnan(pred[rows])]
    p, a = pred[rows], pick[rows]
    dr = rows[lab[rows] == 1]
    return dict(n=len(rows), spearman=spearmanr(p, a).statistic, mae_rounds=np.mean(np.abs(p - a)) / 32,
                within1=np.mean(np.abs(p - a) <= 32), mae_drafted=np.mean(np.abs(pred[dr] - pick[dr])) / 32 if len(dr) else np.nan,
                auc=roc_auc_score(lab[rows], rate[rows]))

zs = lambda col: df.groupby("Season")[col].transform(lambda s: (s - s.mean()) / s.std(ddof=0)).values
results, preds = {}, {}
yz = zs("YDS"); sz = zs("strength of schedule")
D = np.abs(yz[:, None] - yz[None, :]); preds["Baseline: rushing yards"] = predict(D)
D = np.sqrt((yz[:, None] - yz[None, :]) ** 2 + (sz[:, None] - sz[None, :]) ** 2); preds["Baseline: yards + schedule"] = predict(D)

for v in VERS:
    js = json.load(open(f"{B}_{v}.json")); P = np.load(f"{B}_{v}_P.npy"); w = js["weights"]
    g = {k: list(fs) for k, fs in groups0.items()}
    if v >= "v5" and "RB Carry Share" in df: g["Role"] = g["Role"] + ["RB Carry Share"]
    FE = [f for fs in g.values() for f in fs]
    gidx = {k: [FE.index(f) for f in fs] for k, fs in g.items()}; gidx["Schedule"] = [len(FE)]
    assert P.shape[1] == len(FE) + 1, f"{v}: profile has {P.shape[1]} columns, expected {len(FE) + 1}"
    D = np.sqrt(sum(w.get(k, 0) * ((P[:, None, ix] - P[None, :, ix]) ** 2).mean(-1) for k, ix in gidx.items()))
    block = None
    if v >= "v5" and "RB Role" in df:
        role = df["RB Role"].fillna("Starter").to_numpy(dtype=object); block = role[:, None] != role[None, :]
    preds[f"Comps {v}"] = predict(D, block)

print(f"\nTest set: {len(test)} backs, final college season 2017-2025, known draft result "
      f"({int(lab[test].sum())} drafted, {int((1 - lab[test]).sum())} undrafted)\n")
print(f"  {'model':<28}{'Spearman':>9}{'MAE (rds)':>11}{'within 1 rd':>13}{'MAE drafted':>13}{'AUC':>7}")
for name, (p, r) in preds.items():
    s = score(p, r); results[name] = s
    print(f"  {name:<28}{s['spearman']:>9.3f}{s['mae_rounds']:>11.2f}{s['within1']:>13.0%}{s['mae_drafted']:>13.2f}{s['auc']:>7.3f}")

# by actual draft tier, for the newest comps version
best = f"Comps {VERS[-1]}"; p, r = preds[best]
tiers = [("Round 1", 1, 32), ("Rounds 2-3", 33, 105), ("Rounds 4-7", 106, 259), ("Undrafted", 260, 260)]
print(f"\n{best} by actual outcome (average miss in rounds; + = predicted too late, - = too early):")
for name, lo, hi in tiers:
    rows = test[(pick[test] >= lo) & (pick[test] <= hi) & ~np.isnan(p[test])]
    if len(rows): print(f"  {name:<12} n={len(rows):<4} avg miss {np.mean(np.abs(p[rows] - pick[rows])) / 32:5.2f}   bias {np.mean(p[rows] - pick[rows]) / 32:+5.2f}")

out = df.loc[test, ["Player", "Season", "Team", "CAR", "YPC", "PFF Rushing Grade", "strength of schedule"]].copy()
if "RB Role" in df: out["RB Role"] = df.loc[test, "RB Role"].values
out["actual_pick"] = pick[test]; out["predicted_pick"] = p[test].round(0)
out["miss_rounds"] = ((p[test] - pick[test]) / 32).round(2)
out = out.sort_values("miss_rounds")
out.to_csv(f"eval_{VERS[-1]}.csv", index=False)
print(f"\nBiggest misses, {best}: comps predicted TOO HIGH (drafted later or undrafted):")
print(out.head(15)[["Player", "Season", "Team", *(["RB Role"] if "RB Role" in out else []), "actual_pick", "predicted_pick", "miss_rounds"]].to_string(index=False))
print(f"\nBiggest misses, {best}: comps predicted TOO LOW (drafted earlier than predicted):")
print(out.tail(10)[["Player", "Season", "Team", *(["RB Role"] if "RB Role" in out else []), "actual_pick", "predicted_pick", "miss_rounds"]].to_string(index=False))
json.dump({k: {m: float(x) for m, x in s.items()} for k, s in results.items()}, open(f"eval_summary_{VERS[-1]}.json", "w"), indent=1)
print(f"\nSaved eval_{VERS[-1]}.csv (every back) and eval_summary_{VERS[-1]}.json")
