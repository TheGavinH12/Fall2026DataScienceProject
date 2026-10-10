"""
Volunteer Vision - RB comps engine + archetypes
================================================
Input : Data_Science_PLAYER_Data_7.xlsx (one sheet per season)
Output: comps.json (data for the interactive page) + printed diagnostics

Steps
1. Stack the 2014-2025 season sheets (PFF data starts in 2014).
2. Engineer per-carry / per-game features so volume doesn't dominate.
3. Z-score every feature *within its season* (removes era drift).
4. Comps   = nearest neighbours on weighted feature groups (other players only).
5. Archetypes = Gaussian mixture on style features, k chosen by BIC.
6. Validate: do a back's comps' draft rate predict whether he was drafted? (ROC AUC)
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

SRC = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "comps.json"

# ---------- 1. load ----------
sheets = pd.read_excel(SRC, sheet_name=None)
df = pd.concat([d[d.Player.notna()] for s, d in sheets.items() if s.isdigit()], ignore_index=True)
df = df[(df.Season >= 2014) & (df.Season <= 2025)].copy()
df = df[df["PFF Rushing Grade"].notna()].reset_index(drop=True)   # drop the 24 rows PFF lacks

def h2in(h):
    try:
        f, i = str(h).split("-"); return int(f) * 12 + int(i)
    except Exception:
        return np.nan
df["height_in"] = df.Height.map(h2in)
# Weight missing on ~13% of rows: fill with median weight for that height
df["weight"] = df.Weight.fillna(df.groupby("height_in").Weight.transform("median")).fillna(df.Weight.median())
df["height_in"] = df.height_in.fillna(df.height_in.median())

# ---------- 2. features ----------
gp = df["Games Played"].clip(lower=1)
df["car_pg"]      = df.CAR / gp
df["mtf_rate"]    = df["PFF Missed Tackles Forced"] / df.CAR
df["tgt_share"]   = df.Targets / (df.CAR + df.Targets)
df["rec_ypg"]     = df["Receiving Yds"] / gp
df["gl_share"]    = df["Goal-Line Carries"] / df.CAR
df["ranked"]      = df["Star Rating #"].notna().astype(int)

GROUPS = {   # group -> features ; each group gets equal total weight
    "Size":        ["height_in", "weight"],
    "Efficiency":  ["YPC", "Rush Success Rate", "EPA per Rush", "PFF Rushing Grade"],
    "Run style":   ["Explosive Run Rate (10+ yd)", "Stuffed Rate", "PFF Breakaway %",
                    "PFF Yards After Contact per Att", "mtf_rate"],
    "Role":        ["car_pg", "gl_share"],
    "Receiving":   ["tgt_share", "rec_ypg"],
}
FEATS = [f for g in GROUPS.values() for f in g]
for f in FEATS:
    df[f] = df[f].astype(float).fillna(df[f].median())

# ---------- 3. within-season z-scores ----------
Z = df.groupby("Season")[FEATS].transform(lambda s: (s - s.mean()) / s.std(ddof=0)).clip(-3, 3)
w = np.concatenate([[1 / np.sqrt(len(g))] * len(g) for g in GROUPS.values()])
X = Z.values * w

# ---------- 4. comps ----------
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
same_player = df.PlayerId.values[:, None] == df.PlayerId.values[None, :]
D[same_player] = np.inf
ref = np.percentile(D[np.isfinite(D)], 50)          # median pair distance -> 0 similarity
K = 8
comp_idx = np.argsort(D, axis=1)[:, :K]

# ---------- 5. archetypes ----------
STYLE = ["height_in", "weight", "Explosive Run Rate (10+ yd)", "Stuffed Rate", "PFF Breakaway %",
         "PFF Yards After Contact per Att", "mtf_rate", "car_pg", "gl_share", "tgt_share", "rec_ypg"]
S = Z[STYLE].values
bics = {k: GaussianMixture(k, covariance_type="diag", n_init=5, random_state=0).fit(S).bic(S) for k in range(3, 9)}
k_best = min(bics, key=bics.get)
k_best = max(4, min(k_best, 6))                       # keep it explainable: 4-6 types
gmm = GaussianMixture(k_best, covariance_type="diag", n_init=10, random_state=0).fit(S)
df["cluster"] = gmm.predict(S)
prof = pd.DataFrame(gmm.means_, columns=STYLE)
print("BIC by k:", {k: round(v) for k, v in bics.items()}, "-> using", k_best)
print(prof.round(2).to_string())

pca = PCA(2, random_state=0).fit(S)
P = pca.transform(S)
print("PCA variance explained:", pca.explained_variance_ratio_.round(3))
print("PC loadings:\n", pd.DataFrame(pca.components_, columns=STYLE, index=["PC1", "PC2"]).round(2).T.to_string())

# ---------- 6. validation ----------
lab = df.Drafted.map({"Y": 1, "N": 0})
# only use comps with a known outcome; score = share of comps that were drafted
known = lab.notna().values
Dk = D.copy(); Dk[:, ~known] = np.inf
nn = np.argsort(Dk, axis=1)[:, :15]
comp_rate = lab.values[nn].astype(float).mean(1)
mask = known
auc = roc_auc_score(lab[mask], comp_rate[mask])
base = lab[mask].mean()
# Simple baseline for comparison: raw rushing yards
auc_yds = roc_auc_score(lab[mask], df.YDS[mask])
print(f"Comps-based draft AUC = {auc:.3f}  (rushing-yards baseline AUC = {auc_yds:.3f}, base draft rate {base:.2%})")

# draft rate by archetype (completed careers only)
df["lab"] = lab
print(df.groupby("cluster").agg(n=("Player", "size"), draft_rate=("lab", "mean")).round(3))

df["_pc1"], df["_pc2"] = P[:, 0], P[:, 1]
df.to_pickle(OUT + ".pkl")
np.save(OUT + ".comps.npy", comp_idx); np.save(OUT + ".dist.npy", np.take_along_axis(D, comp_idx, 1))
json.dump({"ref": float(ref), "auc": auc, "auc_yds": auc_yds, "base": float(base), "k": int(k_best),
           "groups": GROUPS}, open(OUT + ".meta.json", "w"))

# ---------- 6b. stricter, time-respecting check ----------
# A comp may only come from an EARLIER season (as if predicting in real time).
earlier = df.Season.values[None, :] < df.Season.values[:, None]
Dt = D.copy(); Dt[~earlier | ~known[None, :]] = np.inf
nn_t = np.argsort(Dt, axis=1)[:, :15]
ok = known & (df.Season.values >= 2016)            # need a couple seasons of history first
rate_t = lab.values[nn_t].astype(float).mean(1)
auc_t = roc_auc_score(lab[ok], rate_t[ok]); auc_t_yds = roc_auc_score(lab[ok], df.YDS[ok])
print(f"Time-respecting comps AUC (2016+) = {auc_t:.3f}  vs yards baseline {auc_t_yds:.3f}")
m = json.load(open(OUT + ".meta.json")); m.update(auc_t=auc_t, auc_t_yds=auc_t_yds); json.dump(m, open(OUT + ".meta.json", "w"))