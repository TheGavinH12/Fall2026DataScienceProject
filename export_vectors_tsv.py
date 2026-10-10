"""
Export model vectors for projector.tensorflow.org (Euclidean distance there = the model's distance).
    python export_vectors_tsv.py            -> final tool (find_dgg_comps): all seasons + freshman seasons
Writes vectors_<name>.tsv (no header) + metadata_<name>.tsv (header) pairs.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.argv = sys.argv[:1]
import find_dgg_comps as f
OUT = os.environ.get("OUT", ".")
df, allrows = f.load_workbook(os.environ.get("WB", "Data_Science_PLAYER_Data_Roles.xlsx"))
stats, latest = f.season_stats(df)
pool = df[df.usable].reset_index(drop=True)
Z, _ = f.zscore(pool, stats)
coef = np.array([f.WEIGHTS[g] / len(fs) for g, fs in f.GROUPS.items() for _ in fs])
X = Z * np.sqrt(coef)                       # so plain Euclidean distance = tool distance
last = allrows.sort_values("Season").groupby("PlayerId").last()
def outcome(pid):
    r = last.loc[pid]
    if r.Drafted == "Y" and pd.notna(r["NFL Draft Pick (overall)"]): return f"Rd {int(r['NFL Draft Round'])} #{int(r['NFL Draft Pick (overall)'])}"
    return "Undrafted" if r.Drafted == "N" else "Still in college"
meta = pd.DataFrame({"Label": [f"{p} ({s})" for p, s in zip(pool.Player, pool.Season)], "Player": pool.Player, "Season": pool.Season,
    "Team": pool.Team, "Conference": pool.Conference, "Class": pool["Class Year"],
    "Stage": np.where(pool.is_freshman, "Freshman", np.where(pool.is_first, "First featured", "Later")),
    "RB_Role": pool["RB Role"], "Carries": pool.CAR.astype(int), "YPC": pool.YPC.round(1), "PFF_Run_Grade": pool["PFF Rushing Grade"],
    "Schedule": pool["strength of schedule"].round(2), "Stars": pool["Star Rating #"].fillna(0).astype(int).replace(0, "Unranked"),
    "Outcome": [outcome(p) for p in pool.PlayerId], "Tennessee": np.where(pool.Team == "Tennessee", "Yes", "No")})
for c in ("Label", "Player", "Team", "Conference"): meta[c] = meta[c].astype(str).str.replace(r"[\t\n]", " ", regex=True)
for name, m in (("final_all", np.ones(len(pool), bool)), ("final_freshman", pool.is_freshman.values)):
    np.savetxt(os.path.join(OUT, f"vectors_{name}.tsv"), X[m], delimiter="\t", fmt="%.5f")
    meta[m].to_csv(os.path.join(OUT, f"metadata_{name}.tsv"), sep="\t", index=False)
    print(name, int(m.sum()), "rows,", X.shape[1], "dims")
