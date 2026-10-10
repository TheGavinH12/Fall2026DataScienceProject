"""
Volunteer Vision - 3D style map of every RB season (Plotly)
===========================================================
Run after rb_comps.py and rb_comps_v2.py:
    python rb_styles_3d.py comps/out

Uses the same weighted career-profile vectors the v2 comps engine measures distance on,
squeezes them to 3 axes with PCA, and plots every back-season in an interactive 3D scatter.
Output: rb_styles_3d.html (open in any browser; drag to rotate, scroll to zoom).
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
import plotly.graph_objects as go

B = sys.argv[1] if len(sys.argv) > 1 else "comps/out"
df = pd.read_pickle(B + ".pkl").reset_index(drop=True)
P = np.load(B + "_v2_P.npy")                              # career-to-date z-score profiles
groups = json.load(open(B + ".meta.json"))["groups"]
weights = json.load(open(B + "_v2.json"))["weights"]      # learned group weights

# Weight each feature exactly the way the comps engine does, so distance in this plot
# matches distance in the comps engine (before the squeeze to 3D).
FE = [f for g in groups.values() for f in g]
fw = np.array([np.sqrt(weights[g] / len(fs)) for g, fs in groups.items() for f in fs])
X = P * fw

pca = PCA(n_components=3, random_state=0)
xyz = pca.fit_transform(X)
ve = pca.explained_variance_ratio_
print("variance explained:", ve.round(3), "total", ve.sum().round(3))

NICE = {"height_in": "height", "weight": "weight", "YPC": "yards/carry", "Rush Success Rate": "success rate",
        "EPA per Rush": "EPA/rush", "PFF Rushing Grade": "PFF grade", "Explosive Run Rate (10+ yd)": "explosive rate",
        "Stuffed Rate": "stuffed rate", "PFF Breakaway %": "breakaway %", "PFF Yards After Contact per Att": "YAC/att",
        "mtf_rate": "missed tackles/carry", "car_pg": "carries/game", "gl_share": "goal-line share",
        "tgt_share": "target share", "rec_ypg": "rec yds/game"}
axis_names = []
for k in range(3):
    load = pd.Series(pca.components_[k], index=FE)
    hi = load.sort_values(ascending=False).index[:2]
    lo = load.sort_values().index[:1]
    print(f"PC{k+1} ({ve[k]:.0%}): + {NICE[hi[0]]}, {NICE[hi[1]]} / - {NICE[lo[0]]}")
# Short names chosen from the loadings printed above (re-check them if the data changes)
AXIS = ["Efficiency", "Receiving role", "Size"]
axis_names = [f"{a} ({v:.0%})" for a, v in zip(AXIS, ve)]

STYLES = ["Receiving back", "Workhorse", "Scatback", "Power back", "Explosive creator", "Committee back"]
COLORS = ["#2f7fbf", "#3c8a4a", "#c9417e", "#7a5a3a", "#e0761a", "#8a8f99"]

def outcome(r):
    if r.Drafted == "Y":
        return f"Drafted, Rd {int(r['NFL Draft Round'])}" if pd.notna(r["NFL Draft Round"]) else "Drafted"
    return "Undrafted" if r.Drafted == "N" else "Still in college"

df["x"], df["y"], df["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
df["hover"] = [f"<b>{r.Player}</b> · {r.Season}<br>{r.Team}<br>{STYLES[r.cluster]}<br>"
               f"{int(r.CAR)} car · {r.YPC} ypc · PFF {r['PFF Rushing Grade']}<br>{outcome(r)}"
               for _, r in df.iterrows()]

fig = go.Figure()
for c, (name, col) in enumerate(zip(STYLES, COLORS)):
    d = df[df.cluster == c]
    fig.add_trace(go.Scatter3d(x=d.x, y=d.y, z=d.z, mode="markers", name=f"{name} ({len(d)})",
                               marker=dict(size=3, color=col, opacity=0.55),
                               hovertext=d.hover, hoverinfo="text"))
ut = df[(df.Team == "Tennessee") & (df.Season >= 2021)]
fig.add_trace(go.Scatter3d(x=ut.x, y=ut.y, z=ut.z, mode="markers+text", name="Tennessee, 2021+",
                           marker=dict(size=7, color="#ff8200", symbol="diamond", line=dict(color="#222", width=1)),
                           text=[f"{p.split()[-1]} '{str(s)[2:]}" for p, s in zip(ut.Player, ut.Season)],
                           textposition="top center", textfont=dict(size=10),
                           hovertext=ut.hover, hoverinfo="text"))
fig.update_layout(
    scene=dict(xaxis_title=dict(text=axis_names[0]), yaxis_title=dict(text=axis_names[1]),
               zaxis_title=dict(text=axis_names[2])),
    legend=dict(itemsizing="constant", title="Running style (click to hide)"),
    margin=dict(l=0, r=0, t=10, b=0), height=720)

fig.write_html("rb_styles_3d.html", include_plotlyjs="cdn")
fig.write_json(B + "_3d.json")
print("wrote rb_styles_3d.html")