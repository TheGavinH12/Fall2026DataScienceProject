"""
Print a back's comps with confidence.
    .venv/bin/python show_comps.py "Makhi Hughes" 2023
    .venv/bin/python show_comps.py "Makhi Hughes" 2023 v3     (pick a version: v2, v3, v4, v5)
Uses the newest model it finds: v5 (role) > v4 (recency) > v3 (schedule) > v2.
"""
import json, os, sys
import pandas as pd

if len(sys.argv) < 3:
    sys.exit('Usage: python show_comps.py "Player Name" SEASON')
df = pd.read_pickle("comps/out.pkl").reset_index(drop=True)
newest = next((v for v in ("v5", "v4", "v3", "v2") if os.path.exists(f"comps/out_{v}.json")), "v2")
ver = sys.argv[3] if len(sys.argv) > 3 else newest
comps = json.load(open(f"comps/out_{ver}.json"))["comps"]
name, season = sys.argv[1], int(sys.argv[2])

hit = df.index[(df.Player.str.lower() == name.lower()) & (df.Season == season)]
if len(hit) == 0:
    seasons = sorted(df.Season[df.Player.str.lower() == name.lower()].tolist())
    sys.exit(f"No row for {name} in {season}. Seasons on file: {seasons or 'none (check spelling)'}")
i = hit[0]
sos = df["strength of schedule"]
print(f"\n{df.Player[i]} {season} ({df.Team[i]}) - {int(df.CAR[i])} car, {df.YPC[i]} ypc, PFF {df['PFF Rushing Grade'][i]}, schedule {sos[i]:.2f}   [model {ver}]\n")
role = df["RB Role"] if "RB Role" in df else None
if role is not None:
    print(f"  role: {role[i]} ({df['RB Carry Share'][i]:.0%} of team RB carries, leader {df['Team Carry Leader'][i]})\n")
print(f"  {'conf':>5}  {'comp':<24} {'year':<5} {'team':<20} {'sched':>5}  closeness")
for j, conf, close in comps[i]:
    tag = "  STRONG" if conf >= 70 else ""
    print(f"  {conf:>4}%  {df.Player[j]:<24} {df.Season[j]:<5} {df.Team[j]:<20} {sos[j]:>5.2f}  closer than {close}%{tag}")
print()
