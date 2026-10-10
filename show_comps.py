"""
Print a back's comps with confidence.
    .venv/bin/python show_comps.py "Makhi Hughes" 2023
Needs comps/out.pkl (from RBcomps.py) and comps/out_v2.json (from RBcomp2.py).
"""
import json, sys
import pandas as pd

if len(sys.argv) < 3:
    sys.exit('Usage: python show_comps.py "Player Name" SEASON')
df = pd.read_pickle("comps/out.pkl").reset_index(drop=True)
comps = json.load(open("comps/out_v2.json"))["comps"]
name, season = sys.argv[1], int(sys.argv[2])

hit = df.index[(df.Player.str.lower() == name.lower()) & (df.Season == season)]
if len(hit) == 0:
    seasons = sorted(df.Season[df.Player.str.lower() == name.lower()].tolist())
    sys.exit(f"No row for {name} in {season}. Seasons on file: {seasons or 'none (check spelling)'}")
i = hit[0]
print(f"\n{df.Player[i]} {season} ({df.Team[i]}) - {int(df.CAR[i])} car, {df.YPC[i]} ypc, PFF {df['PFF Rushing Grade'][i]}\n")
print(f"  {'conf':>5}  {'comp':<24} {'year':<5} {'team':<20} closeness")
for j, conf, close in comps[i]:
    tag = "  STRONG" if conf >= 70 else ""
    print(f"  {conf:>4}%  {df.Player[j]:<24} {df.Season[j]:<5} {df.Team[j]:<20} closer than {close}%{tag}")
print()
