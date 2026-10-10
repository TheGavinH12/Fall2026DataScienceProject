"""
Add running-back ROLE columns to every season sheet.
    python role_flags.py Data_Science_PLAYER_Data_7.xlsx pff Data_Science_PLAYER_Data_Roles.xlsx
  pff/ = folder with PFF rushing exports named rushing_2014.csv ... rushing_2025.csv

New columns (added at the far right, nothing else changes):
  Team RB Carries      all halfback carries for his team that season
  RB Carry Share       his carries / Team RB Carries (0-1)
  Team Carry Leader    the team's top halfback by carries
  RB Role              Starter = team's carry leader, next back under 80% of his carries
                       Split   = within 80% of the other top back (either side of a near-even split)
                       Backup  = under 80% of the leader's carries
  Role Source          'PFF all halfbacks' (2014-2025) or 'Sheet backs only' (before 2014)

Why: a backup's stats are capped by his role (Jadarian Price behind Jeremiyah Love).
The comps model uses RB Role to compare backups with backups and starters with starters.
"""
import os, re, sys, unicodedata
from collections import Counter, defaultdict
import openpyxl, pandas as pd
from openpyxl.styles import PatternFill, Font
from openpyxl.comments import Comment

SRC, PFF_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SPLIT = 0.8

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'\-]", " ", s); s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(s.split())
# sheet name -> PFF name, for backs who go by a nickname (found while filling the PFF columns)
ALIAS = {"Braylon McReynolds": "Bam McReynolds", "Zachary Palmer-Smith": "Zach Palmer-Smith", "Camryn Edwards": "Cam Edwards",
 "DeaMonte Trayanum": "Chip Trayanum", "Tre Wisner": "Quintrevion Wisner", "Star Thomas": "Star'Corius Thomas", "Jevyon Ducker": "Jay Ducker",
 "Daylan Smothers": "Hollywood Smothers", "Gregory DeRosiers": "Greg Desrosiers Jr.", "OJ Arnold": "Omari Arnold", "Jo'Nathan Silver": "Jo Silver",
 "Michael Washington": "Mike Washington Jr.", "Jamarion Miller": "Jam Miller", "Kay'Ron Adams": "Kay'Ron Lynch-Adams", "Jo'quavious Marks": "Woody Marks",
 "Nathan Carter": "Nate Carter", "Demie Sumo": "Demie Sumo-Karngbaye", "Joshua McCray": "Josh McCray", "Jadyn Ott": "Jaydn Ott",
 "Cam Porter": "Cameron Porter", "Cameron Skattebo": "Cam Skattebo", "Jayden Brown": "Mel Brown", "Re'Mahn Davis": "Ray Davis",
 "Jacory Merritt": "Jacory Croskey-Merritt", "Mar'Keise Irving": "Bucky Irving", "Donald Chaney Jr.": "Don Chaney", "Dre'Lyn Washington": "Dre Washington",
 "Treyson Potts": "Trey Potts", "Devon Achane": "De'Von Achane", "Cameron Davis": "Cam Davis", "Riley Tyson": "Tyson Riley",
 "Christopher Brown Jr.": "Chris Brooks", "Lala Davis": "Elijah Davis", "Zonovan Knight": "Bam Knight", "Alexander Horvath": "Zander Horvath",
 "Isaih Pacheco": "Isiah Pacheco", "Keyon Brooks": "Keyon Henry-Brooks", "Lamical Perine": "La'Mical Perine", "Malcom Davidson": "Malcolm Davidson",
 "Freddie Holly III": "Fred Holly III", "Tre Harbison": "Tre Harbison III", "Quadraiz Wadley": "Quardraiz Wadley", "Joshua Samuel": "Josh Samuel",
 "Timothy McVey": "Tim McVey", "Gregory Howell Jr.": "Buddy Howell", "LJ Scott": "L.J. Scott", "Nyheim Hines": "Nyheim Miller-Hines",
 "Jeffery Wilson": "Jeff Wilson Jr.", "Matthew Dayes": "Matt Dayes", "Storm Barrs-Woods": "Storm Woods", "Rodriguez Moore": "Rod Moore",
 "Corey Avery": "Cory Avery", "O'Shann Allison": "O'Shaan Allison", "BJ Harris": "B.J. Harris", "C.J. Hester": "CJ Hester", "DJ McKinney": "D.J. McKinney",
 "Jordan Simmons": "Jordon Simmons"}

wb = openpyxl.load_workbook(SRC)
NEW = ["Team RB Carries", "RB Carry Share", "Team Carry Leader", "RB Role", "Role Source"]
HEAD = PatternFill("solid", fgColor="7E57C2")
ROLE_FILL = {"Starter": "E8F5E9", "Split": "FFF8E1", "Backup": "EDE7F6"}
summary = Counter(); unmapped = []

for ws in wb.worksheets:
    if not ws.title.isdigit(): continue
    season = int(ws.title)
    hdr = {c.value: c.column for c in ws[1] if c.value}
    if "Player" not in hdr: continue
    rows = [r for r in range(2, ws.max_row + 1) if ws.cell(r, hdr["Player"]).value]
    if not rows: continue
    get = lambda r, k: ws.cell(r, hdr[k]).value
    base = max(hdr.values())
    cols = {}
    for k in NEW:
        cols[k] = hdr.get(k) or (base := base + 1)
        c = ws.cell(1, cols[k], k); c.fill = HEAD; c.font = Font(name="Arial", bold=True, color="FFFFFF")
    ws.cell(1, cols["RB Role"]).comment = Comment(
        "Starter = team's carry leader (next back under 80% of his carries). Split = within 80% of the other top back. "
        "Backup = under 80% of the leader's carries. 2014+ uses every halfback in the PFF export; earlier seasons use only backs on this sheet.", "Claude")

    path = os.path.join(PFF_DIR, f"rushing_{season}.csv")
    team_backs = defaultdict(list)          # sheet team -> [(name, carries), ...]
    if os.path.exists(path):
        pff = pd.read_csv(path); hb = pff[pff.position == "HB"].copy(); hb["k"] = hb.player.map(norm)
        # map sheet team -> PFF team by the teammates we can match
        votes = defaultdict(Counter)
        for r in rows:
            m = hb[hb.k == norm(ALIAS.get(get(r, "Player"), get(r, "Player")))]
            if len(m) > 1: m = m.iloc[[(m.attempts - (get(r, "CAR") or 0)).abs().argmin()]]
            if len(m) == 1 and abs(m.attempts.iloc[0] - (get(r, "CAR") or 0)) <= 25:
                votes[get(r, "Team")][m.team_name.iloc[0]] += 1
        tmap = {t: v.most_common(1)[0][0] for t, v in votes.items()}
        for t in {get(r, "Team") for r in rows}:
            if t not in tmap: unmapped.append((season, t)); continue
            sub = hb[hb.team_name == tmap[t]]
            team_backs[t] = list(zip(sub.player, sub.attempts))
        source = "PFF all halfbacks"
    else:
        source = "Sheet backs only"
    for r in rows:   # sheet backs always included (covers pre-2014 and any unmapped team)
        t = get(r, "Team")
        if not os.path.exists(path) or t not in team_backs or not team_backs[t]:
            team_backs.setdefault(t, [])
            team_backs[t].append((get(r, "Player"), get(r, "CAR") or 0))

    for r in rows:
        t, me, car = get(r, "Team"), get(r, "Player"), float(get(r, "CAR") or 0)
        backs = sorted(team_backs[t], key=lambda x: -x[1])
        total = sum(c for _, c in backs) or car
        # find "me" in the list: by name, else by carries closest
        others = [(n, c) for n, c in backs if norm(n) not in (norm(me), norm(ALIAS.get(me, me)))]
        if len(others) == len(backs):                       # name not found -> drop the closest-carry entry
            if backs:
                j = min(range(len(backs)), key=lambda i: abs(backs[i][1] - car))
                if abs(backs[j][1] - car) <= 25: others = backs[:j] + backs[j + 1:]
        top_other = max(others, key=lambda x: x[1]) if others else (None, 0)
        if car >= top_other[1]:
            leader = me
            role = "Split" if top_other[1] >= SPLIT * car and top_other[1] > 0 else "Starter"
        else:
            leader = top_other[0]
            role = "Split" if car >= SPLIT * top_other[1] else "Backup"
        share = min(car / total, 1.0) if total else None
        vals = {"Team RB Carries": int(round(max(total, car))), "RB Carry Share": round(share, 3) if share else None,
                "Team Carry Leader": leader, "RB Role": role,
                "Role Source": source if (os.path.exists(path) and t not in [u[1] for u in unmapped if u[0] == season]) else "Sheet backs only"}
        for k, v in vals.items():
            c = ws.cell(r, cols[k], v); c.font = Font(name="Arial", size=10)
        ws.cell(r, cols["RB Carry Share"]).number_format = "0.0%"
        ws.cell(r, cols["RB Role"]).fill = PatternFill("solid", fgColor=ROLE_FILL[role])
        summary[(season >= 2014, role)] += 1
    for k in NEW:
        ws.column_dimensions[openpyxl.utils.get_column_letter(cols[k])].width = 22 if k == "Team Carry Leader" else 15

# document it in Fill-In Notes
if "Fill-In Notes" in wb.sheetnames:
    n = wb["Fill-In Notes"]
    for r in range(1, n.max_row + 1):
        if n.cell(r, 1).value == "PFF columns":
            n.insert_rows(r + 1)
            n.cell(r + 1, 1, "RB Role / Carry Share")
            n.cell(r + 1, 2, "Added Oct 2026 by role_flags.py. Team RB Carries = all halfback carries for the team in the PFF export (2014-2025) or "
                             "only the backs on the sheet (before 2014). RB Carry Share = his carries / team total. RB Role: Starter = carry leader with "
                             "the next back under 80% of his carries; Split = within 80% of the other top back; Backup = under 80% of the leader. "
                             "Used by the comps model so backups are compared with backups.")
            break
wb.save(OUT)
print("role counts (2014+, role):", dict(summary))
print("teams with no PFF match (used sheet backs only):", unmapped[:20], len(unmapped))
