"""
find_dgg_comps.py - Volunteer Vision: find a running back's 5 closest college comps
====================================================================================
Built to answer one question: when David Gabriel-Georges (DGG) plays his first college
season, which backs from 2014-2026 looked most like him at the same stage?

USE
  1. Fill in dgg_input.csv (one row per player; blanks are allowed, see below)
  2. .venv/bin/python find_dgg_comps.py dgg_input.csv
  Options
    --workbook FILE     player workbook (default: Data_Science_PLAYER_Data_Roles.xlsx)
    --top 5             how many comps to show
    --stage freshman    compare with backs' FRESHMAN seasons (true or redshirt freshman, default)
            first       compare with each back's first featured season (90+ carries), any class
            any         compare with every season 2014-2026
    --role              only comp to backs with the same RB Role (Starter / Split / Backup)
    --no-recruiting     leave star rating out of the match
    --player "Name" --season 2025   comp a back who is already in the workbook (for testing)
    --validate          print how well freshman comps predict later outcomes (sanity check)

WHAT IT COMPARES (every playing-style column in the workbook; draft/NFL columns are NOT used)
  Size          Height, Weight
  Efficiency    YPC, Designed Rush YPC, Rush Success Rate, EPA per Rush
  PFF grade     PFF Rushing Grade (its own group with the highest single weight: it sums up every run)
  Run style     Explosive Run Rate, Stuffed Rate (lower is better), PFF Breakaway %,
                PFF Yards After Contact per Att, Missed Tackles Forced per carry, PFF Elusive Rating
  Production    Carries/game, Rush yards/game, Rush TD/game, Goal-line carry share, RB Carry Share
  Receiving     Targets/game, Receptions/game, Receiving yds/game, PFF Pass Block Grade
  Schedule      strength of schedule
  Recruiting    247 composite rating (Star Rating); blank = outside the top-50 RB lists
  Each stat is converted to a z-score against all FBS backs in the same season, so a 2015
  freshman is measured against 2015 backs. A season from a year not in the workbook (e.g. 2027)
  is measured against the most recent full season.
  Low-carry seasons are pulled toward average (a 100-carry season counts half), because rate
  stats on few carries are noisy.

CONFIDENCE
  The match is re-run 300 times with realistic noise added to every back's stats (more noise for
  fewer carries). Confidence = share of runs in which that comp stayed in the top 5.
  90%+ = a player comp; 70-89% = strong; under 70% = same general group only.

BLANKS IN THE CSV
  Any stat left blank is filled with that season's median and the output lists what was filled.
  The fewer blanks, the better the comp. Minimum: CAR, YDS, Games Played.
"""
import argparse, json, re, sys, unicodedata, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

GROUPS = {
    "Size":       ["height_in", "Weight"],
    "Efficiency": ["YPC", "Designed Rush YPC", "Rush Success Rate", "EPA per Rush"],
    "PFF grade":  ["PFF Rushing Grade"],
    "Run style":  ["Explosive Run Rate (10+ yd)", "stuffed_inv", "PFF Breakaway %", "PFF Yards After Contact per Att",
                   "mtf_rate", "PFF Elusive Rating"],
    "Production": ["car_pg", "yds_pg", "td_pg", "gl_share", "RB Carry Share"],
    "Receiving":  ["tgt_pg", "rec_pg", "recyds_pg", "PFF Pass Block Grade"],
    "Schedule":   ["strength of schedule"],
    "Recruiting": ["Star Rating"],
}
# PFF grade is its own group: it sums up every run, and giving it extra weight improved the freshman
# validation from 0.79 to 0.82 AUC (pff_weight_check.py). 1.24 = its old share (1.2 / 5) + 1.0 extra.
WEIGHTS = {"Size": 0.3, "Efficiency": 0.96, "PFF grade": 1.24, "Run style": 1.2, "Production": 1.0, "Receiving": 1.0,
           "Schedule": 1.0, "Recruiting": 0.3}
RATE = set(GROUPS["Efficiency"] + GROUPS["PFF grade"] + GROUPS["Run style"] + GROUPS["Receiving"])   # shrunk for low carries
INPUT_COLS = ["Player", "Season", "Team", "Conference", "Class Year", "Redshirt", "Height", "Weight",
              "CAR", "YDS", "TD", "LONG", "Games Played", "Designed Rush Att", "Designed Rush YPC",
              "Rush Success Rate", "EPA per Rush", "Explosive Run Rate (10+ yd)", "Stuffed Rate", "Goal-Line Carries",
              "PFF Rushing Grade", "PFF Yards After Contact per Att", "PFF Missed Tackles Forced", "PFF Breakaway %",
              "PFF Elusive Rating", "Targets", "Receptions", "Receiving Yds", "PFF Pass Block Grade",
              "strength of schedule", "Team RB Carries", "RB Role", "Star Rating"]
SHRINK_K, UDFA, RUNS = 100, 260, 300


def h2in(h):
    m = re.match(r"^\s*(\d)\s*-\s*(\d{1,2})\s*$", str(h))
    if m: return int(m.group(1)) * 12 + int(m.group(2))
    try: return float(h) if float(h) > 50 else np.nan
    except Exception: return np.nan

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[.'\-]", " ", s).split())

def engineer(d):
    """raw sheet/CSV columns -> model features (same math for workbook rows and CSV rows)"""
    d = d.copy()
    num = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)
    car, gp = num("CAR"), num("Games Played").clip(lower=1)
    d["height_in"] = d["Height"].map(h2in) if "Height" in d else np.nan
    d["Weight"] = num("Weight")
    if "YPC" not in d or d["YPC"].isna().all(): d["YPC"] = num("YDS") / car
    else: d["YPC"] = num("YPC").fillna(num("YDS") / car)
    for c in ["Designed Rush YPC", "Rush Success Rate", "EPA per Rush", "PFF Rushing Grade", "Explosive Run Rate (10+ yd)",
              "PFF Breakaway %", "PFF Yards After Contact per Att", "PFF Elusive Rating", "PFF Pass Block Grade",
              "strength of schedule", "Star Rating"]:
        d[c] = num(c)
    d["stuffed_inv"] = -num("Stuffed Rate")
    d["mtf_rate"] = num("PFF Missed Tackles Forced") / car
    d["car_pg"], d["yds_pg"], d["td_pg"] = car / gp, num("YDS") / gp, num("TD") / gp
    d["gl_share"] = num("Goal-Line Carries") / car
    share = num("RB Carry Share") if "RB Carry Share" in d else pd.Series(np.nan, index=d.index)
    d["RB Carry Share"] = share.fillna(car / num("Team RB Carries"))
    d["tgt_pg"], d["rec_pg"], d["recyds_pg"] = num("Targets") / gp, num("Receptions") / gp, num("Receiving Yds") / gp
    d["CAR"] = car
    return d

def load_workbook(path):
    book = pd.read_excel(path, sheet_name=None)
    allrows = pd.concat([d[d.Player.notna()] for s, d in book.items() if s.isdigit()], ignore_index=True)
    first_season = allrows.groupby("PlayerId").Season.min()
    df = allrows[(allrows.Season >= 2014) & (allrows.Season <= 2026)].copy().reset_index(drop=True)
    df = engineer(df)
    df["first_sheet_season"] = df.PlayerId.map(first_season)
    cls = df["Class Year"].astype(str).str.upper()
    df["is_first"] = df.Season == df.first_sheet_season
    df["is_freshman"] = (cls == "FR") | ((cls == "SO") & (df.Redshirt.astype(str).str.upper() == "Y") & df.is_first)
    df["usable"] = df["PFF Rushing Grade"].notna()          # rows without PFF data stay out of the comp pool
    return df, allrows

FEATS = [f for g in GROUPS.values() for f in g]

def season_stats(df):
    """mean/sd per season per feature; small seasons borrow the latest full season"""
    full = df[df.usable].groupby("Season").filter(lambda g: len(g) >= 100)
    latest = full.Season.max()
    stats = {}
    for s in range(2014, 2031):
        src = full[full.Season == s] if s in set(full.Season) else full[full.Season == latest]
        stats[s] = (src[FEATS].median(), src[FEATS].mean(), src[FEATS].std(ddof=0).replace(0, 1))
    # Recruiting describes a high-school class, not on-field play, so it uses ONE fixed scale for all seasons,
    # built from every back (unranked = 0.80). Scaling it within each season among top-50 recruits only
    # turned small composite gaps (0.84 vs 0.89) into ~2 standard deviations.
    star = df.loc[df.usable, "Star Rating"].fillna(0.80)
    for s in stats:
        stats[s][1]["Star Rating"], stats[s][2]["Star Rating"] = star.mean(), star.std(ddof=0)
    return stats, latest

def zscore(d, stats):
    Z = np.zeros((len(d), len(FEATS)), np.float32); filled = []
    for k, (_, r) in enumerate(d.iterrows()):
        med, mu, sd = stats[int(r.Season)]
        vals = r[FEATS].astype(float)
        miss = [f for f in FEATS if np.isnan(vals[f])]
        if np.isnan(vals["Star Rating"]):
            vals["Star Rating"] = 0.80     # outside the top-50 RB lists: below every listed rating (must run before fillna)
        vals = vals.fillna(med)
        z = ((vals - mu) / sd).clip(-3, 3)
        w = r.CAR / (r.CAR + SHRINK_K) if r.CAR == r.CAR else 0.5
        for f in RATE: z[f] *= w
        Z[k] = z.values; filled.append([m for m in miss if m != "Star Rating"])
    return Z, filled

GIDX = {g: [FEATS.index(f) for f in fs] for g, fs in GROUPS.items()}
def distances(q, P, weights):
    """q: (m, F) queries, P: (n, F) pool -> (m, n) weighted distance"""
    tot = 0
    for g, ix in GIDX.items():
        if weights.get(g, 0): tot = tot + weights[g] * ((q[:, None, ix] - P[None, :, ix]) ** 2).mean(-1)
    return np.sqrt(tot)

def career(allrows, pid):
    r = allrows[allrows.PlayerId == pid].sort_values("Season")
    last = r.iloc[-1]
    if last.Drafted == "Y" and pd.notna(last["NFL Draft Pick (overall)"]):
        out = f"Drafted {int(last['NFL Draft Year'])} Rd {int(last['NFL Draft Round'])}, #{int(last['NFL Draft Pick (overall)'])}"
    elif last.Drafted == "N": out = "Undrafted"
    else: out = "Still in college"
    pff = pd.to_numeric(r["PFF Rushing Grade"], errors="coerce")
    best = f", best PFF {pff.max():.1f}" if pff.notna().any() else ""
    return (f"{len(r)} featured season{'s' if len(r) > 1 else ''} ({r.Season.min()}-{r.Season.max()}): "
            f"{int(r.CAR.sum())} car, {int(r.YDS.sum())} yds, {int(r.TD.sum())} TD{best}. {out}")

def run(queries, df, allrows, stats, args, exclude_pid=None):
    weights = dict(WEIGHTS)
    if args.no_recruiting: weights["Recruiting"] = 0
    pool_mask = df.usable.values & (df.Season.values <= 2026)
    if args.stage == "freshman": pool_mask &= df.is_freshman.values
    elif args.stage == "first": pool_mask &= df.is_first.values
    pool = df[pool_mask].reset_index(drop=True)
    P, _ = zscore(pool, stats)
    Q, filled = zscore(queries, stats)
    rng = np.random.default_rng(7)
    car_p = pool.CAR.values.astype(float); car_q = queries.CAR.values.astype(float)
    for qi, (_, q) in enumerate(queries.iterrows()):
        block = np.zeros(len(pool), bool)
        if exclude_pid is not None: block |= pool.PlayerId.values == exclude_pid
        if args.role and isinstance(q.get("RB Role"), str): block |= pool["RB Role"].values != q["RB Role"]
        d0 = distances(Q[qi:qi + 1], P, weights)[0]; d0[block] = np.inf
        counts = {}
        pids = pool.PlayerId.values
        for _ in range(RUNS):
            Pn = P + rng.normal(0, 1, P.shape).astype(np.float32) * (0.25 * np.sqrt(150 / car_p))[:, None]
            Qn = Q[qi:qi + 1] + rng.normal(0, 1, (1, P.shape[1])).astype(np.float32) * 0.25 * np.sqrt(150 / max(car_q[qi], 30))
            for ix in (GIDX["Size"] + GIDX["Schedule"] + GIDX["Recruiting"]):     # not noisy
                Pn[:, ix] = P[:, ix]; Qn[:, ix] = Q[qi:qi + 1, ix]
            dn = distances(Qn, Pn, weights)[0]; dn[block] = np.inf
            # count PLAYERS, not seasons: one comp player can't take two of the top spots
            seen = []
            for j in np.argsort(dn):
                if not np.isfinite(dn[j]) or len(seen) == args.top: break
                if pids[j] not in seen: seen.append(pids[j])
            for p in seen: counts[p] = counts.get(p, 0) + 1
        finite = d0[np.isfinite(d0)]
        # show each comp player at his season closest to the query (noise-free distance)
        best = {}
        for j in np.where(np.isfinite(d0))[0]:
            if pids[j] not in best or d0[j] < d0[best[pids[j]]]: best[pids[j]] = j
        conf = np.zeros(len(pool))
        for p, c in counts.items(): conf[best[p]] = c / RUNS
        order = sorted((best[p] for p in counts), key=lambda j: (-conf[j], d0[j]))[:args.top]
        if len(order) < args.top:
            order += sorted((j for p, j in best.items() if j not in order), key=lambda j: d0[j])[:args.top - len(order)]
        # ----- print -----
        print("\n" + "=" * 92)
        print(f"{q.Player} ({q.get('Team', '')}, {int(q.Season)}): {int(q.CAR)} car, {q.YPC:.1f} ypc"
              + (f", PFF {q['PFF Rushing Grade']:.1f}" if q['PFF Rushing Grade'] == q['PFF Rushing Grade'] else "")
              + f"   | compared with: {args.stage} seasons 2014-2026 ({len(pool)} backs)"
              + (f", same role ({q['RB Role']})" if args.role and isinstance(q.get('RB Role'), str) else ""))
        if int(q.Season) not in set(df[df.usable].Season) or (df[df.usable].Season == q.Season).sum() < 100:
            print(f"  note: {int(q.Season)} measured against the {stats_latest} season (latest full season in the workbook)")
        if filled[qi]: print(f"  filled with season median (blank in input): {', '.join(filled[qi])}")
        if q.CAR < 90: print(f"  note: {int(q.CAR)} carries is below the ~90-carry level of the backs in the pool; treat as a rough comp")
        print(f"\n  {'#':<3}{'conf':>5}  {'comp':<24}{'season':<8}{'team':<20}{'car':>5}{'ypc':>6}{'PFF':>6}  closeness")
        for rank, j in enumerate(order, 1):
            c = pool.iloc[j]
            close = 100 * (finite > d0[j]).mean()
            tag = "PLAYER COMP" if conf[j] >= .9 else ("strong" if conf[j] >= .7 else "")
            print(f"  {rank:<3}{conf[j]:>4.0%}  {c.Player:<24}{int(c.Season):<8}{str(c.Team)[:19]:<20}{int(c.CAR):>5}{c.YPC:>6.1f}"
                  f"{c['PFF Rushing Grade']:>6.1f}  closer than {close:.1f}%  {tag}")
        print("\n  What they became:")
        for rank, j in enumerate(order, 1):
            print(f"  {rank}. {pool.iloc[j].Player}: {career(allrows, pool.iloc[j].PlayerId)}")
        drafted = [allrows[allrows.PlayerId == pool.iloc[j].PlayerId].iloc[-1] for j in order]
        known = [r for r in drafted if r.Drafted in ("Y", "N")]
        if known:
            dr = [r for r in known if r.Drafted == "Y"]
            print(f"\n  {len(dr)} of {len(known)} comps with a final outcome were drafted"
                  + (f" (picks {', '.join(str(int(r['NFL Draft Pick (overall)'])) for r in dr)})" if dr else "") + ".")

def validate(df, stats):
    """Do freshman comps carry signal? Comps only from EARLIER seasons; score = comps' draft rate."""
    from sklearn.metrics import roc_auc_score
    fr = df[df.usable & df.is_freshman].reset_index(drop=True)
    last = df.groupby("PlayerId").Drafted.last()
    lab = fr.PlayerId.map(last).map({"Y": 1, "N": 0}).values.astype(float)
    P, _ = zscore(fr, stats)
    test = np.where(~np.isnan(lab) & (fr.Season.values >= 2016))[0]
    def auc(weights, K=8):
        D = distances(P, P, weights); sc = []
        for i in test:
            ok = (fr.Season.values < fr.Season.values[i]) & ~np.isnan(lab) & (fr.PlayerId.values != fr.PlayerId[i])
            c = np.where(ok)[0]; c = c[np.argsort(D[i, c])[:K]]; sc.append(lab[c].mean())
        return roc_auc_score(lab[test], sc)
    yz = ((fr.YDS - fr.groupby("Season").YDS.transform("mean")) / fr.groupby("Season").YDS.transform("std")).values
    print(f"\nValidation on {len(test)} freshman seasons 2016+ with a known draft outcome "
          f"({int(np.nansum(lab[test]))} drafted). Comps come only from earlier freshman classes.")
    print(f"  rushing yards alone (AUC):      {roc_auc_score(lab[test], yz[test]):.3f}")
    print(f"  all categories:                 {auc(WEIGHTS):.3f}")
    print(f"  all categories, no recruiting:  {auc(dict(WEIGHTS, Recruiting=0)):.3f}")
    print(f"  without schedule:               {auc(dict(WEIGHTS, Schedule=0)):.3f}")
    print("  (0.5 = coin flip, 1.0 = perfect. This checks that freshman comps carry real signal; it is not the goal of the tool.)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Find a running back's closest college comps.")
    ap.add_argument("csv", nargs="?", help="input CSV (see dgg_input.csv)")
    ap.add_argument("--workbook", default="Data_Science_PLAYER_Data_Roles.xlsx")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--stage", choices=["freshman", "first", "any"], default="freshman")
    ap.add_argument("--role", action="store_true")
    ap.add_argument("--no-recruiting", action="store_true")
    ap.add_argument("--player"); ap.add_argument("--season", type=int)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    df, allrows = load_workbook(args.workbook)
    stats, stats_latest = season_stats(df)
    if args.validate: validate(df, stats)
    if args.player:
        hit = df[(df.Player.map(norm) == norm(args.player)) & (df.Season == args.season)]
        if hit.empty: sys.exit(f"{args.player} {args.season} not found in the workbook.")
        run(hit, df, allrows, stats, args, exclude_pid=hit.PlayerId.iloc[0])
    if args.csv:
        raw = pd.read_csv(args.csv, dtype=str).dropna(how="all")
        raw = raw[raw.Player.notna() & raw.CAR.notna()]
        if raw.empty: sys.exit("No rows with a Player and CAR in the CSV. Fill in at least Player, Season, CAR, YDS, Games Played.")
        for c in raw.columns:
            if c not in ("Player", "Team", "Conference", "Class Year", "Redshirt", "Height", "RB Role"):
                raw[c] = pd.to_numeric(raw[c].str.replace("%", ""), errors="coerce")
        for c in ("Rush Success Rate", "Explosive Run Rate (10+ yd)", "Stuffed Rate", "PFF Breakaway %"):
            if c in raw: raw[c] = raw[c].where(~(raw[c] > 1), raw[c] / 100)   # accept 45 or 0.45
        q = engineer(raw)
        q["Season"] = q.Season.fillna(stats_latest + 1).astype(int)
        run(q.reset_index(drop=True), df, allrows, stats, args)
    if not (args.csv or args.player or args.validate):
        ap.print_help()
