# RB Comps: Finding a Running Back's Closest College Comps

**Volunteer Vision** · Data Science Club, Fall 2026 · Gavin Hurst

When a running back finishes a college season, which backs from the last decade looked most like him at the same stage, and what happened to them?

`find_dgg_comps.py` answers that using 2,000+ FBS running back seasons (2014–2026). It was built for one case study: **David Gabriel-Georges**, a 5-star running back recruited by Tennessee. When his 2027 freshman season is complete, his stats go into `dgg_input.csv` and the model returns his five closest freshman-season comps, how confident each match is, and what each comp went on to do.

---

## Quick start

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# a back already in the data
.venv/bin/python find_dgg_comps.py --player "Saquon Barkley" --season 2017 --stage any

# a back typed into a CSV (example row: Bo Jackson 2025)
.venv/bin/python find_dgg_comps.py example_input.csv

# DGG, after his season
.venv/bin/python find_dgg_comps.py dgg_input.csv
```

> **Data not included.** The model reads `Data_Science_PLAYER_Data_Roles.xlsx`, which contains PFF grades (subscription data that can't be redistributed). Place your own copy in the project folder. `role_flags.py` adds the RB role and carry-share columns to it from PFF rushing exports.

### Example output

```
Saquon Barkley (Penn State, 2017): 217 car, 5.9 ypc, PFF 84.3   | compared with: any seasons 2014-2026

  #   conf  comp                    season  team                  car   ypc   PFF
  1  100%  Breece Hall             2021    Iowa State            253   5.8  87.0  PLAYER COMP
  2  100%  Ollie Gordon            2023    Oklahoma State        285   6.1  90.4  PLAYER COMP
  ...
  What they became:
  1. Breece Hall: 3 featured seasons (2019-2021): ... Drafted 2022 Rd 2, #36
```

---

## How it works

1. **Every playing-style stat in the workbook**, grouped and weighted:

   | Group | Stats | Weight |
   |---|---|---|
   | Size | height, weight | 0.3 |
   | Efficiency | YPC, designed-rush YPC, success rate, EPA per rush, PFF rushing grade | 1.2 |
   | Run style | explosive-run rate, stuffed rate, breakaway %, yards after contact/att, missed tackles per carry, elusive rating | 1.2 |
   | Production | carries, yards and TDs per game, goal-line share, RB carry share | 1.0 |
   | Receiving | targets, receptions and receiving yards per game, pass-block grade | 1.0 |
   | Schedule | strength of schedule | 1.0 |
   | Recruiting | 247 composite rating (blank = outside the top-50 RB lists) | 0.3 |

   Draft and NFL results are **never** used for matching. They only appear afterward, under "What they became."

2. **Season-relative.** Each stat is a z-score against all FBS backs that season. A season not in the workbook (e.g. 2027) is measured against the latest full season.
3. **Small samples pulled toward average.** Rate stats are shrunk by `carries / (carries + 100)`.
4. **Stage matching.** By default a freshman is compared only with other backs' freshman seasons (true or redshirt).
5. **Nearest neighbors.** Weighted Euclidean distance; each comp player appears once, at his closest-matching season.
6. **Confidence.** The search is re-run 300 times with realistic noise added to every stat. Confidence = how often a comp stays in the top list.

| Option | What it does |
|---|---|
| `--top 5` | Number of comps |
| `--stage freshman \| first \| any` | Which seasons to compare against (default `freshman`) |
| `--role` | Compare starters with starters, backups with backups |
| `--no-recruiting` | Leave the recruiting rating out |
| `--player "Name" --season YEAR` | Comp a back already in the workbook |
| `--validate` | Run the accuracy check below |

---

## Reading the results

| Confidence | Meaning |
|---|---|
| **90%+** | Player comp: "his season compares to X" |
| **70–89%** | Strong comp |
| **under 70%** | Same general group only |

The same comp at 70%+ in **two or more seasons** supports a *college career* comp. Example: Breece Hall is Saquon Barkley's top comp in both 2016 and 2017.

A comp means two backs **produced** the same way in college, not that they'll have the same career. The data has no athletic testing, film or scheme information. Saquon's comps went in Rounds 2–6, and he went #2 overall.

---

## Validation

`--validate` tests 190 freshman seasons since 2016. Each freshman's comps come only from **earlier** classes, and the comps' draft rate is used to predict whether he was eventually drafted:

| Model | AUC |
|---|---|
| Rushing yards only | 0.70 |
| All categories | **0.79** |

During development, five earlier versions were each kept or dropped based on held-out tests. That's how strength of schedule, role matching and confidence scores made it in, and how a recency-weighting idea that didn't help overall was left out. In a test on 753 backs' final college seasons, the comps ranked actual draft outcomes far better than rushing yards (0.63 vs 0.29 rank correlation). Known blind spots: top picks are underrated, and backups behind strong starters are missed most often. `RB_Comps_Case_Log.docx` documents the hits and misses.

---

## Data

`Data_Science_PLAYER_Data_Roles.xlsx`: one sheet per season, one row per running back with roughly 90+ carries.

| Source | Used for |
|---|---|
| CollegeFootballData / cfbfastR play-by-play | success rate, EPA, explosive and stuffed rate, goal-line carries, class, size |
| PFF rushing exports (2014–2025) | grades, yards after contact, missed tackles, breakaway %, elusive rating, receiving |
| 247Sports composite | recruiting rating |
| nflverse | draft results (shown only, never matched on) |

## Tools

Python · pandas · NumPy · scikit-learn · openpyxl
