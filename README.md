# RB Comps: Finding a Running Back's Closest College Comps

**Volunteer Vision** · Data Science Club, Fall 2026 · Gavin Hurst

When a running back finishes a college season, which backs from the last decade looked most like him at the same stage, and what happened to them?

This project answers that with a similarity model built on 2,000+ FBS running back seasons (2014–2026). It was built for one case study: **David Gabriel-Georges**, a 5-star running back recruited by Tennessee. When his 2027 freshman season is complete, his stats go into a CSV and the model returns his five closest freshman-season comps, how confident each match is, and what each comp went on to do.

---

## Quick start

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# test with a back who is already in the data
.venv/bin/python find_dgg_comps.py --player "Saquon Barkley" --season 2017 --stage any

# test the CSV route with the example row
.venv/bin/python find_dgg_comps.py example_input.csv

# DGG: fill in dgg_input.csv after his season, then
.venv/bin/python find_dgg_comps.py dgg_input.csv
```

> **Data not included.** The model reads `Data_Science_PLAYER_Data_Roles.xlsx`, which contains PFF grades (subscription data that can't be redistributed). Place your own copy of the workbook in the project folder. See [Data](#data) for the columns it needs.

### Example output

```
Saquon Barkley (Penn State, 2017): 217 car, 5.9 ypc, PFF 84.3   | compared with: any seasons 2014-2026

  #   conf  comp                    season  team                  car   ypc   PFF  closeness
  1  100%  Breece Hall             2021    Iowa State            253   5.8  87.0  PLAYER COMP
  2  100%  Ollie Gordon            2023    Oklahoma State        285   6.1  90.4  PLAYER COMP
  ...
  What they became:
  1. Breece Hall: 3 featured seasons (2019-2021): ... Drafted 2022 Rd 2, #36
```

Breece Hall was also Saquon's top comp for his 2016 season, which is the standard this project uses for calling someone a back's *college* comp (see [Reading the results](#reading-the-results)).

---

## How it works

1. **Every playing-style column in the workbook**, grouped and weighted:

   | Group | Stats | Weight |
   |---|---|---|
   | Size | height, weight | 0.3 |
   | Efficiency | YPC, designed-rush YPC, success rate, EPA per rush, PFF rushing grade | 1.2 |
   | Run style | explosive-run rate, stuffed rate, breakaway %, yards after contact/att, missed tackles per carry, elusive rating | 1.2 |
   | Production | carries, yards and TDs per game, goal-line share, RB carry share | 1.0 |
   | Receiving | targets, receptions and receiving yards per game, pass-block grade | 1.0 |
   | Schedule | strength of schedule | 1.0 |
   | Recruiting | 247 composite rating (blank = outside the top-50 RB lists) | 0.3 |

   Draft and NFL columns are **never** used for matching. They only appear afterward, under "What they became."

2. **Season-relative.** Every stat becomes a z-score against all FBS backs in that same season, so a 2015 back is compared to 2015 norms. A season not in the workbook (e.g. 2027) is measured against the latest full season.
3. **Small samples pulled toward average.** Rate stats are shrunk by `carries / (carries + 100)`, so a 100-carry season counts half.
4. **Stage matching.** By default, a freshman is compared only with other backs' freshman seasons (true or redshirt). `--stage first` uses each back's first featured season; `--stage any` uses every season.
5. **Nearest neighbors.** Weighted Euclidean distance; each comp player appears once, at his closest-matching season.
6. **Confidence.** The search is re-run 300 times with realistic noise added to every stat (more noise for fewer carries). Confidence = how often a comp stays in the top list.

### Options

| Flag | What it does |
|---|---|
| `--top 5` | Number of comps to show |
| `--stage freshman \| first \| any` | Which seasons to compare against (default `freshman`) |
| `--role` | Compare starters only with starters, backups with backups |
| `--no-recruiting` | Leave the recruiting rating out |
| `--player "Name" --season YEAR` | Comp a back already in the workbook |
| `--validate` | Print the freshman-comps sanity check |

---

## Reading the results

| Confidence | Meaning |
|---|---|
| **90%+** | Player comp: "his season compares to X" |
| **70–89%** | Strong comp: name him alongside others |
| **under 70%** | Same general group only |

- **One season, one 90%+ comp** supports a *season* comp.
- **The same comp at 70%+ in two or more seasons** supports a *college career* comp (e.g. Saquon Barkley → Breece Hall, 2016 and 2017).
- **Closeness** ("closer than 99.7% of pairs") shows how near the comp is. A back with high confidence but lower closeness is a statistical outlier: the comps are reliable but not twins (e.g. Ahmad Hardy).

**What comps are and aren't:** a comp means two backs *produced* the same way in college. It doesn't mean they'll have the same career or draft slot. The data has no athletic testing, film or scheme information.

---

## Validation

All tests use only information available at the time: a back's comps can only come from earlier seasons.

**Freshman comps** (`find_dgg_comps.py --validate`, 190 freshman seasons since 2016): the comps' draft rate predicts whether a freshman is eventually drafted.

| Model | AUC |
|---|---|
| Rushing yards only | 0.70 |
| All categories | **0.78** |

**Full evaluation of the comps engine** (`evaluate.py`, 753 backs' final college seasons, 2017–2025):

| Model | Rank correlation with actual pick | Drafted vs. undrafted AUC |
|---|---|---|
| Rushing yards only | 0.29 | 0.72 |
| Yards + strength of schedule | 0.52 | 0.85 |
| Comps (v5) | **0.63** | **0.88** |

**Known limits, found by testing:**
- **Top picks are underrated.** First-round backs are predicted about 3.6 rounds too late, because most backs go undrafted, so even elite backs' comps include undrafted players.
- **Backups behind strong starters.** Drafted backups are missed by about twice as much as starters (Jadarian Price, James Cook, Javonte Williams). Role matching (`--role`) makes the comparison fairer but doesn't fix the draft prediction.
- **Traits the data can't see.** Saquon Barkley's comps went in Rounds 2–6; he went #2 overall. Athletic testing columns are empty.

The full set of hits, misses and fixes is in `RB_Comps_Case_Log.docx`; summary numbers are in `eval_summary_v5.json`.

---

## Model history

Each version was kept only if it held up on held-out seasons. Several ideas that sounded right didn't, and the scripts record why.

| Version | Change | Result |
|---|---|---|
| v1 `RBcomps.py` | Single-season KNN comps + Gaussian-mixture running styles | Comps carry signal (draft AUC 0.78 vs 0.74 for yards) |
| audit `v1Audit.py` | Tested how stable and how close the comps really are | Comp *order* was mostly noise, which led to confidence scores |
| v2 `RBcomp2.py` | Career-to-date profiles, shrinkage, learned weights, 200-run confidence | Same accuracy, more stable lists |
| v3 `rb_comps_v3.py` | Strength of schedule | Comps now face similar competition (schedule gap 0.86 → 0.20 SD) |
| v4 `rb_comps_v4.py` | Recency weighting (decay 0.5) | Fixed the Zach Charbonnet miss, worsened Jadarian Price; overall a wash |
| v5 `rb_comps_v5.py` | RB role and carry share; backups compared with backups | Most stable version (5.4 of 8 comps kept under noise) |
| final `find_dgg_comps.py` | Every category, single seasons, freshman stage matching, CSV input | The tool for DGG |

`decay_check.py` and `v3_check.py` hold the experiments behind the v3 and v4 decisions. `evaluate.py` scores any version on every back with a known draft result.

---

## Data

`Data_Science_PLAYER_Data_Roles.xlsx`: one sheet per season (2004–2026), one row per running back with roughly 90+ carries.

| Source | Columns |
|---|---|
| CollegeFootballData / cfbfastR play-by-play | Designed rush att/YPC, success rate, EPA per rush, explosive and stuffed rate, goal-line carries, class year, redshirt, height, weight |
| PFF rushing exports (2014–2025) | Rushing grade, yards after contact, missed tackles forced, breakaway %, elusive rating, targets, receptions, receiving yards, pass-block grade, games played |
| 247Sports composite | Star rating (top-50 RBs per class) |
| nflverse | Draft year, round, pick (used only for "What they became" and validation) |
| `role_flags.py` | Team RB carries, RB carry share, team carry leader, RB role |

PFF data begins in 2014, so the model covers 2014 onward.

---

## Repository layout

```
The tool
├── find_dgg_comps.py        # the comps tool (start here)
├── dgg_input.csv            # DGG's input row (fill in after his season)
├── example_input.csv        # filled example (Bo Jackson 2025)
├── role_flags.py            # adds RB role / carry-share columns to the workbook
├── export_vectors_tsv.py    # vectors for projector.tensorflow.org
├── rb_styles_3d.py          # interactive 3D style map (Plotly)

Model history (run in order: RBcomps -> RBcomp2 -> v3/v4/v5)
├── RBcomps.py               # v1
├── RBcomp2.py               # v2
├── rb_comps_v3.py / rb_comps_v4.py / rb_comps_v5.py
├── v1Audit.py, v3_check.py, decay_check.py   # experiments behind each decision
├── evaluate.py              # scores any version on every drafted/undrafted back
├── show_comps.py            # print one back's comps from a v2-v5 run

Docs
├── RB_Comps_Case_Log.docx   # hits, misses and fixes
└── eval_summary_v5.json     # evaluation numbers
```

## Tools

Python · pandas · NumPy · scikit-learn (Gaussian mixture, PCA, ROC AUC) · SciPy · Plotly · openpyxl · TensorFlow Embedding Projector
