# Volyn Hunting Grounds: 26 Years of Wildlife Data — Pipeline & Analysis

*[Читати українською → README_UA.md](README_UA.md)*

A machine-learning analysis of wildlife population dynamics across hunting grounds in **Volyn Oblast, Ukraine (2000–2025)**. The project has two halves of roughly equal weight: **(1) a data-engineering pipeline** that turns 26 years of inconsistent government reports into one clean analytical dataset, and **(2) ML models** (time-series forecasting, regression, classification, clustering, anomaly detection) whose results are validated against real historical events.

> **TL;DR for reviewers.** The hard part was not the models — it was the data. Five different reporting formats across 26 years, **184 differently-named entities resolved to 75 canonical grounds**, unit shifts (ha ↔ thousand ha), and a reporting blackout during the 2023 Forestry Agency reform. The models are deliberately conservative: where the signal wasn't there (predator–prey VAR, wild-boar forecasting), the model was **rejected, not forced**. Headline finding: the 2022 hunting ban is associated with a **+18.9% roe-deer recovery** by 2025.

---

## Part 1 — The Data Pipeline (the real work)

Government Form 2-TP was filed differently in five eras. A single parser would not survive them, so the pipeline uses **five format-specific parsers** feeding a common schema:

```
raw .xls / .xlsx  (5 format eras)
        │
        ▼
┌──────────────────────────────────────────────┐
│  old_format.py    2000–2012  (state machine)  │
│  mid_format.py    2013–2017                    │
│  hybrid_2018.py   2018  (transitional year)    │
│  new_format.py    2019–2024                    │
│  new_2025.py      2025  (newest layout)        │
└──────────────────────────────────────────────┘
        │  → processed/*.csv  (year-by-year)
        ▼
  build_final.py   — entity mapping 184 → 75, species canonicalization
        │
        ▼
  data/final/*.csv  — clean analytical dataset
```

**Why five parsers.** The formats genuinely differ, not cosmetically:
- **old (2000–2012):** hosts are *columns*, species are *rows*; Section III needs a **state machine** to track the current species across sub-rows.
- **mid (2013–2017):** hosts become rows; populations on a separate `облік` sheet; harvest in hardcoded column ranges.
- **2018 (hybrid):** meta/finances like *mid*, but the `expense_breeding` metric and relocation sheets like *new*, areas still in hectares. A year that fits no single parser.
- **new (2019–2024):** sheet `8. ОП користувачів`, areas in thousand-ha, sections located by **keyword search**, not fixed indices.
- **2025:** new `Чисельність` sheet layout, species in two sub-columns; the author mislabeled the area unit (wrote ha under a "thousand-ha" header) — the parser **detects and corrects** this (÷1000).

**Robustness choices that matter under review:**
- Sections are found by keyword (`find_section_columns`) instead of magic column numbers, so a shifted column doesn't silently break parsing.
- `is_invalid_host()` strips contaminants found in real data — phone numbers and the report operator's surname leaking into the host field.
- Aggregate rows ("усього", "по області", "разом") are filtered so totals are never double-counted.

### Verified pipeline run — 2018 file (`Волинська_2018_.xls`)

Running `hybrid_2018.py` on the raw 2018 file (the hardest, transitional format) produces:

| Table | Rows | Hosts | Species | Roe-deer check |
|-------|------|-------|---------|----------------|
| populations | 2,420 | 55 | 44 | 7,819 head counted |
| harvest | 2,365 | 55 | 44 | 338 roe deer harvested (38 grounds) |
| hosts_meta | 495 | 55 | — | — |
| finances | 385 | 55 | — | — |
| relocation | 6 | — | — | — |

Host names come out clean and canonicalizable (`Городоцьке ЛГ`, `Горохівське ЛМГ`), quotes normalized, operator/aggregate rows removed.

### Data quality — reported, not hidden

The pipeline produces sparse tables, and this is stated openly rather than masked:

| Table (2018) | Valid | NaN | NaN % |
|--------------|-------|-----|-------|
| hosts_meta | 357 | 138 | 27.9% |
| finances | 203 | 182 | 47.3% |
| populations | 1,103 | 1,317 | 54.4% |
| harvest | 319 | 2,046 | **86.5%** |

**Why harvest is ~86% empty — and why that's expected.** Harvest is a 55 grounds × 44 species matrix (2,420 cells), but only a handful of species are actually hunted at any given ground. Most ground×species pairs are legitimately empty ("nothing harvested"). Verified by spot-check: roe deer shows real per-ground numbers (5, 13, 6, 9, 12 …), ducks 7,449, hare 4,264 — the parser is reading the right cells; the matrix is simply sparse.

**Known limitation (honest).** An empty cell is treated as zero, but a true `0` ("hunted nothing") and "did not report" both collapse to `NaN` and cannot be distinguished in the source. Even roe deer has 17/55 such cells. This is a property of the source data, documented here as a real constraint on harvest-based metrics.

---

## Part 2 — The Models

Each modeling decision is driven by domain reality, not metric-chasing.

### Key finding 1 — the 2022 hunting ban worked (+18.9% roe deer)
Across 18 grounds with continuous 2019–2025 reporting:

| Period | Population | vs baseline |
|--------|-----------|-------------|
| 2019–2021 baseline | 2,865 | — |
| 2022 (ban begins) | 2,903 | +1.3% |
| 2025 | 3,408 | **+18.9%** |

Forecast 2026–2028: **3,576 (conservative trend) → 4,286 (ARIMA aggregate)**. *Robustness:* a looser host set gives +17.4% on the direct 2022→2025 comparison; the pre-ban trend (2015–2021) was essentially flat, so this is not pre-existing growth.

### Key finding 2 — anomaly detection rediscovered every major event
Isolation Forest, with **no events encoded**, flagged top anomalies that map onto history: wild-boar −89% to −98% drops (2019–2020 = African Swine Fever), +200% to +667% jumps in 2023 (grounds resuming reporting after the reform), VESTA M +1678% in 2022 (a new private farm's first report). Independent validation against ground truth — the best available check for an unsupervised method.

### Key finding 3 — grounds split into two scales, no middle tier
K-Means (k=2 by silhouette) → large/established (n=22: ~303 roe deer, 157k UAH, 18 yrs) vs small/emerging (n=52: ~104 roe deer, 57k UAH, 14 yrs). Silhouette stayed below 0.31 for all k≥3 — there is genuinely no "medium" tier. *Caveat:* `n_years` is a clustering feature, so the split partly reflects reporting consistency, not only physical size.

### Methods & integrity

| Script | What it does | Integrity choice |
|--------|--------------|------------------|
| `time_series.py` | per-host `auto_arima` forecasting | auto-selected order (not hardcoded ARIMA(1,1,0)); `STABLE_HOSTS` auto-generated; MAE 16.8 (roe) / 25.9 (boar) on 2000–18 train / 19–21 test |
| `regression.py` | population forecast + investment SHAP | **`TimeSeriesSplit`** (no future→past leakage); honest negative result — finances weakly predict population change |
| `classification.py` | harvest-sustainability classifier | **`TimeSeriesSplit`** (see correction below); `StandardScaler` inside `Pipeline`; `class_weight='balanced'` |
| `clustering.py` | ground typology (K-Means + PCA) | k by silhouette+elbow; State Reserve excluded as a domain outlier (it formed a degenerate singleton, silhouette 0.78) |
| `anomaly_detection.py` | Isolation Forest event detection | `contamination` only sets the flag cutoff — it does **not** enter `score_samples`, so the anomaly *ranking* is identical across 0.02/0.05/0.10; validated against real events, not a self-referential stability metric |

#### Honest correction — classification cross-validation
The classifier originally used `StratifiedKFold(shuffle=True)`, which mixes years and lets the model train on future data to predict the past — a leak the rest of the project explicitly guards against. Switching to **`TimeSeriesSplit`** (train always strictly before test) gives the honest picture:

| Species / model | Leaky F1 (shuffle) | Honest F1 (temporal) | Inflation |
|-----------------|-------------------|---------------------|-----------|
| Roe deer / Random Forest | 0.73 | **0.57 ± 0.20** | +0.16 |
| Wild boar / Random Forest | 0.63 | **0.40 ± 0.20** | +0.23 |
| Wild boar / Logistic Regression | 0.69 | **0.64 ± 0.13** | +0.05 |

Two findings fall out of the fix: (a) the leak was worth up to **0.23 F1** — nearly a quarter of the metric; (b) under honest CV the *simpler* model (Logistic Regression) generalizes best in time, while Random Forest was the biggest beneficiary of the leak. For roe deer, two of five temporal folds have **no positive examples** (all 24 "overharvest" cases sit in later years), so the roe-deer classifier is **not well supported by the data** — stated as a limitation, not a result.

---

## Data

| File | Rows | Description |
|------|------|-------------|
| `populations_final.csv` | 45,476 | Annual counts, 51 species × 75 grounds |
| `harvest_final.csv` | 61,554 | Harvest records (heads shot, licenses) |
| `finances_final.csv` | 7,771 | Expenses by category |
| `hosts_meta_final.csv` | 8,580 | Ground metadata (area, leases) |
| `relocation_events_final.csv` | 96 | Animal translocation events |

Focal species: roe deer (Козуля), wild boar (Кабан), red deer (Олень благородний), moose (Лось), pheasant (Фазан).

## Project structure

```
hunting-volyn/
├── parser/
│   ├── old_format.py       # 2000–2012
│   ├── mid_format.py       # 2013–2017
│   ├── hybrid_2018.py      # 2018 transitional
│   ├── new_format.py       # 2019–2024
│   └── new_2025.py         # 2025
├── build_final.py          # entity mapping 184→75 + canonicalization
├── models/
│   ├── time_series.py · regression.py · classification.py
│   ├── clustering.py · anomaly_detection.py
├── data/
│   ├── processed/          # per-year parsed CSVs
│   └── final/              # clean analytical dataset
└── reports/figures/
```

## Running

```bash
pip install -r requirements.txt

# 1. parse + build the dataset
python build_final.py

# 2. run any model (species as optional CLI args)
python models/time_series.py Козуля Лось
python models/classification.py
python models/clustering.py
python models/anomaly_detection.py
```

## Stack
`pandas` · `numpy` · `scikit-learn` · `statsmodels` · `pmdarima` · `lightgbm` · `xgboost` · `shap` · `matplotlib` · `xlrd` · `openpyxl`

## Known tech debt (honest)
- `metrics_hosts_meta` / `metrics_finances` dicts are duplicated across four parsers; they belong in one config (a format change currently means editing four files).
- `mid_format.py` uses hardcoded column ranges (e.g. `(124,181,"shot_heads")`) — fine for a frozen historical format, but a structural risk if those files were ever re-exported.
- `build_final.py` drops unmapped hosts silently; it should print the unmapped set before filtering.

## Limitations
- 2023–2024 reporting is incomplete due to the Forestry Agency reform (handled explicitly).
- Forecasts are **trends**, not precise counts.
- Empty vs "not reported" are indistinguishable in harvest (both `NaN`).
- Wild-boar forecasting intentionally omitted given ASF volatility.

---
