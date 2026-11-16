# Insurance Claim Prediction (Safe Driver) — Porto Seguro

Predict whether a policyholder will file an insurance claim in the next policy year, using the
[Porto Seguro Safe Driver Prediction](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction) data
(595,212 policyholders, 57 anonymised features, ~3.6 % positives).

Final project · *Programming for Data Analysis and Database* · Applied Artificial Intelligence (MSc) · **Topic 40**
**Team:** Bezhenar Pavel · Abay Zhenis · Alenov Abay (AAI-2604)

Everything lives in one Google Colab notebook, **`Insurance_Claim_Prediction.ipynb`**, organised in three parts. The notebook writes its own helper modules to `src/`.

---

## Results at a glance

Hold-out set: 119,043 policyholders (stratified 20 %), scored once per model. Metric = normalised Gini (2·ROC-AUC − 1), the competition metric.

| Model | ROC-AUC | Gini | PR-AUC | Fit time | Size |
|---|---|---|---|---|---|
| **LightGBM (tuned) — final** | **0.6437** | **0.2873** | 0.0689 | 23 s | 0.6 MB |
| Blend (LightGBM + XGBoost, tuned) | 0.6429 | 0.2858 | 0.0686 | 46 s | 1.3 MB |
| XGBoost (tuned) | 0.6410 | 0.2821 | 0.0681 | 23 s | 0.7 MB |
| LightGBM (default) | 0.6380 | 0.2760 | 0.0663 | 15 s | 1.0 MB |
| HistGradientBoosting | 0.6360 | 0.2719 | 0.0666 | 32 s | 1.1 MB |
| XGBoost (default) | 0.6354 | 0.2708 | 0.0672 | 20 s | 1.3 MB |
| Logistic Regression | 0.6328 | 0.2656 | 0.0651 | 5 s | 0.01 MB |
| Random Forest | 0.6325 | 0.2650 | 0.0640 | 83 s | 7.0 MB |
| Dummy (prior) | 0.5000 | 0.0000 | 0.0364 | <1 s | — |

* Final model chosen by **cross-validation** (CV-AUC 0.6412 vs 0.6410 for tuned XGBoost), not by the hold-out.
* Tuning gain is real, not noise: tuned − default LightGBM ΔAUC = **+0.0058** (95 % bootstrap CI +0.0036 … +0.0077).
* Business view: the riskiest 10 % of policyholders claim at **2.24×** the portfolio rate (8.2 % vs 3.6 %); the riskiest 30 % contain **49.5 %** of all claims.
* The 20 `ps_calc_*` features are noise (0 of 20 significant at 5 % FDR; dropping them *improved* CV-AUC), so the final model uses 37 features.
* Cross-dataset check (freMTPL2freq, French motor insurance): same recipe gives Gini 0.424 — boosting helps far more when features are interpretable.

---

## Notebook structure

| Part | Sections | What it does |
|---|---|---|
| **0 · Modules** | 0 | `%%writefile` cells create the `src/` package |
| **1 · Data engineering, SQL & EDA** | 1.0–1.6 | environment snapshot · zip upload & extraction · chunked load + dtype optimisation · CSV vs Parquet · data-quality audit · cleaning (`-1 → NaN`) · feature dictionary · SQLite schema, ER diagram, 10 SQL queries · EDA, correlations, FDR-corrected tests, Plotly explorer |
| **2 · Modelling & comparison** | 2.0–2.4 | stratified split & leakage checks · feature-set ablation · 6 model families in sklearn `Pipeline`s · Optuna tuning with CV (LightGBM, XGBoost) · bootstrap CIs, paired test · blend · cost/quality trade-off · artefacts saved |
| **3 · Interpretation, extensions, deployment** | 3.1–3.5 | gain vs permutation vs SHAP · out-of-fold thresholds, confusion matrices, cost · lift/gain vs SQL baseline · calibration · segment error analysis · freMTPL2freq extension · model bundle, interactive console menu, submission file · conclusions |

Each part reloads its inputs from disk, so the parts can be run (or split into separate notebooks) independently once the previous one has produced its files.

---

## Quick start (Google Colab)

1. Open `Insurance_Claim_Prediction.ipynb` in Colab (`File → Open notebook → GitHub / Upload`).
2. Download **`porto-seguro-safe-driver-prediction.zip`** from the Kaggle competition page.
3. `Runtime → Run all`. In section 1.1 an upload dialog opens — select the zip. `train.csv`, `test.csv` and `sample_submission.csv` are extracted to `data/raw/`.
4. Part 3's console menu asks for keyboard input when `RUN_INTERACTIVE = True` (the default in `src/config.py`). For an unattended *Run all*, set it to `False` first.

**Time budget:** Optuna dominates — in the recorded Colab run LightGBM (20 trials) took 21.7 min and XGBoost (12 trials) 45.6 min; the rest of the notebook is a matter of minutes (e.g. feature dictionary 40 s, SQLite load 10 s, permutation importance 99 s, out-of-fold predictions 57 s). Lower `N_TRIALS_LGBM` / `N_TRIALS_XGB` for a quicker run.

### Local run
```bash
pip install -r requirements.txt
jupyter lab Insurance_Claim_Prediction.ipynb     # put the zip in the working directory or set MANUAL_ZIP_PATH
```

### Troubleshooting
* **`ImportError: cannot import name 'MANUAL_ZIP_PATH'`** — add `MANUAL_ZIP_PATH = None` under `ZIP_NAME` in the `src/config.py` cell (section 0). Sections 1.0/1.1 read it.
* **`train.csv not found`** — upload the zip when the dialog opens, or place it in the working directory / Google Drive and set `MANUAL_ZIP_PATH`.
* **Upload is slow** — upload the zip to Google Drive once, mount Drive, and set `MANUAL_ZIP_PATH` to the Drive path.

---

## Repository layout
```
├── Insurance_Claim_Prediction.ipynb     # the project (3 parts)
├── src/                                 # written by the notebook (section 0)
│   ├── config.py        # paths, seed, run switches
│   ├── data_utils.py    # chunked loading, dtype downcasting, -1 handling, feature engineering, feature dictionary
│   ├── metrics.py       # Gini, PR-AUC, bootstrap CIs, lift tables, confusion at threshold
│   ├── modeling.py      # ColumnTransformer/Pipeline builders, timing, CV helpers, Optuna studies
│   ├── sql_queries.py   # schema DDL, bulk loader, 10 analytical queries
│   └── cli.py           # interactive console menu
├── requirements.txt
├── README.md
└── (generated, git-ignored)  data/  models/  reports/  figures/  submission.csv
```

### Run switches (`src/config.py`)
| Switch | Default | Meaning |
|---|---|---|
| `SEED` | 42 | all random seeds |
| `TEST_SIZE` | 0.20 | stratified hold-out share |
| `CV_FOLDS` | 3 | folds for ablation and tuning (training part only) |
| `N_TRIALS_LGBM` / `N_TRIALS_XGB` | 20 / 12 | Optuna trials |
| `RUN_EXTERNAL` | True | run the freMTPL2freq extension (downloads from OpenML) |
| `MAKE_SUBMISSION` | True | refit on all labelled rows and write `submission.csv` |
| `RUN_INTERACTIVE` | True | open the live console menu |

### Generated artefacts
| Path | Content |
|---|---|
| `data/processed/train_clean.parquet`, `feature_dictionary.parquet`, `porto.db` | cleaned data, feature dictionary, SQLite database (+ `experiments` table) |
| `models/final_model.joblib`, `lgbm_tuned`, `xgb_tuned`, `final_meta.json`, `safe_driver_bundle.joblib`, `production_model.joblib` | fitted pipelines, chosen features/parameters, bundle used by the menu |
| `reports/model_comparison.csv/.parquet`, `feature_set_ablation.csv`, `permutation_importance.csv`, `segment_auc.csv`, `external_dataset_results.csv`, `data_audit.json`, `environment.json`, `sql/q1..q10.csv`, `claim_rate_explorer.html` | tables and reports |
| `figures/*.png` | all figures |
| `submission.csv` | predictions for the 892,816 competition test rows, validated against `sample_submission.csv` |

---

## Method in brief

* **Data engineering.** Chunked CSV read with per-chunk dtype downcasting: 268 MB → 52 MB in memory (−80.5 %, verified lossless). Parquet is 8.6× faster to read than naive CSV and 6.9× smaller on disk.
* **Cleaning.** Missing values are coded `-1` in 13 features (up to 69 % for `ps_car_03_cat`); they become `NaN`. No duplicate ids/rows, no constant columns, no rows dropped; outliers kept (anonymised data, tree models).
* **SQL layer.** Normalised SQLite schema (`policyholder` + `ind/reg/car/calc_features` 1 : 1, plus `feature_family` and `feature_dictionary`), integrity and foreign-key checks, 10 queries covering JOIN, GROUP BY/HAVING, subqueries, CTEs and window functions (RANK, NTILE, LAG, running SUM).
* **No leakage.** One stratified 80/20 split before any fitting; every fitted transformer lives inside a `Pipeline`; all selection (feature set, hyper-parameters, final model, thresholds) uses CV/out-of-fold predictions on the training part; the hold-out is scored once per model. Engineered features are row-wise and stateless (`n_missing`, `ind_bin_sum`, `car13_x_reg03`, `reg_sum`).
* **Models.** Dummy, Logistic Regression, Random Forest, HistGradientBoosting, LightGBM, XGBoost (default), then Optuna-tuned LightGBM/XGBoost with native CV and early stopping (learning rate fixed at 0.05), and a probability-average blend.
* **Interpretation.** Permutation importance on hold-out (top: `ps_car_13`, `ps_ind_05_cat`, `ps_ind_03`, `ps_ind_15`, `ps_ind_17_bin`), gain importance (Spearman agreement 0.73), SHAP summary/dependence plots.
* **Decision thresholds** are chosen on out-of-fold predictions (top-10 %: 0.0606, F1-optimal: 0.0584, cost-optimal at FN:FP = 10:1: 0.0911). Under that cost ratio the gain over "flag nobody" is small (cost 42,831 vs 43,390), so the value of the score depends strongly on the real cost ratio.
* **Calibration.** The model is already well calibrated: isotonic recalibration leaves log-loss/Brier unchanged (0.1516 → 0.1518 / 0.0347).

## Known limitations
Anonymised features (no domain interpretation) · random split because there is no time/group key (drift untested) · very weak signal, so differences of a few thousandths of Gini are close to bootstrap noise · modest tuning budget · the FN/FP cost ratio is an assumption · the notebook's console menu shows probability and risk band only (no yes/no decision).

## Requirements covered (course document)
Functions, lists/dicts, loops/conditionals, file I/O, `try/except`, interactive console menu, modular `src/` package, ≥ 2 compared approaches with a baseline (CSV vs Parquet vs chunked; Pandas vs SQL; model ladder with time/size), SQL with ER diagram and 10 queries, > 500K rows, plus all Cluster 7 items (EDA, Pipeline/ColumnTransformer, leakage-free split, CV tuning, permutation + SHAP, confusion matrix, error analysis, joblib).

## Team contributions
| Member | Contribution |
|---|---|
| Bezhenar Pavel | |
| Abay Zhenis | |
| Alenov Abay | |

## References
Porto Seguro Safe Driver Prediction, Kaggle (2017) · Ke et al., *LightGBM* (NeurIPS 2017) · Chen & Guestrin, *XGBoost* (KDD 2016) · Akiba et al., *Optuna* (KDD 2019) · Lundberg & Lee, *SHAP* (NeurIPS 2017) · Dutang & Charpentier, *CASdatasets* — freMTPL2freq (OpenML 41214).

Competition data © Porto Seguro, used under the Kaggle competition rules; not redistributed here.
