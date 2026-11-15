# Technical review

Audit of the project as originally written, plus a record of what was changed and re-run during portfolio preparation (November 2026). Findings come from reading the notebooks' code and saved outputs and from re-running them. Statements taken only from the report are labelled.

## 1. What was implemented (confirmed from code)

**Stack:** Python, pandas, scikit-learn, seaborn, matplotlib, tqdm. Originally run in Google Colab (Python 3.10, from saved warning paths).

### Wind (`wind_scada.ipynb`, 50,530 rows, 10-minute data, 2018)
- **Features:** `LV ActivePower (kW)`, `Wind Speed (m/s)`, `Theoretical_Power_Curve (KWh)`, `Wind Direction (°)`. No missing values in the file. Date/time parsed and year/month/day/hour columns added (not used by models).
- **Regression section:** target is `LV ActivePower (kW)`, but the feature matrix `X` is `df_scaled[features_to_scale]`, and `features_to_scale` **includes the target**. Standard scaling is fitted on all rows before a random 80/20 split (`random_state=42`).
- **Estimators:** `LinearRegression()`, `RandomForestRegressor(n_estimators=100, random_state=42)`. Metrics (MSE, RMSE, MAE) are computed on the held-out 20% in scaled units, after dropping rows flagged "IDK".
- **"IDK":** the standard deviation of per-tree Random Forest predictions, flagged when above 0.1. Linear Regression has no trees, so nothing is ever flagged for it. This is a prediction-spread heuristic, **not** the Isolation Distributional Kernel (Ting et al., KDD 2020) that the report cites.
- **Isolation Forest:** `IsolationForest(n_estimators=100, contamination=0.01)`, no seed, on four standardised features (including the target), fitted and scored on all rows. The model is unsupervised; contamination 0.01 forces about 1% of rows to be flagged.

### Solar (`Solar_Power_Generation_Data.ipynb`)
- **Data:** two plants, each with a generation table (68,778 and 67,698 rows) and a weather table (3,182 and 3,259 rows). Merged per plant on `DATE_TIME` and `PLANT_ID` only (not inverter `SOURCE_KEY`), giving 68,774 and 67,698 rows. Each weather row is therefore repeated for every inverter reporting at that timestamp. Four Plant 1 generation rows have no matching weather timestamp.
- **Missing values:** median fill for numeric columns; the loaded data has none, so this has no effect.
- **Features (8):** ambient and module temperature, irradiation, DC power, AC power, daily yield, total yield, and engineered `TEMP_DIFF` = module minus ambient temperature. `PLANT_ID` and `SOURCE_KEY` are excluded.
- **Scaling:** `StandardScaler` fitted on all rows of each plant. No train/test split exists in the solar notebook.
- **Isolation Forest:** same settings as wind (`contamination=0.01`), no seed, one object refitted for each plant.
- **"iNNE":** a custom `iNNE(BaseEstimator, OutlierMixin)` class. `fit` runs `NearestNeighbors(n_neighbors=5)`, scores each point by its mean distance to its 5 nearest neighbours (the point itself included, since it is queried against its own fitted data) and sets the threshold at the 95th percentile of those scores (`contamination=0.05`). `predict` thresholds the same score. This is a k-NN distance score. It does not build hyperspheres on random subsamples, so it is **not** the published iNNE algorithm (Bandaragoda et al., 2018).
- **Confusion matrix (Plant 1):** compares Isolation Forest flags (1%) with the distance-score flags (5%). Cell counts 189 / 499 / 3,250 / 64,836 reproduce the report's Figure 13.

## 2. The report's known questions

| Question | Finding |
|---|---|
| Method names | Report uses three different expansions of "iNNE" ("Incremental Nearest Neighbors Embedding", "Improved Nearest Neighbors Ensemble", "Inverse Nearest Neighbors Ensemble"). None matches the code, which is a kNN distance score. "IDK" is described as "I Don't Know" logic in the report's results and as Isolation Distributional Kernel in its methods; the code supports only the first reading (tree-prediction spread). |
| Tiny regression errors | **Cause confirmed: target leakage.** The target is among the model inputs, so a model only has to copy one input. Re-running the original gave MSE of about 4e-28 (Linear Regression) and 5e-9 (Random Forest), the same order as the saved results. The revised notebook removes the target from the features and the errors become hundreds of kW. Scaling and a random split are also present but are not what causes the near-zero values. |
| Detector agreement | A confusion matrix between two unsupervised detectors is agreement, not accuracy. The flag counts are set by the contamination parameters (688 = 1% and 3,439 = 5% of 68,774 rows), so the large "normal/normal" cell and the disagreement are mostly produced by those settings. The report's statement that agreement "enhances the reliability" of the detection is not supported. |
| Failure prediction vs anomaly detection | The code detects unusual observations in data it already has. No horizon, no failure labels, no forecasting evaluation. The report's "predictive maintenance" wording is aspirational. |
| Runtime and operational claims | Timings (e.g. 16.5 s for Random Forest training, about 1.8 s for solar Isolation Forest) are from single notebook runs. They do not demonstrate real-time readiness, scalability, reduced downtime or cost benefit. |

## 3. Report versus code

- **Confirmed only from the report:** the literature review and Table 1; the interpretations of Figures 9 to 12 (for example "overheating", "shading", "mechanical failures"), which cannot be verified because there are no labels; the Kaggle dataset descriptions and links; the claim that the solar models were trained after a train/test split (the solar notebook has none).
- The report calls the solar Isolation Forest scores "isolation scores" from a fit "into the scaled data", which matches the code.
- The report describes wind Isolation Forest as applying to "the same dataset scaled features", which matches the code.

## 4. Unresolved

- The report header reads "PRIME AI paper". Its origin (template or venue) is unknown, and it should not be described as published.
- Whether flagged points correspond to real faults, maintenance or sensor issues is unknown.
- The licensing terms of the Kaggle datasets are unclear (wind: "Unknown"; solar: "Data files © Original Authors"), so the data is not redistributed.

## 5. What was changed during portfolio preparation

**Editorial (no effect on results):** removed private Dropbox download cells and share tokens (including saved wget outputs); changed data paths to `../data/...`; added a notes cell at the top of each original notebook; removed the author's email address from the report PDF (public copy only, via content-stream edit, text and rendering verified); set PDF title and author metadata; added README, licence, data notes, requirements.

**Substantive, separate file:** `notebooks/wind_scada_revised.ipynb`: target removed from features, chronological 80/20 split, scaler fitted on training rows only, unscaled target, Random Forest seed 42, baseline of the manufacturer power curve. The original wind notebook is unchanged apart from the editorial edits above.

**Not changed:** all modelling code, parameters and splits in the two original notebooks; their saved Colab outputs; the report text.

**Excluded from the repository:** the datasets, the original report PDF (contains an email address), a screenshot from an unrelated third-party paper that was in the source folder, and any unrelated documents.

## 6. What was executed

Executed in an isolated virtual environment (Python 3.14.4 on macOS; pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, tqdm 4.70.1, ipykernel 7.4.0, nbconvert 7.17.2) using the Kaggle data downloaded by the author. `jupyter nbconvert --execute` ran all three notebooks without errors (the Solar notebook emits a pandas date-format warning, and plots use a non-interactive backend).

- Re-run of the original wind notebook: the regression errors had the same magnitudes as the saved Colab outputs (Linear Regression MSE about 4e-28, Random Forest about 5e-9). The runtime figures differ.
- Re-run of the solar notebook: merge sizes match (68,774 and 67,698). Isolation Forest flags 688 rows and the distance score flags 3,439 on Plant 1. A separate re-computation with seeds 0 to 2 gave "both flagged" counts of 166 to 176 instead of the report's 189, because the original was unseeded.
- The saved outputs and figures in the two original notebooks are from the original Colab run, not from the re-run.

Revised wind regression results (test = last 20% of rows, 18 Oct to 31 Dec 2018, 10,106 rows):

| Model | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| Baseline: theoretical power curve | 207.9 | 470.7 | 0.877 |
| Linear Regression | 221.3 | 414.5 | 0.905 |
| Random Forest | 266.8 | 556.6 | 0.828 |

Interpretation: one split, one turbine, no tuning. Random Forest underperforming here may reflect a late-year distribution shift or the lack of tuning; this was not investigated.

**Not verified:** the original Colab environment and package versions; behaviour on other Python or package versions; whether the Kaggle files have changed since 2024 (row counts match the report).

## 7. Recommended improvements (not implemented)

1. Seed all stochastic steps and repeat the detectors to measure stability.
2. Rename "iNNE" and "IDK" in any write-up, or implement the published methods.
3. Merge solar tables on inverter-appropriate keys, or aggregate generation to plant level before joining weather.
4. Evaluate detectors against labelled or synthetically injected anomalies.
5. For wind regression, use time-series cross-validation and a residual-based anomaly flag.
