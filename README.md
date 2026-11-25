# Online News Popularity Prediction

## 1. Project Overview & Objective
Digital news publishers rely on early engagement forecasting to optimize editorial scheduling, multimedia formatting, and social media distribution.

This repository provides a modular, reproducible, configuration-driven regression pipeline that predicts the social media **share count (`shares`)** of online news articles (`35,680` records across `30` raw attributes). Key engineering and analytical highlights include:
1. **Strict Schema & Configuration Validation**: Fail-fast checks on YAML structure and input column schemas before processing begins.
2. **Leakage-Free Preprocessing & Domain Feature Engineering**: Deduplication by `URL`, `-1` sentinel conversion to `NaN`, temporal metadata exclusion (`timedelta`), and creation of `total_media`, `img_x_video`, `is_weekend`, `channel_x_weekend`, and `kw_avg_range` (`31` total features).
3. **Log-Target Regression Benchmarking**: Evaluates **4 regressors** against a naive `DummyBaseline` using **5-Fold Cross-Validation** (`GridSearchCV`) and `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`.
4. **Full Experiment Traceability & Reproducibility**: Persists exact train/test split IDs (`split_indices_<UTC>.json`), timestamped log files (`logs/`), config snapshots, diagnostic plots, and a cumulative `reports/experiment_history.csv` log.

---

## 2. Repository Structure
```text
news_popularity_prediction/
├── config.yaml                          # Centralized paths, logging, feature groups, & hyperparameter grids
├── main.py                              # CLI entry point supporting --mode train and --mode predict
├── requirements.txt                     # Pinned core runtime dependencies
├── requirements-dev.txt                 # Development, EDA, testing, & linting dependencies
├── README.md                            # Project documentation, schema, methodology, & results
├── data/
│   ├── mini_project_1_data.csv          # Raw dataset (35,680 news articles, 30 columns)
│   └── processed_news.csv               # Cleaned & feature-engineered dataset generated during training
├── logs/
│   └── pipeline_<UTC>.log               # Timestamped execution logs for runtime auditability
├── models/
│   ├── best_popularity_model.joblib     # Serialized winning regression pipeline (HistGradientBoosting)
│   ├── fitted_preprocessor.joblib       # Standalone fitted ColumnTransformer artifact
│   ├── split_indices_<UTC>.json         # Exact train/test article IDs for partition reproducibility
│   └── evaluation_metrics.json          # Cross-validation & holdout test metrics across all models
├── reports/
│   ├── latest_regression_diagnostics.png # Actual vs. Predicted log1p(shares) & residual distribution
│   ├── experiment_history.csv           # Cumulative cross-run experiment tracking table
│   └── config_snapshot_<UTC>.yaml       # Exact YAML configuration snapshot saved per training run
├── notebooks/
│   └── eda.ipynb                        # EDA: 20% sample validation, Kruskal-Wallis tests, VIF, & RF probe
├── src/
│   ├── __init__.py
│   ├── data_preparation.py              # Config/schema validation, cleaning, feature engineering, & encoding
│   └── model_training.py                # Config-driven GridSearchCV tuning, plots, & artifact persistence
└── tests/
    └── test_pipeline.py                 # Automated pytest suite (unit, negative-case, CLI, & E2E tests)
```

---

## 3. Input Dataset Schema & Feature Engineering
Input validation (`validate_schema()`) verifies all required columns prior to preprocessing:

| Category | Attributes | Handling & Role |
| :--- | :--- | :--- |
| **Target** | `shares` | Continuous social media share count (`median = 1,400.0`, `mean = 3,402.1`, `max = 843,300.0`). |
| **Identifiers & Temporal Metadata** | `ID`, `URL`, `timedelta` | `URL` used for deduplication (`35,680` unique articles); `ID`, `URL`, and scraping metadata `timedelta` are dropped to prevent memorization and temporal leakage. |
| **Content & Lexical** | `n_tokens_title`, `n_tokens_content`, `n_unique_tokens`, `n_non_stop_words`, `n_non_stop_unique_tokens`, `average_token_length` | Token rate features validated within bounds; scaled via `StandardScaler`. Raw article text is unavailable in the source dataset, constraining NLP analysis to lexical metadata. |
| **Media & Links** | `num_hrefs`, `num_self_hrefs`, `num_imgs`, `num_videos`, `n_comments` | Missing media/link counts (`27,241` total missing cells across the dataset, primarily in `num_videos`) are retained and median-imputed inside CV folds. |
| **Keyword & Reference Shares** | `self_reference_min/max/avg_shares`, `num_keywords`, `kw_min/max/avg_min`, `kw_min/max/avg_max`, `kw_min/max/avg_avg` | Sentinel `< 0` (`-1`) missing codes are converted to `NaN` and median-imputed inside CV folds. |
| **Categorical Context** | `weekday`, `data_channel` | Missing `data_channel` entries (`5,462` rows) are imputed as `"unknown"` and encoded via `OneHotEncoder(handle_unknown="ignore")`. |
| **Engineered Features (`5`)** | `total_media`, `img_x_video`, `is_weekend`, `channel_x_weekend`, `kw_avg_range` | Numeric media interaction (`num_imgs * num_videos`), combined media count, weekend flag, categorical interaction (`data_channel + "_" + is_weekend`), and keyword spread (`kw_max_avg - kw_min_avg`). |

---

## 4. Exploratory Data Analysis & Statistical Findings

Key analytical findings documented in `notebooks/eda.ipynb`:
1. **Sampling & Target Skewness**:
   * A reproducible `20%` working sample (`7,136` articles, `random_state=42`) closely matches the full dataset's target median (`1,400.0`) and mean (`3,402.1`).
   * Raw `shares` exhibits extreme right-skewness (`max = 843,300.0`), while `log1p(shares)` yields a near-symmetric distribution. Genuine viral outliers outside `1.5 * IQR` bounds are retained rather than clipped.
2. **Non-Parametric Hypothesis Testing (`Kruskal-Wallis` & `Spearman`)**:
   * **Significant Group Differences**: Kruskal-Wallis H-tests confirm statistically significant differences (p < 0.0001) across `data_channel`, `weekday`, and the interaction `channel_x_weekend`, with weekend publications and `lifestyle` / `social media` channels achieving higher median shares.
   * **Correlation & Null Findings**: Keyword average shares (`kw_avg_avg`) and self-reference shares show the strongest positive Spearman correlations (p < 0.0001), whereas `timedelta` quartiles and `n_non_stop_words` show negligible/null linear signal.
3. **Multicollinearity (`VIF`)**:
   * Variance Inflation Factor analysis identifies predictors with **VIF > 10** (such as overlapping `kw_*`, `self_reference_*`, and token uniqueness ratios), motivating L2-regularized regression (`RidgeRegression`) and tree-based ensembles (`HistGradientBoosting`, `RandomForest`, `GradientBoosting`).

---

## 5. Model Comparison & Results (Attempt 2)

All models were trained and evaluated using an **80/20 train-test split** (`28,544` train / `7,136` holdout test articles) and **5-Fold Cross-Validation** (`GridSearchCV`, `random_state=42`, `31` raw/engineered features):

| Model | 5-Fold CV RMSE | Holdout Test RMSE | Holdout Test MAE | Holdout Test RMSLE | Holdout Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Selected)** | 11,019.90 | 6,651.12 | 1,791.28 | **0.6980** | 0.2848 |
| **Random Forest** | 10,704.42 | **6,508.39** | 1,779.55 | 0.7021 | **0.3152** |
| **Gradient Boosting** | **10,695.63** | 6,563.23 | **1,771.39** | 0.7024 | 0.3036 |
| **Ridge Regression** | 8.22e+28 | 18,052.92 | 2,389.52 | 0.8210 | -4.2688 |
| **Dummy Baseline (`median`)** | 12,300.69 | 8,089.59 | 2,355.23 | 0.9586 | -0.0580 |

### Key Analytical Takeaways
1. **Strong Lift Over Naive Baseline**: All three tree-based ensembles substantially outperform `DummyBaseline` (`0.9586` RMSLE, `8,089.59` RMSE, `-0.0580` $R^2$), reducing **Test RMSLE by ~27%** (`0.9586` $\rightarrow$ `0.6980`) and **Test MAE by ~24%** (`2,355.23` $\rightarrow$ `1,791.28`).
2. **Non-Linearity & Exponential Back-Transformation**: While `RidgeRegression` achieves a reasonable log-scale error (`0.8210` RMSLE), unbounded linear extrapolation on extreme feature combinations gets magnified when inverted via `expm1()`, causing severe raw-scale RMSE spikes (`18,052.92`). Bounded tree ensembles prevent extreme extrapolation.
3. **Winning Model Selection (`HistGradientBoosting`)**: Selected as the production artifact (`models/best_popularity_model.joblib`) for achieving the lowest **Test RMSLE (`0.6980`)** while training **~30x faster** (`46s` vs. `28m` for `RandomForest` and `14m` for `GradientBoosting`) on `35,680` records.

---

## 6. Setup, Reproducibility & Usage

### Environment & Dependencies
* **Python**: `3.11+` (tested on `Python 3.11.17` in `prj1env`)
* Global random seeds (`random`, `numpy`) and estimator seeds are pinned via `training.random_state: 42` in `config.yaml`.

```bash
# Create and activate Conda environment
conda create -n prj1env python=3.11.17 -y
conda activate prj1env

# Install core runtime dependencies only:
pip install -r requirements.txt

# Or install full development, EDA, testing, and linting dependencies:
pip install -r requirements-dev.txt
```

### Code Quality Checks & Automated Tests
```bash
black --check src tests main.py
flake8 src tests main.py --max-line-length=120
python -m pytest -v
```

### Train & Evaluate Models
```bash
python main.py --mode train --config config.yaml
```

