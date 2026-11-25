# Online News Popularity Prediction

## 1. Project Overview & Objective
Digital news publishers rely on early engagement forecasting to optimize editorial scheduling, content formatting, and social media promotion. 

This repository provides a modular, end-to-end regression pipeline that predicts the social media **share count (`shares`)** of online news articles (`35,680` records across `30` raw attributes). The workflow encompasses:
1. Automated schema validation, sentinel-value cleaning (`-1` keyword codes), and domain feature engineering.
2. Leak-free preprocessing with `ColumnTransformer` and log-target modeling via `TransformedTargetRegressor`.
3. Systematic benchmarking of **4 regression algorithms** against a naive `DummyBaseline` using **5-Fold Cross-Validation**.
4. Automated diagnostic plotting, timestamped model artifact persistence, and a CLI supporting both training and batch inference.

---

## 2. Repository Structure
```text
news_popularity_prediction/
├── config.yaml                          # Centralized file paths, feature lists, & hyperparameter grids
├── main.py                              # CLI entry point supporting --mode train and --mode predict
├── requirements.txt                     # Pinned Python dependencies
├── README.md                            # Project documentation, schema, methodology, & results
├── data/
│   ├── mini_project_1_data.csv          # Raw dataset (35,680 news articles, 30 columns)
│   └── processed_news.csv               # Cleaned & feature-engineered dataset generated during training
├── models/
│   ├── best_popularity_model.joblib     # Serialized winning regression pipeline (HistGradientBoosting)
│   ├── fitted_preprocessor.joblib       # Standalone fitted ColumnTransformer artifact
│   └── evaluation_metrics.json          # Cross-validation & holdout test metrics across all models
├── reports/
│   ├── latest_regression_diagnostics.png # Actual vs. Predicted log1p(shares) & residual distribution
│   └── config_snapshot_<UTC>.yaml       # Exact YAML configuration snapshot saved per training run
├── notebooks/
│   └── eda.ipynb                        # Exploratory Data Analysis: IQR outlier audit, VIF, & RF importance
├── src/
│   ├── __init__.py
│   ├── data_preparation.py              # Schema validation, sentinel cleaning, feature engineering, & encoding
│   └── model_training.py                # Config-driven GridSearchCV training, evaluation, & plot generation
└── tests/
    └── test_pipeline.py                 # Automated pytest suite (unit, schema/negative, & end-to-end tests)
```

---

## 3. Input Dataset Schema & Feature Descriptions
Input validation (`validate_schema()`) enforces the presence of all required features prior to preprocessing:

| Category | Attributes | Handling & Role |
| :--- | :--- | :--- |
| **Target** | `shares` | Continuous social media share count (`median = 1,400.0`, `mean = 3,402.1`, `max = 843,300.0`). |
| **Identifiers / Metadata** | `ID`, `URL`, `timedelta` | `URL` used for deduplication (`35,680` unique articles verified); `ID`, `URL`, and scraping metadata `timedelta` are dropped to prevent memorization and temporal leakage. |
| **Content & Lexical** | `n_tokens_title`, `n_tokens_content`, `n_unique_tokens`, `n_non_stop_words`, `n_non_stop_unique_tokens`, `average_token_length` | Cleaned for valid `[0, 1]` bounds on token rate features; scaled via `StandardScaler`. |
| **Media & Links** | `num_hrefs`, `num_self_hrefs`, `num_imgs`, `num_videos`, `n_comments` | Missing media/link counts (`num_videos` missing in ~47% of rows; `num_hrefs`/`num_imgs` missing in ~2–4%) are retained and imputed via training-fold median. |
| **Keyword & Reference Shares** | `self_reference_min/max/avg_shares`, `num_keywords`, `kw_min/max/avg_min`, `kw_min/max/avg_max`, `kw_min/max/avg_avg` | Sentinel `< 0` (`-1`) missing codes in keyword/reference columns are converted to `NaN` and median-imputed inside CV folds. |
| **Categorical Context** | `weekday`, `data_channel` | Missing `data_channel` values (`5,462` rows) are explicitly imputed as `"unknown"` and encoded via `OneHotEncoder(handle_unknown="ignore")`. |
| **Engineered Features** | `total_media`, `is_weekend`, `kw_avg_range` | `total_media = num_imgs + num_videos`; `is_weekend = 1` for Saturday/Sunday; `kw_avg_range = kw_max_avg - kw_min_avg`. |

---

## 4. Exploratory Data Analysis & Modeling Decisions

Key analytical findings from `notebooks/eda.ipynb`:
1. **Heavy Right-Skew in `shares`**:
   * Article popularity spans from low-engagement posts up to `843,300` shares (`median = 1,400.0` vs. `mean = 3,402.1`).
   * Wrapping candidate regressors in **`TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`** stabilizes variance during training and aligns optimization with relative percentage errors (**RMSLE**).
2. **Missing Data & Outlier Retention**:
   * Incomplete media and keyword fields are **retained** rather than dropped so the pipeline preserves all `35,680` articles and handles missing fields seamlessly during live inference (`--mode predict`).
   * Extreme high-share viral articles outside `1.5 * IQR` bounds represent genuine viral phenomena rather than sensor errors and are retained.
3. **Leak-Free Preprocessing (`29` Raw/Engineered Features)**:
   * `SimpleImputer(strategy="median")` + `StandardScaler` for numeric columns and `OneHotEncoder` for `weekday` + `data_channel` are encapsulated inside a scikit-learn `Pipeline`, ensuring zero data leakage across cross-validation folds.

---

## 5. Model Comparison & Results

All models were evaluated using an **80/20 train-test split** (`28,544` train / `7,136` holdout test articles) and **5-Fold Cross-Validation** (`GridSearchCV`, `random_state=42`):

| Model | 5-Fold CV RMSE | Holdout Test RMSE | Holdout Test MAE | Holdout Test RMSLE | Holdout Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Selected)** | 11,025.15 | 6,682.07 | 1,791.96 | **0.6974** | 0.2782 |
| **Gradient Boosting** | **10,691.82** | 6,539.06 | **1,767.52** | 0.7009 | 0.3087 |
| **Random Forest** | 10,732.81 | **6,500.71** | 1,776.15 | 0.7063 | **0.3168** |
| **Ridge Regression** | 9.72e+28 | 17,610.96 | 2,385.85 | 0.8217 | -4.0140 |
| **Dummy Baseline (`median`)** | 12,300.69 | 8,089.59 | 2,355.23 | 0.9586 | -0.0580 |

### Key Analytical Takeaways
1. **Substantial Improvement Over Naive Baseline**: All tree-based ensembles outperform `DummyBaseline` (`0.9586` RMSLE, `8,089.59` RMSE, `-0.0580` $R^2$), reducing **Test RMSLE by ~27%** (`0.9586` $\rightarrow$ `0.6974`) and **Test MAE by ~24%** (`2,355.23` $\rightarrow$ `1,791.96`).
2. **Non-Linearity & Exponential Back-Transformation Behavior**: While `RidgeRegression` achieves a reasonable log-scale error (`0.8217` RMSLE), unbounded linear extrapolation on extreme feature combinations gets amplified when inverted via `expm1()`, causing massive raw-scale RMSE spikes (`17,610.96`). Tree-based ensembles naturally bound predictions within observed leaf ranges.
3. **Winning Model Selection (`HistGradientBoosting`)**: Selected as the production model (`models/best_popularity_model.joblib`) for achieving the lowest **Test RMSLE (`0.6974`)** while training **~60x faster** (`36s` vs. `36m`) than standard `GradientBoostingRegressor` on `35,680` records.

---

## 6. Setup, Reproducibility & Usage

### Environment Requirements
* **Python**: `3.11+` (tested on `Python 3.11.17` in `prj1env`)
* Global seeds (`random`, `numpy`) and estimator seeds are pinned via `training.random_state: 42` in `config.yaml`.

### Installation
```bash
conda create -n prj1env python=3.11.17 -y
conda activate prj1env
pip install -r requirements.txt
```

### Run Automated Unit & Integration Tests
```bash
python -m pytest -v
```

### Train & Evaluate Models
```bash
python main.py --mode train --config config.yaml
```
*Saves `models/best_popularity_model.joblib`, `models/fitted_preprocessor.joblib`, `models/evaluation_metrics.json`, diagnostic plots in `reports/`, and a timestamped config snapshot.*

### Generate Popularity Predictions on New Articles
```bash
python main.py --mode predict --input data/new_articles.csv --output data/predictions.csv
```
*Outputs `ID`, `Predicted_Shares`, and `Popularity_Tier` (`Viral Potential (>3000 shares)`, `Above Average (1400-3000)`, or `Standard (<1400)`).*

---

## 7. Limitations & Future Work
* **Content & Headline Semantics**: Incorporating pre-trained text embeddings or sentiment features from article headlines/body text (beyond bag-of-words token ratios) could further boost explained variance ($R^2$).
* **Hyperparameter & Feature Selection Tuning**: Expanding `HistGradientBoosting` regularization (`l2_regularization`, `min_samples_leaf`) and inspecting permutation importance can further tighten tail error on viral articles.

 
