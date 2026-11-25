from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_log_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
import yaml

from src.data_preparation import build_preprocessor

logger = logging.getLogger(__name__)


def build_regressor_registry(random_state: int) -> Dict[str, Any]:
    """Return mapping of supported model names to instantiated scikit-learn regressors."""
    return {
        "DummyBaseline": DummyRegressor(),
        "RidgeRegression": Ridge(random_state=random_state),
        "RandomForest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=random_state),
    }


def save_regression_plots(
    y_test: pd.Series,
    best_y_pred: np.ndarray,
    best_model_name: str,
    reports_dir: Path,
    timestamp: str,
) -> None:
    """Save Actual vs. Predicted (Log Scale) and Residual Distribution diagnostic plots."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    y_true_log = np.log1p(y_test.values)
    y_pred_log = np.log1p(np.clip(best_y_pred, 0, None))
    residuals_log = y_true_log - y_pred_log

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Actual vs Predicted on log1p(shares) scale
    ax1.scatter(y_true_log, y_pred_log, alpha=0.25, s=10, color="steelblue")
    lims = [min(y_true_log.min(), y_pred_log.min()), max(y_true_log.max(), y_pred_log.max())]
    ax1.plot(lims, lims, "r--", linewidth=1.5, label="Ideal Fit (y = x)")
    ax1.set_title(f"Actual vs. Predicted log1p(shares) — {best_model_name}")
    ax1.set_xlabel("Actual log1p(shares)")
    ax1.set_ylabel("Predicted log1p(shares)")
    ax1.legend()

    # 2. Residual Histogram
    ax2.hist(residuals_log, bins=40, color="teal", edgecolor="white", alpha=0.85)
    ax2.axvline(0.0, color="red", linestyle="--", linewidth=1.5)
    ax2.set_title(f"Log-Scale Residual Distribution — {best_model_name}")
    ax2.set_xlabel("Residual: Actual log1p(shares) - Predicted log1p(shares)")
    ax2.set_ylabel("Article Count")

    plt.tight_layout()
    plt.savefig(reports_dir / f"regression_diagnostics_{timestamp}.png", dpi=150)
    plt.savefig(reports_dir / "latest_regression_diagnostics.png", dpi=150)
    plt.close(fig)


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, config: dict) -> dict:
    """Split data, tune config-defined regressors via K-Fold CV, and save best artifacts."""
    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]
    cv_folds = config["training"]["cv_folds"]
    primary_metric = config["training"]["primary_metric"]
    use_log_target = bool(config["training"].get("log_transform_target", True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    base_preprocessor = build_preprocessor(config)
    registry = build_regressor_registry(random_state)

    results = {}
    best_model_name = None
    best_pipeline = None
    best_y_pred = None
    best_rmse = np.inf

    for name, model_cfg in config["models"].items():
        if not model_cfg.get("enabled", True):
            continue
        if name not in registry:
            raise ValueError(f"Unsupported model '{name}' in config['models'].")

        logger.info("Tuning regressor: %s", name)
        try:
            base_reg = registry[name]
            if use_log_target:
                estimator_step = TransformedTargetRegressor(
                    regressor=base_reg,
                    func=np.log1p,
                    inverse_func=np.expm1,
                )
                # Map config param keys from regressor__param -> regressor__regressor__param
                raw_params = model_cfg.get("params", {})
                param_grid = {
                    k.replace("regressor__", "regressor__regressor__", 1): v
                    for k, v in raw_params.items()
                }
            else:
                estimator_step = base_reg
                param_grid = model_cfg.get("params", {})

            pipeline = Pipeline(steps=[
                ("preprocessor", clone(base_preprocessor)),
                ("regressor", estimator_step),
            ])

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=primary_metric,
                cv=cv,
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)
        except Exception as exc:
            logger.error("Regressor training failed for '%s': %s", name, exc)
            raise RuntimeError(f"Training crashed on model '{name}': {exc}") from exc

        best_est = grid.best_estimator_
        y_pred = np.clip(best_est.predict(X_test), 0, None)

        test_rmse = float(root_mean_squared_error(y_test, y_pred))
        test_mae = float(mean_absolute_error(y_test, y_pred))
        test_rmsle = float(np.sqrt(mean_squared_log_error(y_test, y_pred)))
        test_r2 = float(r2_score(y_test, y_pred))

        results[name] = {
            "best_params": grid.best_params_,
            "cv_rmse": round(float(-grid.best_score_), 2),
            "test_rmse": round(test_rmse, 2),
            "test_mae": round(test_mae, 2),
            "test_rmsle": round(test_rmsle, 4),
            "test_r2": round(test_r2, 4),
        }

        # Select winning model by lowest RMSLE (best accuracy across skewed popularity scale)
        if name != "DummyBaseline" and test_rmsle < best_rmse:
            best_rmse = test_rmsle
            best_model_name = name
            best_pipeline = best_est
            best_y_pred = y_pred

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reports_dir = Path(config["paths"].get("reports_dir", "reports"))
    save_regression_plots(y_test, best_y_pred, best_model_name, reports_dir, timestamp)

    model_path = Path(config["paths"]["model_output"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)
    joblib.dump(best_pipeline, model_path.with_name(f"{model_path.stem}_{timestamp}.joblib"))

    if "preprocessor_output" in config["paths"]:
        prep_path = Path(config["paths"]["preprocessor_output"])
        prep_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline.named_steps["preprocessor"], prep_path)

    with open(reports_dir / f"config_snapshot_{timestamp}.yaml", "w", encoding="utf-8") as cfg_out:
        yaml.safe_dump(config, cfg_out, sort_keys=False)

    summary = {
        "run_timestamp_utc": timestamp,
        "selected_best_model": best_model_name,
        "best_test_rmsle": round(float(best_rmse), 4),
        "split_sizes": {"train_rows": len(X_train), "test_rows": len(X_test)},
        "model_comparisons": results,
    }

    metrics_path = Path(config["paths"]["metrics_output"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
