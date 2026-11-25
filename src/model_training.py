import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any
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


def build_regressor_registry(random_state: int) -> dict[str, Any]:
    """Return mapping of supported model names to instantiated scikit-learn regressors."""
    return {
        "DummyBaseline": DummyRegressor(),
        "RidgeRegression": Ridge(random_state=random_state),
        "RandomForest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=random_state),
    }


def _tune_and_score_model(
    name: str,
    base_reg: Any,
    model_cfg: dict,
    base_preprocessor: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: KFold,
    primary_metric: str,
    use_log_target: bool,
) -> tuple[Any, np.ndarray, dict]:
    """Tune a single regressor via GridSearchCV and compute holdout metrics."""
    if use_log_target:
        estimator_step = TransformedTargetRegressor(
            regressor=base_reg,
            func=np.log1p,
            inverse_func=np.expm1,
        )
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

    best_est = grid.best_estimator_
    y_pred = np.clip(best_est.predict(X_test), 0, None)

    metrics = {
        "best_params": grid.best_params_,
        "cv_rmse": round(float(-grid.best_score_), 2),
        "test_rmse": round(float(root_mean_squared_error(y_test, y_pred)), 2),
        "test_mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "test_rmsle": round(float(np.sqrt(mean_squared_log_error(y_test, y_pred))), 4),
        "test_r2": round(float(r2_score(y_test, y_pred)), 4),
    }
    return best_est, y_pred, metrics


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
    ax1.scatter(y_true_log, y_pred_log, alpha=0.25, s=10, color="steelblue")
    lims = [min(y_true_log.min(), y_pred_log.min()), max(y_true_log.max(), y_pred_log.max())]
    ax1.plot(lims, lims, "r--", linewidth=1.5, label="Ideal Fit (y = x)")
    ax1.set_title(f"Actual vs. Predicted log1p(shares) — {best_model_name}")
    ax1.set_xlabel("Actual log1p(shares)")
    ax1.set_ylabel("Predicted log1p(shares)")
    ax1.legend()

    ax2.hist(residuals_log, bins=40, color="teal", edgecolor="white", alpha=0.85)
    ax2.axvline(0.0, color="red", linestyle="--", linewidth=1.5)
    ax2.set_title(f"Log-Scale Residual Distribution — {best_model_name}")
    ax2.set_xlabel("Residual: Actual log1p(shares) - Predicted log1p(shares)")
    ax2.set_ylabel("Article Count")

    plt.tight_layout()
    plt.savefig(reports_dir / f"regression_diagnostics_{timestamp}.png", dpi=150)
    plt.savefig(reports_dir / "latest_regression_diagnostics.png", dpi=150)
    plt.close(fig)


def _persist_run_artifacts(
    config: dict,
    best_pipeline: Any,
    best_model_name: str,
    best_rmsle: float,
    results: dict,
    train_ids: list,
    test_ids: list,
    timestamp: str,
) -> dict:
    """Persist serialized pipeline, split indices, config snapshot, metrics JSON, and cumulative CSV log."""
    reports_dir = Path(config["paths"].get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(config["paths"]["model_output"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)
    joblib.dump(best_pipeline, model_path.with_name(f"{model_path.stem}_{timestamp}.joblib"))

    if "preprocessor_output" in config["paths"]:
        prep_path = Path(config["paths"]["preprocessor_output"])
        prep_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline.named_steps["preprocessor"], prep_path)

    split_path = model_path.parent / f"split_indices_{timestamp}.json"
    with open(split_path, "w", encoding="utf-8") as sf:
        json.dump({"train_ids": train_ids, "test_ids": test_ids}, sf)

    with open(reports_dir / f"config_snapshot_{timestamp}.yaml", "w", encoding="utf-8") as cfg_out:
        yaml.safe_dump(config, cfg_out, sort_keys=False)

    history_csv = reports_dir / "experiment_history.csv"
    file_exists = history_csv.exists()
    with open(history_csv, "a", newline="", encoding="utf-8") as hf:
        writer = csv.writer(hf)
        if not file_exists:
            writer.writerow(["timestamp_utc", "model", "is_selected", "cv_rmse", "test_rmse", "test_mae", "test_rmsle", "test_r2"])
        for m_name, m_vals in results.items():
            writer.writerow([
                timestamp,
                m_name,
                m_name == best_model_name,
                m_vals["cv_rmse"],
                m_vals["test_rmse"],
                m_vals["test_mae"],
                m_vals["test_rmsle"],
                m_vals["test_r2"],
            ])

    summary = {
        "run_timestamp_utc": timestamp,
        "selected_best_model": best_model_name,
        "best_test_rmsle": round(float(best_rmsle), 4),
        "split_sizes": {"train_rows": len(train_ids), "test_rows": len(test_ids)},
        "split_manifest": str(split_path),
        "model_comparisons": results,
    }

    metrics_path = Path(config["paths"]["metrics_output"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    article_ids: pd.Series | None = None,
) -> dict:
    """Split data, tune config-defined regressors via K-Fold CV, and persist all artifacts."""
    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]
    cv_folds = config["training"]["cv_folds"]
    primary_metric = config["training"]["primary_metric"]
    use_log_target = bool(config["training"].get("log_transform_target", True))

    ids_series = article_ids if article_ids is not None else pd.Series(X.index, index=X.index)
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids_series, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Leakage Prevention: Preprocessing fit strictly inside %d-fold CV on %d training rows (holdout=%d rows).",
        cv_folds,
        len(X_train),
        len(X_test),
    )

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    base_preprocessor = build_preprocessor(config)
    registry = build_regressor_registry(random_state)

    results = {}
    best_model_name = None
    best_pipeline = None
    best_y_pred = None
    best_rmsle = np.inf

    for name, model_cfg in config["models"].items():
        if not model_cfg.get("enabled", True):
            continue
        if name not in registry:
            raise ValueError(f"Unsupported model '{name}' in config['models'].")

        logger.info("Tuning regressor: %s", name)
        try:
            est, y_pred, metrics = _tune_and_score_model(
                name, registry[name], model_cfg, base_preprocessor,
                X_train, y_train, X_test, y_test, cv, primary_metric, use_log_target
            )
        except Exception as exc:
            logger.error("Regressor training failed for '%s': %s", name, exc)
            raise RuntimeError(f"Training crashed on model '{name}': {exc}") from exc

        results[name] = metrics
        if name != "DummyBaseline" and metrics["test_rmsle"] < best_rmsle:
            best_rmsle = metrics["test_rmsle"]
            best_model_name = name
            best_pipeline = est
            best_y_pred = y_pred

    if best_pipeline is None or best_model_name is None:
        raise RuntimeError("No non-baseline regressor was enabled or successfully trained.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reports_dir = Path(config["paths"].get("reports_dir", "reports"))
    save_regression_plots(y_test, best_y_pred, best_model_name, reports_dir, timestamp)

    return _persist_run_artifacts(
        config=config,
        best_pipeline=best_pipeline,
        best_model_name=best_model_name,
        best_rmsle=best_rmsle,
        results=results,
        train_ids=ids_train.tolist(),
        test_ids=ids_test.tolist(),
        timestamp=timestamp,
    )
