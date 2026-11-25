import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import random
import sys
import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

from src.data_preparation import (
    clean_news_data,
    load_raw_data,
    split_features_target,
    validate_config,
)
from src.model_training import train_and_evaluate


def configure_logging(config: dict | None = None) -> None:
    """Configure console and optional timestamped file logging from YAML settings."""
    log_cfg = (config or {}).get("logging", {})
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_cfg.get("log_to_file", False) and config is not None:
        logs_dir = Path(config.get("paths", {}).get("logs_dir", "logs"))
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        handlers.append(logging.FileHandler(logs_dir / f"pipeline_{ts}.log", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def set_global_seeds(seed: int) -> None:
    """Set global Python and NumPy seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load and validate YAML configuration from disk."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML in '{path}': {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file '{path}' must define a YAML mapping.")
    validate_config(config)
    return config


def run_train(config: dict) -> dict:
    """Execute full regression training, cross-validation, and report generation."""
    logger = logging.getLogger(__name__)
    seed = int(config["training"]["random_state"])
    set_global_seeds(seed)
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    logger.info(
        "Environment | Python=%s | pandas=%s | numpy=%s | scikit-learn=%s | seed=%d",
        py_ver,
        pd.__version__,
        np.__version__,
        sklearn.__version__,
        seed,
    )
    logger.info(
        "Active Config | primary_metric=%s | cv_folds=%d | log_target=%s",
        config["training"]["primary_metric"],
        config["training"]["cv_folds"],
        config["training"].get("log_transform_target", True),
    )

    raw_df = load_raw_data(config["paths"]["raw_data"])
    cleaned_df, article_ids = clean_news_data(raw_df, config, is_training=True)

    processed_path = Path(config["paths"]["processed_data"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)

    X, y = split_features_target(cleaned_df, config["data"]["target_col"])
    logger.info("Starting model training on %d articles (%d features)", len(X), len(X.columns))
    summary = train_and_evaluate(X, y, config, article_ids=article_ids)

    logger.info(
        "Selected Best Model: %s (Test RMSLE: %.4f)",
        summary["selected_best_model"],
        summary["best_test_rmsle"],
    )
    for model_name, metrics in summary["model_comparisons"].items():
        logger.info(
            "Model %-22s | CV RMSE: %8.2f | Test RMSE: %8.2f | MAE: %7.2f | RMSLE: %.4f | R2: %.4f",
            model_name,
            metrics["cv_rmse"],
            metrics["test_rmse"],
            metrics["test_mae"],
            metrics["test_rmsle"],
            metrics["test_r2"],
        )
    return summary


def run_predict(config: dict, input_file: str | Path, output_file: str | Path) -> pd.DataFrame:
    """Generate predicted share counts and vectorized popularity tiers on new articles."""
    logger = logging.getLogger(__name__)
    model_path = Path(config["paths"]["model_output"])
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run --mode train first.")

    model = joblib.load(model_path)
    raw_df = load_raw_data(input_file)
    id_col = config["data"]["id_col"]

    cleaned_df, article_ids = clean_news_data(raw_df, config, is_training=False)
    if config["data"]["target_col"] in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=[config["data"]["target_col"]])

    preds = np.clip(model.predict(cleaned_df), 0, None)
    tiers = np.select(
        [preds >= 3000, preds >= 1400],
        ["Viral Potential (>3000 shares)", "Above Average (1400-3000)"],
        default="Standard (<1400)",
    )

    out_df = pd.DataFrame({
        "Predicted_Shares": np.round(preds).astype(int),
        "Popularity_Tier": tiers,
    })
    if article_ids is not None:
        out_df.insert(0, id_col, article_ids.values)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Saved %d article predictions to %s", len(out_df), out_path.resolve())
    return out_df


def main(argv: list[str] | None = None) -> int:
    """CLI entry point returning 0 on success and 1 on error."""
    configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Online News Popularity Prediction Pipeline")
    parser.add_argument("--mode", choices=["train", "predict"], required=True, help="Execution mode")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    parser.add_argument("--input", default="data/new_articles.csv", help="Input file for prediction mode")
    parser.add_argument("--output", default="data/predictions.csv", help="Output file for prediction mode")

    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        configure_logging(cfg)
        if args.mode == "train":
            run_train(cfg)
        elif args.mode == "predict":
            run_predict(cfg, args.input, args.output)
        return 0
    except Exception:
        logger.exception("Pipeline execution failed due to an unhandled error.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
