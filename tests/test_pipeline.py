import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from main import load_config, main, run_predict, run_train
from src.data_preparation import (
    build_preprocessor,
    clean_news_data,
    load_raw_data,
    split_features_target,
)


@pytest.fixture
def project_config():
    """Load project YAML configuration."""
    return load_config("config.yaml")


@pytest.fixture
def sample_news_df():
    """Create synthetic news article records including sentinel -1 values, missing values, and duplicates."""
    rows = []
    channels = ["entertainment", "business", "technology", None, "lifestyle"]
    weekdays = ["monday", "tuesday", "wednesday", "saturday", "sunday"]
    for i in range(20):
        url = f"http://mashable.com{i if i < 19 else 0}"
        rows.append({
            "ID": i,
            "URL": url,
            "timedelta": 731.0,
            "weekday": weekdays[i % 5],
            "shares": 600 + (i * 250),
            "n_tokens_title": 10.0,
            "n_tokens_content": 250.0 + (i * 15),
            "n_unique_tokens": 0.60,
            "n_non_stop_words": 1.0,
            "n_non_stop_unique_tokens": 0.70,
            "num_hrefs": np.nan if i % 4 == 0 else 5.0,
            "num_self_hrefs": 2.0,
            "num_imgs": 3.0,
            "num_videos": np.nan if i % 2 == 0 else 1.0,
            "n_comments": float(i * 2),
            "average_token_length": 4.5,
            "data_channel": channels[i % 5],
            "self_reference_min_shares": 500.0,
            "self_reference_max_shares": 2000.0,
            "self_reference_avg_shares": 1200.0,
            "num_keywords": 6.0,
            "kw_min_min": -1.0 if i % 3 == 0 else 4.0,
            "kw_max_min": 500.0,
            "kw_avg_min": 150.0,
            "kw_min_max": 1000.0,
            "kw_max_max": 843300.0,
            "kw_avg_max": 250000.0,
            "kw_min_avg": -1.0 if i % 2 == 0 else 1200.0,
            "kw_max_avg": 3500.0,
            "kw_avg_avg": 2100.0,
        })
    return pd.DataFrame(rows)


def test_clean_news_data_sentinel_and_deduplication(sample_news_df, project_config):
    """Verify URL deduplication, -1 sentinel conversion to NaN, and engineered features."""
    cleaned, article_ids = clean_news_data(sample_news_df, project_config, is_training=True)

    assert len(cleaned) == 19, "Expected 19 unique rows after removing 1 duplicate URL"
    assert len(article_ids) == 19, "Expected article_ids length to match cleaned DataFrame"
    assert (cleaned["kw_min_min"].dropna() >= 0).all(), "Sentinel < 0 values must be converted to NaN"
    assert {"total_media", "img_x_video", "is_weekend", "channel_x_weekend", "kw_avg_range"}.issubset(cleaned.columns)


def test_preprocessor_transforms_cleanly(sample_news_df, project_config):
    """Verify numeric median imputation/scaling and categorical OneHotEncoding produce zero NaNs."""
    cleaned, _ = clean_news_data(sample_news_df, project_config, is_training=True)
    X, _ = split_features_target(cleaned, "shares")
    preprocessor = build_preprocessor(project_config)
    X_transformed = preprocessor.fit_transform(X)

    assert len(X_transformed) == 19, "Expected 19 transformed rows"
    assert not np.isnan(X_transformed).any(), "Transformed feature matrix must contain zero NaNs"


def test_negative_cases_missing_files_schema_and_model(tmp_path, sample_news_df, project_config):
    """Verify defensive exceptions on missing data, missing columns, and missing trained model."""
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "missing.csv")

    broken = sample_news_df.drop(columns=["n_tokens_content"])
    with pytest.raises(ValueError, match="missing required schema columns"):
        clean_news_data(broken, project_config, is_training=True)

    cfg = copy.deepcopy(project_config)
    cfg["paths"]["model_output"] = str(tmp_path / "nonexistent_model.joblib")
    unlabeled = tmp_path / "in.csv"
    sample_news_df.drop(columns=["shares"]).to_csv(unlabeled, index=False)
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        run_predict(cfg, unlabeled, tmp_path / "out.csv")


def test_end_to_end_train_predict_and_cli(tmp_path, sample_news_df, project_config):
    """Verify full regression train -> predict lifecycle, artifact contents, and CLI exit codes."""
    raw_csv = tmp_path / "news.csv"
    sample_news_df.to_csv(raw_csv, index=False)

    cfg = copy.deepcopy(project_config)
    cfg["paths"]["raw_data"] = str(raw_csv)
    cfg["paths"]["processed_data"] = str(tmp_path / "processed.csv")
    cfg["paths"]["model_output"] = str(tmp_path / "model.joblib")
    cfg["paths"]["preprocessor_output"] = str(tmp_path / "prep.joblib")
    cfg["paths"]["metrics_output"] = str(tmp_path / "metrics.json")
    cfg["paths"]["reports_dir"] = str(tmp_path / "reports")
    cfg["paths"]["logs_dir"] = str(tmp_path / "logs")
    cfg["training"]["cv_folds"] = 2
    for m_name in cfg["models"]:
        cfg["models"][m_name]["enabled"] = m_name in {"DummyBaseline", "RidgeRegression"}

    summary = run_train(cfg)
    assert summary["selected_best_model"] == "RidgeRegression"
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "reports" / "experiment_history.csv").exists()
    assert Path(summary["split_manifest"]).exists(), "Expected saved split indices JSON file"

    with open(tmp_path / "metrics.json", "r", encoding="utf-8") as mf:
        saved_metrics = json.load(mf)
    assert "RidgeRegression" in saved_metrics["model_comparisons"]

    unlabeled = tmp_path / "unlabeled.csv"
    sample_news_df.drop(columns=["shares"]).to_csv(unlabeled, index=False)
    out_csv = tmp_path / "preds.csv"

    preds_df = run_predict(cfg, unlabeled, out_csv)
    assert len(preds_df) == 19
    assert {"ID", "Predicted_Shares", "Popularity_Tier"}.issubset(preds_df.columns)

    # Test CLI entry point failure return code (1) on missing config
    exit_code = main(["--mode", "train", "--config", str(tmp_path / "missing_cfg.yaml")])
    assert exit_code == 1
