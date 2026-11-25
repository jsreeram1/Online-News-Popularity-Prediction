import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def validate_config(config: dict) -> None:
    """Validate required sections and keys in the YAML configuration."""
    for section in ["paths", "data", "training", "models"]:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")
    for key in ["target_col", "id_col", "numeric_cols", "categorical_cols"]:
        if key not in config["data"]:
            raise ValueError(f"Missing required key '{key}' under config['data']")


def validate_schema(df: pd.DataFrame, config: dict, is_training: bool = True) -> None:
    """Verify expected feature and target columns exist before preprocessing."""
    expected = list(config["data"]["numeric_cols"]) + list(config["data"]["categorical_cols"])
    if is_training:
        expected.append(config["data"]["target_col"])
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required schema columns: {missing}")


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load CSV or Excel dataset safely with stripped column headers and diagnostics."""
    path = Path(filepath)
    if not path.exists():
        alt_path = path.with_suffix(".xlsx" if path.suffix.lower() == ".csv" else ".csv")
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    df.columns = [col.strip() for col in df.columns]
    total_missing = int(df.isna().sum().sum())
    logger.info(
        "Loaded '%s' | shape=%s | total_missing_cells=%d",
        path.name,
        df.shape,
        total_missing,
    )
    return df


def _engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction and domain features with explicit column-presence guards."""
    if {"num_imgs", "num_videos"}.issubset(df.columns):
        imgs = df["num_imgs"].fillna(0)
        vids = df["num_videos"].fillna(0)
        df["total_media"] = imgs + vids
        df["img_x_video"] = imgs * vids
    else:
        logger.warning("Skipping media interaction features: 'num_imgs' or 'num_videos' missing.")

    if "weekday" in df.columns:
        df["is_weekend"] = df["weekday"].isin(["saturday", "sunday"]).astype(int)
        if "data_channel" in df.columns:
            df["channel_x_weekend"] = df["data_channel"] + "_" + df["is_weekend"].astype(str)

    if {"kw_max_avg", "kw_min_avg"}.issubset(df.columns):
        df["kw_avg_range"] = df["kw_max_avg"] - df["kw_min_avg"]
    else:
        logger.warning("Skipping 'kw_avg_range': 'kw_max_avg' or 'kw_min_avg' missing.")

    return df


def clean_news_data(
    df: pd.DataFrame,
    config: dict,
    is_training: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Clean anomalies, handle sentinel values, deduplicate, and engineer domain features."""
    validate_schema(df, config, is_training=is_training)
    df = df.copy()
    initial_rows = len(df)
    id_col = config["data"]["id_col"]
    url_col = config["data"].get("url_col", "URL")
    target_col = config["data"]["target_col"]

    # 1. Deduplicate articles by URL or ID
    dedup_subset = [url_col] if url_col in df.columns else ([id_col] if id_col in df.columns else None)
    if dedup_subset:
        df = df.drop_duplicates(subset=dedup_subset, keep="first").reset_index(drop=True)
        logger.info("Deduplicated articles (%s): %d -> %d rows", dedup_subset, initial_rows, len(df))

    article_ids = df[id_col].copy() if id_col in df.columns else None

    # 2. Standardize categorical strings & impute missing values as 'unknown'
    for col in config["data"]["categorical_cols"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": "unknown", "none": "unknown", "": "unknown"})
            )

    # 3. Coerce numeric columns and replace sentinel/out-of-range values with NaN
    rate_cols = {"n_unique_tokens", "n_non_stop_words", "n_non_stop_unique_tokens"}
    for col in config["data"]["numeric_cols"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col in rate_cols:
                df.loc[(df[col] < 0) | (df[col] > 1), col] = np.nan
            else:
                df.loc[df[col] < 0, col] = np.nan

    # 4. Feature Engineering
    df = _engineer_domain_features(df)

    # 5. Validate target column during training
    if is_training and target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        valid_mask = df[target_col].notna() & (df[target_col] >= 0)
        df = df.loc[valid_mask].reset_index(drop=True)
        if article_ids is not None:
            article_ids = article_ids.loc[valid_mask].reset_index(drop=True)
        logger.info(
            "Target '%s' summary | count=%d | median=%.1f | mean=%.1f | max=%.1f",
            target_col,
            len(df),
            df[target_col].median(),
            df[target_col].mean(),
            df[target_col].max(),
        )

    # 6. Drop identifier and temporal leakage columns with explicit audit log
    cols_to_drop = [c for c in config["data"]["drop_cols"] if c in df.columns]
    if cols_to_drop:
        logger.info("Leakage Prevention: Dropping identifier/metadata columns %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    return df, article_ids


def build_preprocessor(config: dict) -> ColumnTransformer:
    """Build leak-free preprocessing pipeline for numeric, categorical, and interaction features."""
    engineered_numeric = ["total_media", "img_x_video", "is_weekend", "kw_avg_range"]
    engineered_categorical = ["channel_x_weekend"]

    numeric_features: list[str] = list(config["data"]["numeric_cols"]) + engineered_numeric
    categorical_features: list[str] = list(config["data"]["categorical_cols"]) + engineered_categorical

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and continuous target vector y."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    return df.drop(columns=[target_col]), df[target_col]
