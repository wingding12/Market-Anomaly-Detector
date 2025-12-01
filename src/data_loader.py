"""
Data Loader for Market Anomaly Detector
========================================

This module handles loading and preprocessing of financial market data.
Supports the included FinancialMarketData.csv and user-uploaded CSVs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime
import warnings

from .feature_schema import (
    MODEL_FEATURES,
    N_FEATURES,
    CSV_CONFIG,
    LAG_SOURCE_FEATURES,
    validate_features,
)


# =============================================================================
# Constants
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_CSV = Path(__file__).parent.parent / "FinancialMarketData.csv"


# =============================================================================
# CSV Parsing Functions
# =============================================================================

def load_financial_market_data(
    filepath: Optional[Path] = None,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load the FinancialMarketData.csv file with proper parsing.
    
    Args:
        filepath: Path to CSV file. Defaults to included dataset.
        parse_dates: Whether to parse the date column.
        
    Returns:
        DataFrame with cleaned market data.
    """
    if filepath is None:
        filepath = DEFAULT_CSV
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Read with proper header configuration
    df = pd.read_csv(
        filepath,
        header=CSV_CONFIG["header_row"],
        skiprows=lambda x: x in [CSV_CONFIG["header_row"] + 1],  # Skip row after header
    )
    
    # Rename date column
    if CSV_CONFIG["date_column"] in df.columns:
        df = df.rename(columns={CSV_CONFIG["date_column"]: CSV_CONFIG["date_column_rename"]})
    
    # Drop unnecessary columns
    df = df.drop(columns=CSV_CONFIG["drop_columns"], errors="ignore")
    
    # Drop extra columns not used by model
    for col in CSV_CONFIG["extra_columns"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Clean the dataframe
    df = _clean_dataframe(df, parse_dates=parse_dates)
    
    return df


def _clean_dataframe(df: pd.DataFrame, parse_dates: bool = True) -> pd.DataFrame:
    """
    Clean and preprocess the dataframe.
    
    Args:
        df: Raw dataframe from CSV.
        parse_dates: Whether to parse date column.
        
    Returns:
        Cleaned dataframe.
    """
    # Remove rows where Date is NaN or not a valid date string
    if "Date" in df.columns:
        # Filter out non-date rows (metadata rows)
        df = df[df["Date"].notna()].copy()
        df = df[df["Date"].str.contains(r"\d{1,2}/\d{1,2}/\d{4}", na=False)].copy()
        
        if parse_dates:
            df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
            df = df.dropna(subset=["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
    
    # Convert numeric columns
    numeric_cols = [col for col in df.columns if col != "Date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def load_user_csv(
    filepath: Path,
    date_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a user-provided CSV file.
    
    Attempts to detect the format and parse appropriately.
    
    Args:
        filepath: Path to user's CSV file.
        date_column: Name of the date column (auto-detected if None).
        
    Returns:
        DataFrame with parsed data.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Try standard CSV load first
    df = pd.read_csv(filepath)
    
    # Try to detect date column
    if date_column is None:
        date_column = _detect_date_column(df)
    
    if date_column and date_column in df.columns:
        df = df.rename(columns={date_column: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    
    # Convert numeric columns
    numeric_cols = [col for col in df.columns if col != "Date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Attempt to detect which column contains dates."""
    date_keywords = ["date", "time", "timestamp", "datetime", "period"]
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            return col
    
    # Check first column if it looks like dates
    if len(df) > 0:
        first_col = df.columns[0]
        sample = str(df[first_col].iloc[0])
        if "/" in sample or "-" in sample:
            try:
                pd.to_datetime(sample)
                return first_col
            except:
                pass
    
    return None


# =============================================================================
# Data Validation
# =============================================================================

def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the dataframe has required features.
    
    Args:
        df: DataFrame to validate.
        
    Returns:
        Tuple of (is_valid, list of issues).
    """
    issues = []
    
    # Check for Date column
    if "Date" not in df.columns:
        issues.append("Missing 'Date' column")
    
    # Check for required features (excluding lag features which will be computed)
    base_features = [f for f in MODEL_FEATURES if "_lag_" not in f]
    missing = [f for f in base_features if f not in df.columns]
    
    if missing:
        issues.append(f"Missing {len(missing)} required features: {missing[:5]}...")
    
    # Check for NaN values
    numeric_cols = [col for col in df.columns if col != "Date"]
    nan_cols = [col for col in numeric_cols if df[col].isna().any()]
    if nan_cols:
        issues.append(f"{len(nan_cols)} columns contain NaN values")
    
    # Check minimum data points
    if len(df) < 10:
        issues.append(f"Insufficient data points: {len(df)} (minimum 10 required)")
    
    return len(issues) == 0, issues


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Get a summary of the loaded data.
    
    Args:
        df: DataFrame to summarize.
        
    Returns:
        Dictionary with summary statistics.
    """
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
    }
    
    if "Date" in df.columns:
        summary["date_range"] = {
            "start": df["Date"].min().strftime("%Y-%m-%d") if pd.notna(df["Date"].min()) else None,
            "end": df["Date"].max().strftime("%Y-%m-%d") if pd.notna(df["Date"].max()) else None,
        }
    
    # Count features by availability
    base_features = [f for f in MODEL_FEATURES if "_lag_" not in f]
    available = sum(1 for f in base_features if f in df.columns)
    summary["features"] = {
        "required": len(base_features),
        "available": available,
        "missing": len(base_features) - available,
    }
    
    # NaN statistics
    numeric_cols = [col for col in df.columns if col != "Date"]
    nan_counts = {col: df[col].isna().sum() for col in numeric_cols}
    total_nans = sum(nan_counts.values())
    summary["missing_values"] = {
        "total": total_nans,
        "columns_affected": sum(1 for v in nan_counts.values() if v > 0),
    }
    
    return summary


# =============================================================================
# Data Preprocessing
# =============================================================================

def fill_missing_values(
    df: pd.DataFrame,
    method: str = "ffill",
) -> pd.DataFrame:
    """
    Fill missing values in the dataframe.
    
    Args:
        df: DataFrame with potential missing values.
        method: Fill method ('ffill', 'bfill', 'interpolate', 'mean').
        
    Returns:
        DataFrame with filled values.
    """
    df = df.copy()
    numeric_cols = [col for col in df.columns if col != "Date"]
    
    if method == "ffill":
        df[numeric_cols] = df[numeric_cols].ffill()
        # Also backfill to handle NaNs at the start
        df[numeric_cols] = df[numeric_cols].bfill()
    elif method == "bfill":
        df[numeric_cols] = df[numeric_cols].bfill()
        df[numeric_cols] = df[numeric_cols].ffill()
    elif method == "interpolate":
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
    elif method == "mean":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    else:
        raise ValueError(f"Unknown fill method: {method}")
    
    return df


def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lag features required by the model.
    
    Creates lag features for VIX Index and MXWO Index.
    
    Args:
        df: DataFrame with base features.
        
    Returns:
        DataFrame with lag features added.
    """
    df = df.copy()
    
    for feature, n_lags in LAG_SOURCE_FEATURES.items():
        if feature not in df.columns:
            warnings.warn(f"Cannot compute lags for missing feature: {feature}")
            continue
        
        for lag in range(1, n_lags + 1):
            lag_name = f"{feature}_lag_{lag}"
            df[lag_name] = df[feature].shift(lag)
    
    return df


def prepare_features(
    df: pd.DataFrame,
    fill_method: str = "ffill",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare features for model prediction.
    
    This is the main function to call for preprocessing.
    
    Args:
        df: Raw dataframe from load functions.
        fill_method: Method to fill missing values.
        
    Returns:
        Tuple of (dates DataFrame, features DataFrame ready for model).
    """
    # Fill missing values
    df = fill_missing_values(df, method=fill_method)
    
    # Compute lag features
    df = compute_lag_features(df)
    
    # Drop rows with NaN in lag features (first few rows)
    max_lag = max(LAG_SOURCE_FEATURES.values())
    df = df.iloc[max_lag:].reset_index(drop=True)
    
    # Extract dates
    dates = df[["Date"]].copy() if "Date" in df.columns else pd.DataFrame()
    
    # Select and order features for model
    features = df[[f for f in MODEL_FEATURES if f in df.columns]].copy()
    
    # Validate feature count
    if len(features.columns) != N_FEATURES:
        missing = set(MODEL_FEATURES) - set(features.columns)
        raise ValueError(
            f"Feature count mismatch. Expected {N_FEATURES}, got {len(features.columns)}. "
            f"Missing: {missing}"
        )
    
    return dates, features


# =============================================================================
# Caching
# =============================================================================

_cache: Dict[str, pd.DataFrame] = {}


def load_data_cached(
    filepath: Optional[Path] = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Load data with caching for performance.
    
    Args:
        filepath: Path to data file.
        force_reload: Force reload even if cached.
        
    Returns:
        Loaded DataFrame.
    """
    cache_key = str(filepath or "default")
    
    if not force_reload and cache_key in _cache:
        return _cache[cache_key].copy()
    
    df = load_financial_market_data(filepath)
    _cache[cache_key] = df
    
    return df.copy()


def clear_cache():
    """Clear the data cache."""
    global _cache
    _cache = {}


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Data Loader Test")
    print("=" * 60)
    
    # Load default dataset
    print("\nLoading FinancialMarketData.csv...")
    df = load_financial_market_data()
    
    print(f"\n=== Data Summary ===")
    summary = get_data_summary(df)
    print(f"Rows: {summary['rows']}")
    print(f"Columns: {summary['columns']}")
    if 'date_range' in summary:
        print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Features: {summary['features']['available']}/{summary['features']['required']}")
    print(f"Missing Values: {summary['missing_values']['total']}")
    
    # Validate
    print(f"\n=== Validation ===")
    is_valid, issues = validate_data(df)
    print(f"Valid: {is_valid}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    # Prepare features
    print(f"\n=== Feature Preparation ===")
    dates, features = prepare_features(df)
    print(f"Dates shape: {dates.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Feature columns: {list(features.columns[:5])}... ({len(features.columns)} total)")

