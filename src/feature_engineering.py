"""
Feature Engineering for Market Anomaly Detector
================================================

This module provides feature transformation and engineering utilities
for market data preprocessing before model prediction.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


# =============================================================================
# Feature Transformation Classes
# =============================================================================

@dataclass
class FeatureStats:
    """Statistics for feature normalization."""
    mean: Dict[str, float]
    std: Dict[str, float]
    min: Dict[str, float]
    max: Dict[str, float]


class FeatureTransformer:
    """
    Feature transformation pipeline for market data.
    
    Handles normalization, scaling, and derived feature computation.
    """
    
    def __init__(self):
        self.stats: Optional[FeatureStats] = None
        self._fitted = False
    
    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        """
        Compute statistics from training data.
        
        Args:
            df: Training data (features only, no date column).
            
        Returns:
            Self for method chaining.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        self.stats = FeatureStats(
            mean={col: df[col].mean() for col in numeric_cols},
            std={col: df[col].std() for col in numeric_cols},
            min={col: df[col].min() for col in numeric_cols},
            max={col: df[col].max() for col in numeric_cols},
        )
        self._fitted = True
        return self
    
    def transform_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply z-score normalization.
        
        Args:
            df: Features to transform.
            
        Returns:
            Normalized features.
        """
        if not self._fitted:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        df = df.copy()
        for col in df.columns:
            if col in self.stats.mean:
                std = self.stats.std[col]
                if std > 0:
                    df[col] = (df[col] - self.stats.mean[col]) / std
                else:
                    df[col] = 0
        return df
    
    def transform_minmax(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply min-max scaling to [0, 1].
        
        Args:
            df: Features to transform.
            
        Returns:
            Scaled features.
        """
        if not self._fitted:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        df = df.copy()
        for col in df.columns:
            if col in self.stats.min:
                range_val = self.stats.max[col] - self.stats.min[col]
                if range_val > 0:
                    df[col] = (df[col] - self.stats.min[col]) / range_val
                else:
                    df[col] = 0.5
        return df


# =============================================================================
# Technical Indicators
# =============================================================================

def compute_returns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    periods: int = 1,
) -> pd.DataFrame:
    """
    Compute percentage returns for specified columns.
    
    Args:
        df: DataFrame with price data.
        columns: Columns to compute returns for. If None, uses all numeric.
        periods: Number of periods for return calculation.
        
    Returns:
        DataFrame with return columns added.
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df.columns:
            return_col = f"{col}_return_{periods}d"
            df[return_col] = df[col].pct_change(periods=periods) * 100
    
    return df


def compute_moving_averages(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Compute simple moving averages.
    
    Args:
        df: DataFrame with data.
        columns: Columns to compute MAs for.
        windows: Window sizes for moving averages.
        
    Returns:
        DataFrame with MA columns added.
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            for window in windows:
                ma_col = f"{col}_MA{window}"
                df[ma_col] = df[col].rolling(window=window).mean()
    
    return df


def compute_volatility(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Compute rolling volatility (standard deviation).
    
    Args:
        df: DataFrame with data.
        columns: Columns to compute volatility for.
        windows: Window sizes.
        
    Returns:
        DataFrame with volatility columns added.
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            for window in windows:
                vol_col = f"{col}_vol{window}"
                df[vol_col] = df[col].rolling(window=window).std()
    
    return df


def compute_momentum(
    df: pd.DataFrame,
    columns: List[str],
    periods: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Compute momentum (price change over N periods).
    
    Args:
        df: DataFrame with data.
        columns: Columns to compute momentum for.
        periods: Lookback periods.
        
    Returns:
        DataFrame with momentum columns added.
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            for period in periods:
                mom_col = f"{col}_mom{period}"
                df[mom_col] = df[col] - df[col].shift(period)
    
    return df


def compute_rsi(
    series: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute Relative Strength Index.
    
    Args:
        series: Price series.
        period: RSI period.
        
    Returns:
        RSI series.
    """
    delta = series.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands.
    
    Args:
        series: Price series.
        window: Moving average window.
        num_std: Number of standard deviations.
        
    Returns:
        Tuple of (middle_band, upper_band, lower_band).
    """
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return middle, upper, lower


# =============================================================================
# Market Regime Detection
# =============================================================================

def detect_market_regime(
    vix: pd.Series,
    thresholds: Tuple[float, float, float] = (15, 25, 35),
) -> pd.Series:
    """
    Detect market regime based on VIX levels.
    
    Args:
        vix: VIX index series.
        thresholds: (low, medium, high) thresholds.
        
    Returns:
        Series with regime labels (0=calm, 1=normal, 2=elevated, 3=crisis).
    """
    low, medium, high = thresholds
    
    conditions = [
        vix < low,
        (vix >= low) & (vix < medium),
        (vix >= medium) & (vix < high),
        vix >= high,
    ]
    choices = [0, 1, 2, 3]
    
    return pd.Series(
        np.select(conditions, choices, default=1),
        index=vix.index,
        name="market_regime",
    )


def compute_drawdown(series: pd.Series) -> pd.Series:
    """
    Compute drawdown from peak.
    
    Args:
        series: Price or index series.
        
    Returns:
        Drawdown series (as percentage, negative values).
    """
    peak = series.expanding().max()
    drawdown = (series - peak) / peak * 100
    return drawdown


# =============================================================================
# Feature Engineering Pipeline
# =============================================================================

def engineer_features(
    df: pd.DataFrame,
    add_returns: bool = False,
    add_volatility: bool = False,
    add_momentum: bool = False,
    vix_column: str = "VIX Index",
    equity_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply feature engineering pipeline.
    
    Note: The pre-trained model expects specific 62 features.
    This function is for exploration and future model training.
    Do NOT use additional features with the pre-trained model.
    
    Args:
        df: Raw market data.
        add_returns: Add return features.
        add_volatility: Add volatility features.
        add_momentum: Add momentum features.
        vix_column: Name of VIX column.
        equity_columns: Equity index columns for additional features.
        
    Returns:
        DataFrame with engineered features.
    """
    df = df.copy()
    
    if equity_columns is None:
        equity_columns = ["MXUS Index", "MXWO Index", "ES1 Index", "NQ1 Index"]
    
    # Filter to columns that exist
    equity_columns = [c for c in equity_columns if c in df.columns]
    
    if add_returns and equity_columns:
        df = compute_returns(df, columns=equity_columns, periods=1)
        df = compute_returns(df, columns=equity_columns, periods=5)
    
    if add_volatility and vix_column in df.columns:
        df = compute_volatility(df, columns=[vix_column], windows=[5, 10])
    
    if add_momentum and equity_columns:
        df = compute_momentum(df, columns=equity_columns, periods=[5, 10])
    
    # Add market regime if VIX available
    if vix_column in df.columns:
        df["market_regime"] = detect_market_regime(df[vix_column])
    
    return df


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    from .data_loader import load_financial_market_data, prepare_features
    
    print("=" * 60)
    print("Feature Engineering Test")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df = load_financial_market_data()
    
    # Test transformer
    print("\n=== Feature Transformer ===")
    dates, features = prepare_features(df)
    
    transformer = FeatureTransformer()
    transformer.fit(features)
    
    normalized = transformer.transform_zscore(features)
    print(f"Normalized shape: {normalized.shape}")
    print(f"Mean of first column: {normalized.iloc[:, 0].mean():.6f} (should be ~0)")
    print(f"Std of first column: {normalized.iloc[:, 0].std():.6f} (should be ~1)")
    
    # Test technical indicators
    print("\n=== Technical Indicators ===")
    if "VIX Index" in df.columns:
        regime = detect_market_regime(df["VIX Index"].dropna())
        print(f"Market Regime Distribution:")
        print(regime.value_counts().sort_index())
    
    # Test engineered features
    print("\n=== Engineered Features ===")
    engineered = engineer_features(df, add_returns=True, add_momentum=True)
    new_cols = [c for c in engineered.columns if c not in df.columns]
    print(f"New columns added: {len(new_cols)}")
    print(f"Sample: {new_cols[:5]}")

