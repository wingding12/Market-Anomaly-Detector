# Market Anomaly Detector - Source Package
# This package contains the core modules for market crash prediction

__version__ = "0.1.0"

from .feature_schema import (
    MODEL_FEATURES,
    N_FEATURES,
    N_BASE_FEATURES,
    N_LAG_FEATURES,
    LAG_SOURCE_FEATURES,
    TOP_FEATURES_BY_IMPORTANCE,
    validate_features,
    get_feature_category,
)

from .data_loader import (
    load_financial_market_data,
    load_user_csv,
    prepare_features,
    validate_data,
    get_data_summary,
    fill_missing_values,
    compute_lag_features,
    load_data_cached,
    clear_cache,
)

from .feature_engineering import (
    FeatureTransformer,
    compute_returns,
    compute_moving_averages,
    compute_volatility,
    compute_momentum,
    compute_rsi,
    compute_bollinger_bands,
    detect_market_regime,
    compute_drawdown,
    engineer_features,
)

__all__ = [
    # Version
    "__version__",
    # Feature Schema
    "MODEL_FEATURES",
    "N_FEATURES",
    "N_BASE_FEATURES",
    "N_LAG_FEATURES",
    "LAG_SOURCE_FEATURES",
    "TOP_FEATURES_BY_IMPORTANCE",
    "validate_features",
    "get_feature_category",
    # Data Loader
    "load_financial_market_data",
    "load_user_csv",
    "prepare_features",
    "validate_data",
    "get_data_summary",
    "fill_missing_values",
    "compute_lag_features",
    "load_data_cached",
    "clear_cache",
    # Feature Engineering
    "FeatureTransformer",
    "compute_returns",
    "compute_moving_averages",
    "compute_volatility",
    "compute_momentum",
    "compute_rsi",
    "compute_bollinger_bands",
    "detect_market_regime",
    "compute_drawdown",
    "engineer_features",
]
