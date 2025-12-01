# Market Anomaly Detector - Source Package
# This package contains the core modules for market crash prediction

__version__ = "0.1.0"

# Feature Schema
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

# Data Loader
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

# Feature Engineering
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

# Model Utilities
from .model_utils import (
    load_model,
    validate_model,
    get_model_info,
    get_model,
    clear_model_cache,
    get_feature_importance,
)

# Predictor
from .predictor import (
    RiskLevel,
    RISK_THRESHOLDS,
    RISK_COLORS,
    RISK_DESCRIPTIONS,
    PredictionResult,
    CrashPredictor,
    predict_crash_probability,
    classify_risk_level,
    get_risk_summary,
)

# Explainer
from .explainer import (
    FeatureContribution,
    PredictionExplanation,
    CrashExplainer,
    explain_prediction,
    get_top_contributors,
    format_explanation_text,
)

# Strategy Engine
from .strategy_engine import (
    StrategyType,
    ActionType,
    AssetAllocation,
    StrategyRecommendation,
    HedgeRecommendation,
    StrategyEngine,
    get_hedge_recommendations,
    format_recommendation_text,
    DEFAULT_ALLOCATIONS,
    RISK_STRATEGY_MAP,
    STRATEGY_DETAILS,
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
    # Model Utilities
    "load_model",
    "validate_model",
    "get_model_info",
    "get_model",
    "clear_model_cache",
    "get_feature_importance",
    # Predictor
    "RiskLevel",
    "RISK_THRESHOLDS",
    "RISK_COLORS",
    "RISK_DESCRIPTIONS",
    "PredictionResult",
    "CrashPredictor",
    "predict_crash_probability",
    "classify_risk_level",
    "get_risk_summary",
    # Explainer
    "FeatureContribution",
    "PredictionExplanation",
    "CrashExplainer",
    "explain_prediction",
    "get_top_contributors",
    "format_explanation_text",
    # Strategy Engine
    "StrategyType",
    "ActionType",
    "AssetAllocation",
    "StrategyRecommendation",
    "HedgeRecommendation",
    "StrategyEngine",
    "get_hedge_recommendations",
    "format_recommendation_text",
    "DEFAULT_ALLOCATIONS",
    "RISK_STRATEGY_MAP",
    "STRATEGY_DETAILS",
]
