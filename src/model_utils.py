"""
Model Utilities for Market Anomaly Detector
============================================

This module provides utilities for loading, managing, and validating
the XGBoost crash prediction model.
"""

import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import warnings

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

from .feature_schema import MODEL_FEATURES, N_FEATURES


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "xgb_weights.pkl"


# =============================================================================
# Model Loading
# =============================================================================

def load_model(filepath: Optional[Path] = None) -> Any:
    """
    Load the XGBoost model from a pickle file.
    
    Args:
        filepath: Path to model file. Defaults to xgb_weights.pkl.
        
    Returns:
        Loaded XGBoost model.
        
    Raises:
        FileNotFoundError: If model file doesn't exist.
        ImportError: If XGBoost is not installed.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "XGBoost is required but not installed. "
            "Install with: pip install xgboost"
        )
    
    if filepath is None:
        filepath = DEFAULT_MODEL_PATH
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    # Suppress XGBoost version warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(filepath, "rb") as f:
            model = pickle.load(f)
    
    return model


def validate_model(model: Any) -> Tuple[bool, List[str]]:
    """
    Validate that the loaded model meets requirements.
    
    Args:
        model: Loaded model object.
        
    Returns:
        Tuple of (is_valid, list of issues).
    """
    issues = []
    
    # Check model type
    if not hasattr(model, "predict_proba"):
        issues.append("Model lacks predict_proba method")
    
    if not hasattr(model, "predict"):
        issues.append("Model lacks predict method")
    
    # Check feature count
    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != N_FEATURES:
            issues.append(
                f"Feature count mismatch: model expects {model.n_features_in_}, "
                f"schema defines {N_FEATURES}"
            )
    
    # Check feature names
    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
        if model_features != MODEL_FEATURES:
            mismatched = [
                (i, m, s) for i, (m, s) in enumerate(zip(model_features, MODEL_FEATURES))
                if m != s
            ]
            if mismatched:
                issues.append(f"Feature name mismatch at positions: {[m[0] for m in mismatched[:5]]}")
    
    # Check classes
    if hasattr(model, "classes_"):
        if len(model.classes_) != 2:
            issues.append(f"Expected 2 classes, got {len(model.classes_)}")
    
    return len(issues) == 0, issues


def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Extract information about the loaded model.
    
    Args:
        model: Loaded model object.
        
    Returns:
        Dictionary with model information.
    """
    info = {
        "type": type(model).__name__,
        "module": type(model).__module__,
    }
    
    # XGBoost specific attributes
    if hasattr(model, "n_features_in_"):
        info["n_features"] = model.n_features_in_
    
    if hasattr(model, "n_estimators"):
        info["n_estimators"] = model.n_estimators
    
    if hasattr(model, "max_depth"):
        info["max_depth"] = model.max_depth
    
    if hasattr(model, "learning_rate"):
        info["learning_rate"] = model.learning_rate
    
    if hasattr(model, "objective"):
        info["objective"] = model.objective
    
    if hasattr(model, "classes_"):
        info["classes"] = list(model.classes_)
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        info["feature_importance_stats"] = {
            "max": float(np.max(importances)),
            "mean": float(np.mean(importances)),
            "nonzero": int(np.sum(importances > 0)),
        }
    
    return info


# =============================================================================
# Model Caching
# =============================================================================

_model_cache: Dict[str, Any] = {}


def get_model(filepath: Optional[Path] = None, force_reload: bool = False) -> Any:
    """
    Get the model with caching for performance.
    
    Args:
        filepath: Path to model file.
        force_reload: Force reload even if cached.
        
    Returns:
        Loaded model.
    """
    cache_key = str(filepath or "default")
    
    if not force_reload and cache_key in _model_cache:
        return _model_cache[cache_key]
    
    model = load_model(filepath)
    _model_cache[cache_key] = model
    
    return model


def clear_model_cache():
    """Clear the model cache."""
    global _model_cache
    _model_cache = {}


# =============================================================================
# Feature Importance
# =============================================================================

def get_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Get feature importance from the model.
    
    Args:
        model: Loaded model.
        feature_names: Feature names (uses model's if None).
        top_n: Return only top N features.
        
    Returns:
        DataFrame with feature importances.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    if feature_names is None:
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    if top_n is not None:
        importance_df = importance_df.head(top_n)
    
    return importance_df


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Model Utilities Test")
    print("=" * 60)
    
    # Load model
    print("\n[1] Loading model...")
    model = load_model()
    print(f"  ✓ Model loaded: {type(model).__name__}")
    
    # Validate
    print("\n[2] Validating model...")
    is_valid, issues = validate_model(model)
    if is_valid:
        print("  ✓ Model validation passed")
    else:
        print("  ✗ Validation issues:")
        for issue in issues:
            print(f"    - {issue}")
    
    # Get info
    print("\n[3] Model information:")
    info = get_model_info(model)
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Feature importance
    print("\n[4] Top 10 feature importances:")
    importance = get_feature_importance(model, top_n=10)
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Test caching
    print("\n[5] Testing cache...")
    model2 = get_model()
    print(f"  ✓ Cached model retrieved (same object: {model is model2})")

