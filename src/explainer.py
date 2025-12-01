"""
Explainability Module for Market Anomaly Detector
==================================================

This module provides SHAP-based explanations for model predictions,
helping users understand which features drive crash predictions.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from .model_utils import get_model
from .feature_schema import MODEL_FEATURES, get_feature_category


# =============================================================================
# Explanation Data Classes
# =============================================================================

@dataclass
class FeatureContribution:
    """Single feature's contribution to prediction."""
    feature: str
    value: float
    shap_value: float
    category: str
    direction: str  # "increases_risk" or "decreases_risk"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "value": self.value,
            "shap_value": self.shap_value,
            "category": self.category,
            "direction": self.direction,
        }


@dataclass  
class PredictionExplanation:
    """Complete explanation for a single prediction."""
    base_value: float
    prediction_value: float
    contributions: List[FeatureContribution]
    
    @property
    def top_risk_factors(self) -> List[FeatureContribution]:
        """Get top features increasing risk."""
        return [c for c in self.contributions if c.direction == "increases_risk"][:5]
    
    @property
    def top_protective_factors(self) -> List[FeatureContribution]:
        """Get top features decreasing risk."""
        return [c for c in self.contributions if c.direction == "decreases_risk"][:5]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_value": self.base_value,
            "prediction_value": self.prediction_value,
            "top_risk_factors": [c.to_dict() for c in self.top_risk_factors],
            "top_protective_factors": [c.to_dict() for c in self.top_protective_factors],
        }


# =============================================================================
# SHAP Explainer
# =============================================================================

class CrashExplainer:
    """
    SHAP-based explainer for crash predictions.
    
    Provides local and global explanations for model predictions.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the explainer.
        
        Args:
            model_path: Optional path to model file.
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is required but not installed. "
                "Install with: pip install shap"
            )
        
        self.model = get_model(model_path)
        self._explainer = None
        self._expected_value = None
    
    def _get_explainer(self) -> "shap.TreeExplainer":
        """Get or create the SHAP explainer."""
        if self._explainer is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self._explainer = shap.TreeExplainer(self.model)
                self._expected_value = self._explainer.expected_value
                # Handle case where expected_value is an array
                if isinstance(self._expected_value, np.ndarray):
                    if len(self._expected_value) > 1:
                        self._expected_value = self._expected_value[1]  # Use class 1 (crash)
                    else:
                        self._expected_value = float(self._expected_value[0])
        return self._explainer
    
    def explain_single(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> PredictionExplanation:
        """
        Explain a single prediction.
        
        Args:
            features: Single row of features.
            feature_names: Feature names (uses defaults if None).
            
        Returns:
            PredictionExplanation object.
        """
        if feature_names is None:
            feature_names = MODEL_FEATURES
        
        # Ensure 2D array
        if isinstance(features, pd.DataFrame):
            feature_values = features.values.flatten()
            features_array = features.values
        else:
            feature_values = features.flatten()
            features_array = features.reshape(1, -1) if features.ndim == 1 else features
        
        # Get SHAP values
        explainer = self._get_explainer()
        shap_values = explainer.shap_values(features_array)
        
        # Handle multi-output case (binary classification)
        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                shap_values = shap_values[1]  # Use class 1 (crash)
            else:
                shap_values = shap_values[0]
        
        shap_values = shap_values.flatten()
        
        # Create contributions
        contributions = []
        for i, (name, value, shap_val) in enumerate(zip(feature_names, feature_values, shap_values)):
            contribution = FeatureContribution(
                feature=name,
                value=float(value),
                shap_value=float(shap_val),
                category=get_feature_category(name),
                direction="increases_risk" if shap_val > 0 else "decreases_risk",
            )
            contributions.append(contribution)
        
        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)
        
        # Calculate prediction value
        prediction_value = float(self._expected_value + np.sum(shap_values))
        
        return PredictionExplanation(
            base_value=float(self._expected_value),
            prediction_value=prediction_value,
            contributions=contributions,
        )
    
    def explain_batch(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Get SHAP values for multiple samples.
        
        Args:
            features: Multiple rows of features.
            feature_names: Feature names.
            
        Returns:
            Array of SHAP values (N, features).
        """
        if isinstance(features, pd.DataFrame):
            features = features.values
        
        explainer = self._get_explainer()
        shap_values = explainer.shap_values(features)
        
        # Handle multi-output case (binary classification)
        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                shap_values = shap_values[1]  # Use class 1 (crash)
            else:
                shap_values = shap_values[0]
        
        return shap_values
    
    def get_global_importance(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get global feature importance based on mean |SHAP|.
        
        Args:
            features: Feature data for computing SHAP.
            feature_names: Feature names.
            
        Returns:
            DataFrame with global importance.
        """
        if feature_names is None:
            feature_names = MODEL_FEATURES
        
        shap_values = self.explain_batch(features)
        
        # Mean absolute SHAP value
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
            "category": [get_feature_category(f) for f in feature_names],
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def get_category_importance(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get importance aggregated by feature category.
        
        Args:
            features: Feature data.
            feature_names: Feature names.
            
        Returns:
            DataFrame with category importance.
        """
        global_importance = self.get_global_importance(features, feature_names)
        
        category_importance = global_importance.groupby("category").agg({
            "mean_abs_shap": ["sum", "mean", "count"]
        }).reset_index()
        
        category_importance.columns = ["category", "total_importance", "mean_importance", "feature_count"]
        category_importance = category_importance.sort_values("total_importance", ascending=False)
        
        return category_importance


# =============================================================================
# Convenience Functions
# =============================================================================

def explain_prediction(
    features: Union[pd.DataFrame, np.ndarray],
) -> PredictionExplanation:
    """
    Quick function to explain a single prediction.
    
    Args:
        features: Single row of features.
        
    Returns:
        PredictionExplanation object.
    """
    explainer = CrashExplainer()
    return explainer.explain_single(features)


def get_top_contributors(
    features: Union[pd.DataFrame, np.ndarray],
    n_risk: int = 5,
    n_protective: int = 5,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Get top contributing features for a prediction.
    
    Args:
        features: Single row of features.
        n_risk: Number of risk factors to return.
        n_protective: Number of protective factors to return.
        
    Returns:
        Tuple of (risk_factors, protective_factors) as lists of dicts.
    """
    explanation = explain_prediction(features)
    
    risk_factors = [c.to_dict() for c in explanation.top_risk_factors[:n_risk]]
    protective_factors = [c.to_dict() for c in explanation.top_protective_factors[:n_protective]]
    
    return risk_factors, protective_factors


def format_explanation_text(explanation: PredictionExplanation) -> str:
    """
    Format explanation as human-readable text.
    
    Args:
        explanation: PredictionExplanation object.
        
    Returns:
        Formatted string.
    """
    lines = []
    lines.append(f"Prediction Score: {explanation.prediction_value:.4f}")
    lines.append(f"Base Value: {explanation.base_value:.4f}")
    lines.append("")
    
    if explanation.top_risk_factors:
        lines.append("Top Risk Factors (increasing crash probability):")
        for contrib in explanation.top_risk_factors:
            lines.append(f"  • {contrib.feature}: {contrib.shap_value:+.4f}")
    
    lines.append("")
    
    if explanation.top_protective_factors:
        lines.append("Top Protective Factors (decreasing crash probability):")
        for contrib in explanation.top_protective_factors:
            lines.append(f"  • {contrib.feature}: {contrib.shap_value:+.4f}")
    
    return "\n".join(lines)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    from .data_loader import load_financial_market_data, prepare_features
    
    print("=" * 60)
    print("Explainer Module Test")
    print("=" * 60)
    
    if not SHAP_AVAILABLE:
        print("\n⚠ SHAP not installed. Install with: pip install shap")
        exit(1)
    
    # Load data
    print("\n[1] Loading data...")
    df = load_financial_market_data()
    dates, features = prepare_features(df)
    print(f"  ✓ Loaded {len(features)} samples")
    
    # Initialize explainer
    print("\n[2] Initializing explainer...")
    explainer = CrashExplainer()
    print(f"  ✓ Explainer initialized")
    
    # Explain single prediction (latest)
    print("\n[3] Explaining latest prediction...")
    explanation = explainer.explain_single(features.iloc[[-1]])
    print(f"\n{format_explanation_text(explanation)}")
    
    # Global importance (sample of data for speed)
    print("\n[4] Computing global importance (sample)...")
    sample_features = features.sample(min(200, len(features)), random_state=42)
    global_importance = explainer.get_global_importance(sample_features)
    print("\nTop 10 globally important features:")
    for _, row in global_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['mean_abs_shap']:.4f} ({row['category']})")
    
    # Category importance
    print("\n[5] Category importance:")
    category_importance = explainer.get_category_importance(sample_features)
    for _, row in category_importance.iterrows():
        print(f"  {row['category']}: {row['total_importance']:.4f} ({int(row['feature_count'])} features)")

