"""
Predictor Module for Market Anomaly Detector
=============================================

This module provides a high-level prediction interface for crash detection,
including probability scoring, risk classification, and batch predictions.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .model_utils import get_model, get_feature_importance
from .feature_schema import MODEL_FEATURES, N_FEATURES


# =============================================================================
# Risk Level Classification
# =============================================================================

class RiskLevel(Enum):
    """Risk level classification based on crash probability."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Risk thresholds (crash probability)
RISK_THRESHOLDS = {
    RiskLevel.LOW: (0.0, 0.25),
    RiskLevel.MEDIUM: (0.25, 0.50),
    RiskLevel.HIGH: (0.50, 0.75),
    RiskLevel.CRITICAL: (0.75, 1.0),
}

RISK_COLORS = {
    RiskLevel.LOW: "#22c55e",      # Green
    RiskLevel.MEDIUM: "#eab308",   # Yellow
    RiskLevel.HIGH: "#f97316",     # Orange
    RiskLevel.CRITICAL: "#ef4444", # Red
}

RISK_DESCRIPTIONS = {
    RiskLevel.LOW: "Normal market conditions. Continue regular operations.",
    RiskLevel.MEDIUM: "Elevated uncertainty. Increase monitoring frequency.",
    RiskLevel.HIGH: "Significant risk detected. Consider reducing exposure.",
    RiskLevel.CRITICAL: "High crash probability. Implement defensive positioning.",
}


# =============================================================================
# Prediction Result
# =============================================================================

@dataclass
class PredictionResult:
    """Container for prediction results."""
    crash_probability: float
    prediction: int  # 0 or 1
    risk_level: RiskLevel
    confidence: float
    
    @property
    def risk_color(self) -> str:
        """Get color for risk level visualization."""
        return RISK_COLORS[self.risk_level]
    
    @property
    def risk_description(self) -> str:
        """Get description for risk level."""
        return RISK_DESCRIPTIONS[self.risk_level]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "crash_probability": self.crash_probability,
            "prediction": self.prediction,
            "risk_level": self.risk_level.value,
            "risk_color": self.risk_color,
            "risk_description": self.risk_description,
            "confidence": self.confidence,
        }


# =============================================================================
# Market Crash Predictor
# =============================================================================

class CrashPredictor:
    """
    High-level interface for market crash predictions.
    
    Provides probability scoring, risk classification, and batch predictions.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Optional path to model file.
        """
        self.model = get_model(model_path)
        self._validate_model()
    
    def _validate_model(self):
        """Validate the loaded model."""
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Model must support predict_proba")
        
        if hasattr(self.model, "n_features_in_"):
            if self.model.n_features_in_ != N_FEATURES:
                raise ValueError(
                    f"Model expects {self.model.n_features_in_} features, "
                    f"but {N_FEATURES} are defined in schema"
                )
    
    def predict_single(
        self,
        features: Union[pd.DataFrame, np.ndarray],
    ) -> PredictionResult:
        """
        Make a prediction for a single sample.
        
        Args:
            features: Single row of features (1, 62) shape.
            
        Returns:
            PredictionResult object.
        """
        # Ensure 2D array
        if isinstance(features, pd.DataFrame):
            features = features.values
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if features.shape[0] != 1:
            raise ValueError(f"Expected single sample, got {features.shape[0]}")
        
        # Get prediction
        proba = self.model.predict_proba(features)[0]
        crash_prob = proba[1]  # Probability of class 1 (crash)
        prediction = int(crash_prob >= 0.5)
        
        # Determine risk level
        risk_level = self._classify_risk(crash_prob)
        
        # Calculate confidence
        confidence = max(proba)  # Distance from 0.5
        
        return PredictionResult(
            crash_probability=crash_prob,
            prediction=prediction,
            risk_level=risk_level,
            confidence=confidence,
        )
    
    def predict_batch(
        self,
        features: Union[pd.DataFrame, np.ndarray],
    ) -> pd.DataFrame:
        """
        Make predictions for multiple samples.
        
        Args:
            features: Multiple rows of features (N, 62) shape.
            
        Returns:
            DataFrame with predictions.
        """
        if isinstance(features, pd.DataFrame):
            features_array = features.values
        else:
            features_array = features
        
        # Get probabilities
        probas = self.model.predict_proba(features_array)
        crash_probs = probas[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            "crash_probability": crash_probs,
            "prediction": (crash_probs >= 0.5).astype(int),
            "risk_level": [self._classify_risk(p).value for p in crash_probs],
            "confidence": np.maximum(probas[:, 0], probas[:, 1]),
        })
        
        return results
    
    def predict_with_dates(
        self,
        dates: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Make predictions and include dates.
        
        Args:
            dates: DataFrame with Date column.
            features: Features DataFrame.
            
        Returns:
            DataFrame with dates and predictions.
        """
        predictions = self.predict_batch(features)
        
        if "Date" in dates.columns:
            predictions.insert(0, "Date", dates["Date"].values)
        
        return predictions
    
    def _classify_risk(self, probability: float) -> RiskLevel:
        """Classify risk level based on crash probability."""
        for level, (low, high) in RISK_THRESHOLDS.items():
            if low <= probability < high:
                return level
        return RiskLevel.CRITICAL  # Default if >= 1.0
    
    def get_risk_distribution(
        self,
        features: Union[pd.DataFrame, np.ndarray],
    ) -> Dict[str, int]:
        """
        Get distribution of risk levels for a batch.
        
        Args:
            features: Feature data.
            
        Returns:
            Dictionary with counts per risk level.
        """
        predictions = self.predict_batch(features)
        return predictions["risk_level"].value_counts().to_dict()
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get top feature importances from the model."""
        return get_feature_importance(self.model, top_n=top_n)


# =============================================================================
# Convenience Functions
# =============================================================================

def predict_crash_probability(
    features: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """
    Quick function to get crash probabilities.
    
    Args:
        features: Feature array (N, 62).
        
    Returns:
        Array of crash probabilities.
    """
    predictor = CrashPredictor()
    if isinstance(features, pd.DataFrame):
        features = features.values
    probas = predictor.model.predict_proba(features)
    return probas[:, 1]


def classify_risk_level(probability: float) -> str:
    """
    Classify a single probability into risk level.
    
    Args:
        probability: Crash probability (0-1).
        
    Returns:
        Risk level string.
    """
    for level, (low, high) in RISK_THRESHOLDS.items():
        if low <= probability < high:
            return level.value
    return RiskLevel.CRITICAL.value


def get_risk_summary(probabilities: np.ndarray) -> Dict[str, Any]:
    """
    Get summary statistics for a batch of probabilities.
    
    Args:
        probabilities: Array of crash probabilities.
        
    Returns:
        Summary dictionary.
    """
    return {
        "count": len(probabilities),
        "mean": float(np.mean(probabilities)),
        "std": float(np.std(probabilities)),
        "min": float(np.min(probabilities)),
        "max": float(np.max(probabilities)),
        "median": float(np.median(probabilities)),
        "high_risk_count": int(np.sum(probabilities >= 0.5)),
        "critical_count": int(np.sum(probabilities >= 0.75)),
    }


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    from .data_loader import load_financial_market_data, prepare_features
    
    print("=" * 60)
    print("Predictor Module Test")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    df = load_financial_market_data()
    dates, features = prepare_features(df)
    print(f"  ✓ Loaded {len(features)} samples")
    
    # Initialize predictor
    print("\n[2] Initializing predictor...")
    predictor = CrashPredictor()
    print(f"  ✓ Predictor initialized")
    
    # Single prediction
    print("\n[3] Single prediction (latest data point):")
    result = predictor.predict_single(features.iloc[[-1]])
    print(f"  Date: {dates.iloc[-1]['Date'].strftime('%Y-%m-%d')}")
    print(f"  Crash Probability: {result.crash_probability:.2%}")
    print(f"  Risk Level: {result.risk_level.value.upper()}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Description: {result.risk_description}")
    
    # Batch prediction
    print("\n[4] Batch prediction (all data):")
    predictions = predictor.predict_with_dates(dates, features)
    print(f"  ✓ Generated {len(predictions)} predictions")
    
    # Risk distribution
    print("\n[5] Risk distribution:")
    for level in RiskLevel:
        count = (predictions["risk_level"] == level.value).sum()
        pct = count / len(predictions) * 100
        print(f"  {level.value.upper()}: {count} ({pct:.1f}%)")
    
    # Summary statistics
    print("\n[6] Summary statistics:")
    summary = get_risk_summary(predictions["crash_probability"].values)
    print(f"  Mean probability: {summary['mean']:.2%}")
    print(f"  Std deviation: {summary['std']:.2%}")
    print(f"  Range: {summary['min']:.2%} - {summary['max']:.2%}")
    print(f"  High risk periods: {summary['high_risk_count']}")
    print(f"  Critical periods: {summary['critical_count']}")
    
    # Feature importance
    print("\n[7] Top 5 features:")
    importance = predictor.get_feature_importance(top_n=5)
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

