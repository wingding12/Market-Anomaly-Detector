"""
Strategy Engine for Market Anomaly Detector
============================================

This module provides investment strategy recommendations based on
crash predictions and risk levels. Strategies range from aggressive
growth to defensive positioning depending on market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .predictor import RiskLevel, RISK_THRESHOLDS, PredictionResult


# =============================================================================
# Strategy Types
# =============================================================================

class StrategyType(Enum):
    """Investment strategy types based on risk tolerance."""
    AGGRESSIVE = "aggressive"
    GROWTH = "growth"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    DEFENSIVE = "defensive"


class ActionType(Enum):
    """Types of recommended actions."""
    HOLD = "hold"
    REDUCE = "reduce"
    INCREASE = "increase"
    HEDGE = "hedge"
    EXIT = "exit"
    REBALANCE = "rebalance"


# =============================================================================
# Asset Classes
# =============================================================================

@dataclass
class AssetAllocation:
    """Recommended allocation for an asset class."""
    asset_class: str
    current_weight: float
    target_weight: float
    action: ActionType
    rationale: str
    
    @property
    def change(self) -> float:
        """Change from current to target weight."""
        return self.target_weight - self.current_weight
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_class": self.asset_class,
            "current_weight": self.current_weight,
            "target_weight": self.target_weight,
            "change": self.change,
            "action": self.action.value,
            "rationale": self.rationale,
        }


# Default portfolio weights by strategy type
DEFAULT_ALLOCATIONS = {
    StrategyType.AGGRESSIVE: {
        "equities": 0.80,
        "bonds": 0.10,
        "commodities": 0.05,
        "cash": 0.05,
    },
    StrategyType.GROWTH: {
        "equities": 0.70,
        "bonds": 0.20,
        "commodities": 0.05,
        "cash": 0.05,
    },
    StrategyType.BALANCED: {
        "equities": 0.55,
        "bonds": 0.30,
        "commodities": 0.05,
        "cash": 0.10,
    },
    StrategyType.CONSERVATIVE: {
        "equities": 0.35,
        "bonds": 0.45,
        "commodities": 0.05,
        "cash": 0.15,
    },
    StrategyType.DEFENSIVE: {
        "equities": 0.15,
        "bonds": 0.40,
        "commodities": 0.10,
        "cash": 0.35,
    },
}

# Risk-level to strategy mapping (use string keys for compatibility)
RISK_STRATEGY_MAP = {
    "low": StrategyType.GROWTH,
    "medium": StrategyType.BALANCED,
    "high": StrategyType.CONSERVATIVE,
    "critical": StrategyType.DEFENSIVE,
}


# =============================================================================
# Strategy Recommendations
# =============================================================================

@dataclass
class StrategyRecommendation:
    """Complete strategy recommendation based on market conditions."""
    risk_level: RiskLevel
    crash_probability: float
    recommended_strategy: StrategyType
    allocations: List[AssetAllocation]
    actions: List[str]
    summary: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level.value,
            "crash_probability": self.crash_probability,
            "recommended_strategy": self.recommended_strategy.value,
            "allocations": [a.to_dict() for a in self.allocations],
            "actions": self.actions,
            "summary": self.summary,
            "confidence": self.confidence,
        }


# Strategy descriptions and action items
STRATEGY_DETAILS = {
    StrategyType.AGGRESSIVE: {
        "summary": "Market conditions favor aggressive positioning. Maximize equity exposure for growth.",
        "actions": [
            "Maintain high equity allocation (80%+)",
            "Focus on growth and momentum stocks",
            "Consider leveraged positions if risk-tolerant",
            "Minimize cash holdings",
            "Monitor VIX for early warning signs",
        ],
    },
    StrategyType.GROWTH: {
        "summary": "Low risk environment supports growth-oriented allocation with moderate protection.",
        "actions": [
            "Maintain equity allocation at 70%",
            "Diversify across sectors and geographies",
            "Hold modest bond position for stability",
            "Keep cash reserves minimal (5%)",
            "Review positions weekly",
        ],
    },
    StrategyType.BALANCED: {
        "summary": "Elevated uncertainty warrants balanced approach between growth and preservation.",
        "actions": [
            "Reduce equity exposure to 55%",
            "Increase bond allocation to 30%",
            "Raise cash position to 10%",
            "Implement stop-loss orders on volatile positions",
            "Increase monitoring to daily reviews",
        ],
    },
    StrategyType.CONSERVATIVE: {
        "summary": "High risk detected. Shift to capital preservation while maintaining some growth exposure.",
        "actions": [
            "Reduce equity to 35%",
            "Increase bonds to 45%",
            "Raise cash to 15%",
            "Exit speculative positions",
            "Consider protective puts on core holdings",
            "Review positions daily",
        ],
    },
    StrategyType.DEFENSIVE: {
        "summary": "Critical risk level. Implement maximum defensive positioning to protect capital.",
        "actions": [
            "Minimize equity exposure to 15%",
            "Maximize bonds and cash to 75%",
            "Hold gold/commodities as hedge (10%)",
            "Exit all leveraged positions",
            "Implement tight stop-losses",
            "Consider VIX calls for crash protection",
            "Monitor hourly during market hours",
        ],
    },
}


# =============================================================================
# Strategy Engine
# =============================================================================

class StrategyEngine:
    """
    Engine for generating investment strategy recommendations.
    
    Takes crash predictions and generates actionable investment advice
    including portfolio allocations and specific action items.
    """
    
    def __init__(
        self,
        current_allocation: Optional[Dict[str, float]] = None,
        risk_tolerance: StrategyType = StrategyType.BALANCED,
    ):
        """
        Initialize the strategy engine.
        
        Args:
            current_allocation: Current portfolio weights by asset class.
            risk_tolerance: Investor's baseline risk tolerance.
        """
        self.current_allocation = current_allocation or DEFAULT_ALLOCATIONS[risk_tolerance]
        self.risk_tolerance = risk_tolerance
    
    def get_recommendation(
        self,
        prediction: PredictionResult,
    ) -> StrategyRecommendation:
        """
        Generate strategy recommendation based on prediction.
        
        Args:
            prediction: PredictionResult from the predictor.
            
        Returns:
            Complete StrategyRecommendation.
        """
        # Determine recommended strategy based on risk level
        base_strategy = RISK_STRATEGY_MAP[prediction.risk_level.value]
        
        # Adjust for investor risk tolerance
        recommended_strategy = self._adjust_for_tolerance(base_strategy)
        
        # Generate allocations
        allocations = self._generate_allocations(recommended_strategy)
        
        # Get actions and summary
        details = STRATEGY_DETAILS[recommended_strategy]
        
        return StrategyRecommendation(
            risk_level=prediction.risk_level,
            crash_probability=prediction.crash_probability,
            recommended_strategy=recommended_strategy,
            allocations=allocations,
            actions=details["actions"],
            summary=details["summary"],
            confidence=prediction.confidence,
        )
    
    def _adjust_for_tolerance(self, base_strategy: StrategyType) -> StrategyType:
        """Adjust strategy based on investor's risk tolerance."""
        strategy_order = list(StrategyType)
        base_idx = strategy_order.index(base_strategy)
        tolerance_idx = strategy_order.index(self.risk_tolerance)
        
        # Blend towards investor's tolerance (but don't go more aggressive than market suggests)
        if tolerance_idx < base_idx:  # Investor more aggressive
            # Only shift one level more aggressive at most
            adjusted_idx = max(base_idx - 1, tolerance_idx)
        else:  # Investor more conservative or same
            # Can shift more conservative based on tolerance
            adjusted_idx = min(base_idx + 1, tolerance_idx)
        
        return strategy_order[adjusted_idx]
    
    def _generate_allocations(
        self,
        strategy: StrategyType,
    ) -> List[AssetAllocation]:
        """Generate allocation recommendations."""
        target = DEFAULT_ALLOCATIONS[strategy]
        allocations = []
        
        for asset_class, target_weight in target.items():
            current_weight = self.current_allocation.get(asset_class, 0.0)
            change = target_weight - current_weight
            
            # Determine action
            if abs(change) < 0.02:
                action = ActionType.HOLD
                rationale = "Maintain current position"
            elif change > 0:
                action = ActionType.INCREASE
                rationale = f"Increase by {abs(change)*100:.0f}pp for protection" if asset_class in ["bonds", "cash"] else f"Increase by {abs(change)*100:.0f}pp for growth"
            else:
                action = ActionType.REDUCE
                rationale = f"Reduce by {abs(change)*100:.0f}pp to lower risk exposure"
            
            allocations.append(AssetAllocation(
                asset_class=asset_class,
                current_weight=current_weight,
                target_weight=target_weight,
                action=action,
                rationale=rationale,
            ))
        
        return allocations
    
    def get_recommendation_from_probability(
        self,
        crash_probability: float,
    ) -> StrategyRecommendation:
        """
        Generate recommendation directly from crash probability.
        
        Args:
            crash_probability: Crash probability (0-1).
            
        Returns:
            StrategyRecommendation.
        """
        # Create a mock PredictionResult
        risk_level = self._classify_risk(crash_probability)
        prediction = PredictionResult(
            crash_probability=crash_probability,
            prediction=1 if crash_probability >= 0.5 else 0,
            risk_level=risk_level,
            confidence=max(crash_probability, 1 - crash_probability),
        )
        return self.get_recommendation(prediction)
    
    def _classify_risk(self, probability: float) -> RiskLevel:
        """Classify risk level from probability."""
        for level, (low, high) in RISK_THRESHOLDS.items():
            if low <= probability < high:
                return level
        return RiskLevel.CRITICAL


# =============================================================================
# Hedging Recommendations
# =============================================================================

@dataclass
class HedgeRecommendation:
    """Recommendation for hedging strategies."""
    instrument: str
    action: str
    size_percent: float
    rationale: str
    cost_estimate: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument": self.instrument,
            "action": self.action,
            "size_percent": self.size_percent,
            "rationale": self.rationale,
            "cost_estimate": self.cost_estimate,
        }


def get_hedge_recommendations(
    risk_level: RiskLevel,
    portfolio_value: float = 100000,
) -> List[HedgeRecommendation]:
    """
    Get hedging recommendations based on risk level.
    
    Args:
        risk_level: Current risk level.
        portfolio_value: Portfolio value for sizing.
        
    Returns:
        List of hedge recommendations.
    """
    hedges = []
    
    if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
        # VIX calls
        hedges.append(HedgeRecommendation(
            instrument="VIX Call Options",
            action="Buy OTM calls, 30-60 DTE",
            size_percent=2.0 if risk_level == RiskLevel.HIGH else 5.0,
            rationale="VIX spikes during crashes, providing crash protection",
            cost_estimate="1-3% of notional protected",
        ))
        
        # Put options on SPY
        hedges.append(HedgeRecommendation(
            instrument="SPY Put Options",
            action="Buy ATM or slightly OTM puts",
            size_percent=5.0 if risk_level == RiskLevel.HIGH else 10.0,
            rationale="Direct downside protection for equity exposure",
            cost_estimate="2-5% of notional protected",
        ))
    
    if risk_level == RiskLevel.CRITICAL:
        # Treasury futures
        hedges.append(HedgeRecommendation(
            instrument="Treasury Futures (TY, ZN)",
            action="Long position for flight-to-safety",
            size_percent=10.0,
            rationale="Treasuries rally during risk-off events",
            cost_estimate="Margin requirement ~3% of notional",
        ))
        
        # Gold
        hedges.append(HedgeRecommendation(
            instrument="Gold (GLD, GC futures)",
            action="Establish or increase position",
            size_percent=5.0,
            rationale="Safe haven asset during market stress",
            cost_estimate="Full notional or futures margin",
        ))
    
    return hedges


# =============================================================================
# Historical Strategy Performance
# =============================================================================

def backtest_strategy(
    predictions: pd.DataFrame,
    returns: pd.Series,
    initial_strategy: StrategyType = StrategyType.BALANCED,
) -> Dict[str, Any]:
    """
    Backtest strategy switching based on predictions.
    
    Args:
        predictions: DataFrame with risk_level column.
        returns: Market returns series (aligned with predictions).
        initial_strategy: Starting strategy.
        
    Returns:
        Backtest results.
    """
    # Map risk levels to equity weights
    equity_weights = {
        "low": 0.70,
        "medium": 0.55,
        "high": 0.35,
        "critical": 0.15,
    }
    
    # Calculate strategy returns
    strategy_weights = predictions["risk_level"].map(equity_weights)
    strategy_returns = returns * strategy_weights
    
    # Buy and hold for comparison
    buy_hold_returns = returns * 0.60  # 60% equity baseline
    
    # Calculate metrics
    cumulative_strategy = (1 + strategy_returns).cumprod()
    cumulative_buyhold = (1 + buy_hold_returns).cumprod()
    
    return {
        "strategy_total_return": float(cumulative_strategy.iloc[-1] - 1),
        "buyhold_total_return": float(cumulative_buyhold.iloc[-1] - 1),
        "strategy_volatility": float(strategy_returns.std() * np.sqrt(52)),  # Annualized
        "buyhold_volatility": float(buy_hold_returns.std() * np.sqrt(52)),
        "max_drawdown_strategy": float(_max_drawdown(cumulative_strategy)),
        "max_drawdown_buyhold": float(_max_drawdown(cumulative_buyhold)),
        "periods_defensive": int((predictions["risk_level"] == "critical").sum()),
        "periods_conservative": int((predictions["risk_level"] == "high").sum()),
    }


def _max_drawdown(cumulative: pd.Series) -> float:
    """Calculate maximum drawdown."""
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


# =============================================================================
# Formatting Utilities
# =============================================================================

def format_recommendation_text(rec: StrategyRecommendation) -> str:
    """Format recommendation as human-readable text."""
    lines = []
    
    # Header
    lines.append(f"{'='*60}")
    lines.append(f"STRATEGY RECOMMENDATION")
    lines.append(f"{'='*60}")
    lines.append("")
    
    # Risk assessment
    lines.append(f"Risk Level: {rec.risk_level.value.upper()}")
    lines.append(f"Crash Probability: {rec.crash_probability:.1%}")
    lines.append(f"Confidence: {rec.confidence:.1%}")
    lines.append("")
    
    # Strategy
    lines.append(f"Recommended Strategy: {rec.recommended_strategy.value.upper()}")
    lines.append(f"Summary: {rec.summary}")
    lines.append("")
    
    # Allocations
    lines.append("Target Allocations:")
    for alloc in rec.allocations:
        change_str = f"{alloc.change:+.0%}" if abs(alloc.change) >= 0.01 else "—"
        lines.append(f"  {alloc.asset_class.capitalize():12} {alloc.target_weight:5.0%}  ({change_str})")
    lines.append("")
    
    # Actions
    lines.append("Recommended Actions:")
    for i, action in enumerate(rec.actions, 1):
        lines.append(f"  {i}. {action}")
    
    return "\n".join(lines)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    from .data_loader import load_financial_market_data, prepare_features
    from .predictor import CrashPredictor
    
    print("=" * 60)
    print("Strategy Engine Test")
    print("=" * 60)
    
    # Load data and make prediction
    print("\n[1] Loading data and making prediction...")
    df = load_financial_market_data()
    dates, features = prepare_features(df)
    
    predictor = CrashPredictor()
    result = predictor.predict_single(features.iloc[[-1]])
    latest_date = dates.iloc[-1]['Date'].strftime('%Y-%m-%d')
    
    print(f"  Date: {latest_date}")
    print(f"  Crash Probability: {result.crash_probability:.1%}")
    print(f"  Risk Level: {result.risk_level.value}")
    
    # Generate recommendation
    print("\n[2] Generating strategy recommendation...")
    engine = StrategyEngine(risk_tolerance=StrategyType.BALANCED)
    recommendation = engine.get_recommendation(result)
    
    print(f"\n{format_recommendation_text(recommendation)}")
    
    # Test different risk levels
    print("\n[3] Testing all risk levels...")
    for level in RiskLevel:
        prob = (RISK_THRESHOLDS[level][0] + RISK_THRESHOLDS[level][1]) / 2
        rec = engine.get_recommendation_from_probability(prob)
        print(f"  {level.value.upper():10} → {rec.recommended_strategy.value.upper():12} (Equity: {rec.allocations[0].target_weight:.0%})")
    
    # Hedge recommendations
    print("\n[4] Hedge recommendations for CRITICAL risk...")
    hedges = get_hedge_recommendations(RiskLevel.CRITICAL)
    for hedge in hedges:
        print(f"  • {hedge.instrument}: {hedge.action}")

