"""
Investment Strategies Module
============================

Data-driven investment strategies based on crash probability predictions.
Implements systematic approaches to minimize drawdowns and optimize risk-adjusted returns.

Strategies:
1. Dynamic Risk Allocation - Scales equity exposure inversely to crash probability
2. Regime-Based Switching - Binary switches between risk-on/risk-off portfolios
3. Probability-Weighted Hedging - Proportional hedge ratios based on risk levels
4. Momentum + Risk Overlay - Combines trend signals with crash protection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Strategy Configuration
# =============================================================================

class StrategyName(Enum):
    """Available investment strategies."""
    BUY_AND_HOLD = "buy_and_hold"
    DYNAMIC_RISK = "dynamic_risk_allocation"
    REGIME_SWITCH = "regime_switching"
    PROBABILITY_HEDGE = "probability_weighted_hedge"
    MOMENTUM_OVERLAY = "momentum_risk_overlay"


@dataclass
class StrategyConfig:
    """Configuration for investment strategy."""
    name: StrategyName
    base_equity_weight: float = 0.60  # Default equity allocation
    min_equity_weight: float = 0.10   # Minimum equity during high risk
    max_equity_weight: float = 0.90   # Maximum equity during low risk
    risk_free_rate: float = 0.02      # Annual risk-free rate
    rebalance_threshold: float = 0.05 # Rebalance when weights drift >5%
    transaction_cost: float = 0.001   # 10bps per trade
    
    # Risk thresholds
    low_risk_threshold: float = 0.25
    high_risk_threshold: float = 0.50
    critical_threshold: float = 0.75


# =============================================================================
# Performance Metrics
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Strategy performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in periods
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_holding_period: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "Total Return": f"{self.total_return:.2%}",
            "Annualized Return": f"{self.annualized_return:.2%}",
            "Volatility": f"{self.volatility:.2%}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{self.sortino_ratio:.2f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Max DD Duration": f"{self.max_drawdown_duration} periods",
            "Calmar Ratio": f"{self.calmar_ratio:.2f}",
            "Win Rate": f"{self.win_rate:.2%}",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Number of Trades": self.num_trades,
        }


def calculate_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 52,
) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics."""
    
    # Basic returns
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Volatility
    volatility = returns.std() * np.sqrt(periods_per_year)
    
    # Sharpe Ratio
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0.001
    sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
    
    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    # Max drawdown duration
    underwater = drawdowns < 0
    if underwater.any():
        groups = (~underwater).cumsum()
        dd_durations = underwater.groupby(groups).sum()
        max_drawdown_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0
    else:
        max_drawdown_duration = 0
    
    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate and profit factor
    winning_periods = returns[returns > 0]
    losing_periods = returns[returns < 0]
    win_rate = len(winning_periods) / len(returns) if len(returns) > 0 else 0
    
    total_gains = winning_periods.sum() if len(winning_periods) > 0 else 0
    total_losses = abs(losing_periods.sum()) if len(losing_periods) > 0 else 0.001
    profit_factor = total_gains / total_losses if total_losses > 0 else 0
    
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        calmar_ratio=calmar_ratio,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=0,  # Updated by strategy
        avg_holding_period=0,
    )


# =============================================================================
# Strategy Implementations
# =============================================================================

class InvestmentStrategy:
    """Base class for investment strategies."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.positions: List[float] = []
        self.trades: List[Dict] = []
    
    def calculate_weights(
        self,
        crash_probability: float,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Calculate target portfolio weights based on crash probability."""
        raise NotImplementedError
    
    def backtest(
        self,
        crash_probabilities: pd.Series,
        market_returns: pd.Series,
        bond_returns: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, PerformanceMetrics]:
        """Run backtest and return strategy returns and metrics."""
        raise NotImplementedError


class DynamicRiskAllocation(InvestmentStrategy):
    """
    Dynamic Risk Allocation Strategy
    ================================
    
    Scales equity exposure inversely proportional to crash probability.
    As risk increases, systematically reduces equity and increases bonds/cash.
    
    Key insight: Don't wait for confirmation of a crash. Reduce exposure
    proportionally as warning signs accumulate.
    
    Weight calculation:
        equity_weight = max_weight - (crash_prob * (max_weight - min_weight))
    """
    
    def calculate_weights(
        self,
        crash_probability: float,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        
        # Linear scaling between max and min equity weights
        prob_scaled = min(max(crash_probability, 0), 1)
        
        equity_weight = (
            self.config.max_equity_weight - 
            prob_scaled * (self.config.max_equity_weight - self.config.min_equity_weight)
        )
        
        # Allocate remainder to bonds and cash
        remaining = 1 - equity_weight
        bond_weight = remaining * 0.7  # 70% of defensive allocation to bonds
        cash_weight = remaining * 0.3  # 30% to cash
        
        return {
            "equity": equity_weight,
            "bonds": bond_weight,
            "cash": cash_weight,
            "crash_probability": crash_probability,
        }
    
    def backtest(
        self,
        crash_probabilities: pd.Series,
        market_returns: pd.Series,
        bond_returns: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, PerformanceMetrics]:
        
        if bond_returns is None:
            # Assume flat bond returns if not provided
            bond_returns = pd.Series(0.0004, index=market_returns.index)  # ~2% annual
        
        strategy_returns = []
        num_trades = 0
        prev_equity_weight = self.config.base_equity_weight
        
        for i, (prob, mkt_ret, bond_ret) in enumerate(zip(
            crash_probabilities, market_returns, bond_returns
        )):
            weights = self.calculate_weights(prob)
            equity_weight = weights["equity"]
            bond_weight = weights["bonds"]
            
            # Calculate portfolio return
            portfolio_return = (
                equity_weight * mkt_ret +
                bond_weight * bond_ret +
                weights["cash"] * (self.config.risk_free_rate / 52)  # Weekly risk-free
            )
            
            # Transaction costs
            if abs(equity_weight - prev_equity_weight) > self.config.rebalance_threshold:
                turnover = abs(equity_weight - prev_equity_weight)
                portfolio_return -= turnover * self.config.transaction_cost
                num_trades += 1
            
            strategy_returns.append(portfolio_return)
            prev_equity_weight = equity_weight
        
        returns_series = pd.Series(strategy_returns, index=market_returns.index)
        metrics = calculate_metrics(returns_series, self.config.risk_free_rate)
        metrics.num_trades = num_trades
        
        return returns_series, metrics


class RegimeSwitching(InvestmentStrategy):
    """
    Regime Switching Strategy
    =========================
    
    Binary approach: fully invested when risk is low, defensive when high.
    Simpler than dynamic allocation, potentially more tax-efficient.
    
    Regimes:
    - Low Risk (prob < 25%): Full equity exposure
    - Elevated (25-50%): Balanced allocation
    - High Risk (>50%): Defensive positioning
    """
    
    def calculate_weights(
        self,
        crash_probability: float,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        
        if crash_probability < self.config.low_risk_threshold:
            # Risk-on: Full equity
            return {"equity": 0.80, "bonds": 0.15, "cash": 0.05}
        
        elif crash_probability < self.config.high_risk_threshold:
            # Balanced
            return {"equity": 0.50, "bonds": 0.35, "cash": 0.15}
        
        elif crash_probability < self.config.critical_threshold:
            # Defensive
            return {"equity": 0.25, "bonds": 0.45, "cash": 0.30}
        
        else:
            # Maximum defense
            return {"equity": 0.10, "bonds": 0.40, "cash": 0.50}
    
    def backtest(
        self,
        crash_probabilities: pd.Series,
        market_returns: pd.Series,
        bond_returns: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, PerformanceMetrics]:
        
        if bond_returns is None:
            bond_returns = pd.Series(0.0004, index=market_returns.index)
        
        strategy_returns = []
        num_trades = 0
        prev_regime = None
        
        for prob, mkt_ret, bond_ret in zip(
            crash_probabilities, market_returns, bond_returns
        ):
            weights = self.calculate_weights(prob)
            
            # Determine current regime
            if prob < self.config.low_risk_threshold:
                regime = "risk_on"
            elif prob < self.config.high_risk_threshold:
                regime = "balanced"
            elif prob < self.config.critical_threshold:
                regime = "defensive"
            else:
                regime = "max_defense"
            
            # Calculate return
            portfolio_return = (
                weights["equity"] * mkt_ret +
                weights["bonds"] * bond_ret +
                weights["cash"] * (self.config.risk_free_rate / 52)
            )
            
            # Count regime changes as trades
            if prev_regime is not None and regime != prev_regime:
                portfolio_return -= self.config.transaction_cost * 0.5  # Estimate 50% turnover
                num_trades += 1
            
            strategy_returns.append(portfolio_return)
            prev_regime = regime
        
        returns_series = pd.Series(strategy_returns, index=market_returns.index)
        metrics = calculate_metrics(returns_series, self.config.risk_free_rate)
        metrics.num_trades = num_trades
        
        return returns_series, metrics


class ProbabilityWeightedHedge(InvestmentStrategy):
    """
    Probability-Weighted Hedging Strategy
    =====================================
    
    Maintains base equity exposure but adds proportional hedges.
    Uses crash probability to size hedge positions.
    
    Approach:
    - Keep 60% base equity allocation
    - Allocate 0-20% to hedges (puts, VIX, gold) based on probability
    - Hedge ratio = crash_probability * max_hedge_allocation
    
    This preserves upside participation while providing tail protection.
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.max_hedge_allocation = 0.20  # Maximum 20% in hedges
    
    def calculate_weights(
        self,
        crash_probability: float,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        
        # Hedge allocation proportional to crash probability
        hedge_allocation = crash_probability * self.max_hedge_allocation
        
        # Reduce equity slightly when hedging (hedge cost drag)
        equity_weight = self.config.base_equity_weight - (hedge_allocation * 0.5)
        
        remaining = 1 - equity_weight - hedge_allocation
        
        return {
            "equity": max(equity_weight, 0.30),
            "hedge": hedge_allocation,
            "bonds": remaining * 0.6,
            "cash": remaining * 0.4,
        }
    
    def backtest(
        self,
        crash_probabilities: pd.Series,
        market_returns: pd.Series,
        bond_returns: Optional[pd.Series] = None,
        vix_returns: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, PerformanceMetrics]:
        
        if bond_returns is None:
            bond_returns = pd.Series(0.0004, index=market_returns.index)
        
        # Model hedge returns: inversely correlated to market
        # In reality, this would be VIX calls, put spreads, etc.
        if vix_returns is None:
            # Rough approximation: hedges gain when market falls
            hedge_returns = -market_returns * 2  # 2x inverse exposure
            hedge_returns = hedge_returns.clip(-0.20, 0.50)  # Cap gains/losses
        else:
            hedge_returns = vix_returns
        
        strategy_returns = []
        
        for prob, mkt_ret, bond_ret, hedge_ret in zip(
            crash_probabilities, market_returns, bond_returns, hedge_returns
        ):
            weights = self.calculate_weights(prob)
            
            # Hedge cost (premium decay)
            hedge_cost = weights["hedge"] * 0.002  # ~10% annual decay
            
            portfolio_return = (
                weights["equity"] * mkt_ret +
                weights["hedge"] * hedge_ret - hedge_cost +
                weights["bonds"] * bond_ret +
                weights["cash"] * (self.config.risk_free_rate / 52)
            )
            
            strategy_returns.append(portfolio_return)
        
        returns_series = pd.Series(strategy_returns, index=market_returns.index)
        metrics = calculate_metrics(returns_series, self.config.risk_free_rate)
        
        return returns_series, metrics


class MomentumRiskOverlay(InvestmentStrategy):
    """
    Momentum with Risk Overlay Strategy
    ====================================
    
    Combines price momentum signals with crash probability overlay.
    
    Logic:
    1. Calculate market momentum (price vs moving average)
    2. When momentum positive AND risk low: Full exposure
    3. When momentum positive BUT risk high: Reduced exposure
    4. When momentum negative: Exit regardless of risk level
    
    This captures trend-following benefits while adding crash protection.
    """
    
    def __init__(self, config: StrategyConfig, momentum_window: int = 20):
        super().__init__(config)
        self.momentum_window = momentum_window
    
    def calculate_weights(
        self,
        crash_probability: float,
        momentum_signal: float = 1.0,  # 1 = positive, 0 = negative
        current_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        
        if momentum_signal <= 0:
            # Negative momentum: exit equities
            return {"equity": 0.10, "bonds": 0.50, "cash": 0.40}
        
        # Positive momentum: scale by inverse of crash probability
        risk_scalar = 1 - (crash_probability * 0.8)  # At most reduce by 80%
        equity_weight = self.config.max_equity_weight * risk_scalar
        
        remaining = 1 - equity_weight
        return {
            "equity": equity_weight,
            "bonds": remaining * 0.6,
            "cash": remaining * 0.4,
        }
    
    def backtest(
        self,
        crash_probabilities: pd.Series,
        market_returns: pd.Series,
        prices: Optional[pd.Series] = None,
        bond_returns: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, PerformanceMetrics]:
        
        if bond_returns is None:
            bond_returns = pd.Series(0.0004, index=market_returns.index)
        
        # Calculate momentum from cumulative returns if prices not provided
        if prices is None:
            prices = (1 + market_returns).cumprod()
        
        # Moving average momentum signal
        ma = prices.rolling(self.momentum_window).mean()
        momentum_signals = (prices > ma).astype(float)
        momentum_signals = momentum_signals.fillna(1)  # Assume positive initially
        
        strategy_returns = []
        num_trades = 0
        prev_in_market = True
        
        for prob, mkt_ret, bond_ret, mom in zip(
            crash_probabilities, market_returns, bond_returns, momentum_signals
        ):
            weights = self.calculate_weights(prob, mom)
            
            portfolio_return = (
                weights["equity"] * mkt_ret +
                weights["bonds"] * bond_ret +
                weights["cash"] * (self.config.risk_free_rate / 52)
            )
            
            # Track position changes
            in_market = weights["equity"] > 0.30
            if in_market != prev_in_market:
                portfolio_return -= self.config.transaction_cost * 0.4
                num_trades += 1
            prev_in_market = in_market
            
            strategy_returns.append(portfolio_return)
        
        returns_series = pd.Series(strategy_returns, index=market_returns.index)
        metrics = calculate_metrics(returns_series, self.config.risk_free_rate)
        metrics.num_trades = num_trades
        
        return returns_series, metrics


# =============================================================================
# Strategy Factory
# =============================================================================

def create_strategy(
    strategy_name: StrategyName,
    config: Optional[StrategyConfig] = None,
) -> InvestmentStrategy:
    """Factory function to create strategy instances."""
    
    if config is None:
        config = StrategyConfig(name=strategy_name)
    
    strategies = {
        StrategyName.DYNAMIC_RISK: DynamicRiskAllocation,
        StrategyName.REGIME_SWITCH: RegimeSwitching,
        StrategyName.PROBABILITY_HEDGE: ProbabilityWeightedHedge,
        StrategyName.MOMENTUM_OVERLAY: MomentumRiskOverlay,
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategies[strategy_name](config)


# =============================================================================
# Comparison and Analysis
# =============================================================================

def compare_strategies(
    crash_probabilities: pd.Series,
    market_returns: pd.Series,
    bond_returns: Optional[pd.Series] = None,
    strategies: Optional[List[StrategyName]] = None,
) -> pd.DataFrame:
    """Compare multiple strategies against buy-and-hold benchmark."""
    
    if strategies is None:
        strategies = [
            StrategyName.DYNAMIC_RISK,
            StrategyName.REGIME_SWITCH,
            StrategyName.PROBABILITY_HEDGE,
            StrategyName.MOMENTUM_OVERLAY,
        ]
    
    results = {}
    
    # Buy and hold benchmark (60/40)
    if bond_returns is not None:
        benchmark_returns = 0.60 * market_returns + 0.40 * bond_returns
    else:
        benchmark_returns = 0.60 * market_returns + 0.40 * 0.0004
    
    benchmark_metrics = calculate_metrics(benchmark_returns)
    results["Buy & Hold 60/40"] = benchmark_metrics
    
    # Test each strategy
    for strategy_name in strategies:
        strategy = create_strategy(strategy_name)
        returns, metrics = strategy.backtest(
            crash_probabilities, market_returns, bond_returns
        )
        results[strategy_name.value] = metrics
    
    # Create comparison DataFrame
    comparison_data = []
    for name, metrics in results.items():
        comparison_data.append({
            "Strategy": name,
            "Total Return": metrics.total_return,
            "Annual Return": metrics.annualized_return,
            "Volatility": metrics.volatility,
            "Sharpe": metrics.sharpe_ratio,
            "Sortino": metrics.sortino_ratio,
            "Max Drawdown": metrics.max_drawdown,
            "Calmar": metrics.calmar_ratio,
        })
    
    return pd.DataFrame(comparison_data)


def calculate_drawdown_series(returns: pd.Series) -> pd.Series:
    """Calculate drawdown time series."""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    return drawdowns


def get_strategy_weights_history(
    strategy: InvestmentStrategy,
    crash_probabilities: pd.Series,
) -> pd.DataFrame:
    """Get historical weight allocations for a strategy."""
    
    weights_history = []
    for prob in crash_probabilities:
        weights = strategy.calculate_weights(prob)
        weights_history.append(weights)
    
    return pd.DataFrame(weights_history, index=crash_probabilities.index)


# =============================================================================
# Strategy Recommendations
# =============================================================================

def get_current_recommendation(
    crash_probability: float,
    risk_tolerance: str = "moderate",
) -> Dict[str, Any]:
    """
    Get current investment recommendation based on crash probability.
    
    Args:
        crash_probability: Current crash probability (0-1)
        risk_tolerance: "conservative", "moderate", or "aggressive"
    
    Returns:
        Detailed recommendation dictionary
    """
    
    # Adjust thresholds based on risk tolerance
    tolerance_adjustments = {
        "conservative": {"shift": 0.10, "equity_cap": 0.70},
        "moderate": {"shift": 0.0, "equity_cap": 0.80},
        "aggressive": {"shift": -0.10, "equity_cap": 0.95},
    }
    
    adj = tolerance_adjustments.get(risk_tolerance, tolerance_adjustments["moderate"])
    adjusted_prob = crash_probability + adj["shift"]
    
    # Create strategy and get weights
    config = StrategyConfig(
        name=StrategyName.DYNAMIC_RISK,
        max_equity_weight=adj["equity_cap"],
    )
    strategy = DynamicRiskAllocation(config)
    weights = strategy.calculate_weights(adjusted_prob)
    
    # Generate specific recommendations
    if crash_probability < 0.25:
        stance = "Risk-On"
        actions = [
            "Maintain full equity allocation per your risk tolerance",
            "Consider tactical overweights in high-beta sectors",
            "Review stop-losses but no need to tighten",
            "Monitor VIX for early warning signs",
        ]
        hedging = "No hedges required at current risk levels"
        
    elif crash_probability < 0.50:
        stance = "Cautious"
        actions = [
            "Reduce equity allocation by 10-20%",
            "Increase quality bias within equity holdings",
            "Raise cash position for opportunistic deployment",
            "Increase monitoring frequency to daily",
        ]
        hedging = "Consider 2-5% allocation to put spreads or VIX calls"
        
    elif crash_probability < 0.75:
        stance = "Defensive"
        actions = [
            "Reduce equity to minimum comfortable level",
            "Exit speculative and high-beta positions",
            "Increase duration in fixed income",
            "Build cash for potential buying opportunity",
        ]
        hedging = "Implement 5-10% tail hedge via puts or VIX"
        
    else:
        stance = "Maximum Defense"
        actions = [
            "Minimize equity exposure immediately",
            "Exit all leveraged positions",
            "Move to short-duration, high-quality bonds",
            "Hold maximum cash position",
            "Consider inverse ETFs for additional protection",
        ]
        hedging = "Full hedge implementation: 10-15% in puts, VIX calls, or inverse exposure"
    
    return {
        "crash_probability": crash_probability,
        "risk_tolerance": risk_tolerance,
        "stance": stance,
        "target_weights": weights,
        "actions": actions,
        "hedging": hedging,
        "rebalance_urgency": "high" if crash_probability > 0.50 else "normal",
    }


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Investment Strategies Module Test")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_periods = 520  # 10 years of weekly data
    
    # Simulated crash probabilities with some regime changes
    base_prob = np.random.beta(2, 5, n_periods)  # Skewed toward low
    # Add some crisis periods
    base_prob[100:120] = np.random.uniform(0.6, 0.9, 20)  # Crisis 1
    base_prob[300:330] = np.random.uniform(0.5, 0.8, 30)  # Crisis 2
    
    crash_probs = pd.Series(base_prob)
    
    # Simulated market returns (with crashes during high-prob periods)
    market_rets = np.random.normal(0.002, 0.02, n_periods)
    market_rets[100:120] = np.random.normal(-0.03, 0.04, 20)  # Crisis 1 losses
    market_rets[300:330] = np.random.normal(-0.02, 0.03, 30)  # Crisis 2 losses
    market_returns = pd.Series(market_rets)
    
    # Compare strategies
    print("\n[1] Comparing Strategies...")
    comparison = compare_strategies(crash_probs, market_returns)
    print(comparison.to_string())
    
    # Current recommendation
    print("\n[2] Current Recommendation (prob=0.65, moderate risk)...")
    rec = get_current_recommendation(0.65, "moderate")
    print(f"Stance: {rec['stance']}")
    print(f"Target Equity: {rec['target_weights']['equity']:.1%}")
    print(f"Hedging: {rec['hedging']}")
    
    print("\n[3] Strategy completed successfully!")

