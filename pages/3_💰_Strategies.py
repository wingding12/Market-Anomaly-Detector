"""
Investment Strategies Page
==========================

Backtest and compare data-driven investment strategies based on crash predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Strategies | Market Anomaly Detector",
    page_icon="💰",
    layout="wide",
)

from src import (
    load_financial_market_data,
    prepare_features,
    CrashPredictor,
)

from src.investment_strategies import (
    StrategyName,
    StrategyConfig,
    create_strategy,
    compare_strategies,
    calculate_drawdown_series,
    get_strategy_weights_history,
    get_current_recommendation,
    calculate_metrics,
    DynamicRiskAllocation,
    RegimeSwitching,
    ProbabilityWeightedHedge,
    MomentumRiskOverlay,
)


# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data
def load_data():
    df = load_financial_market_data()
    dates, features = prepare_features(df)
    return df, dates, features


@st.cache_resource
def get_predictor():
    return CrashPredictor()


@st.cache_data
def get_predictions(_predictor, features):
    return _predictor.predict_batch(features)


@st.cache_data
def calculate_market_returns(features):
    """Calculate proxy market returns from MSCI World index."""
    if "MXWO Index" in features.columns:
        prices = features["MXWO Index"]
        returns = prices.pct_change().fillna(0)
        return returns
    return pd.Series(0.002, index=features.index)


@st.cache_data
def calculate_bond_returns(features):
    """Calculate proxy bond returns from Treasury data."""
    # Use inverse of yield changes as bond return proxy
    if "GT10 Govt" in features.columns:
        yields = features["GT10 Govt"]
        # Approximate: 1% yield rise = ~8% price drop for 10Y bond
        yield_changes = yields.diff().fillna(0) / 100
        returns = -yield_changes * 8
        returns = returns.clip(-0.05, 0.05)  # Cap extreme moves
        return returns
    return pd.Series(0.0004, index=features.index)


# =============================================================================
# Visualizations
# =============================================================================

def create_cumulative_returns_chart(
    strategy_returns: dict,
    benchmark_returns: pd.Series,
    dates: pd.Series,
):
    """Create cumulative returns comparison chart."""
    fig = go.Figure()
    
    # Benchmark
    cum_benchmark = (1 + benchmark_returns).cumprod()
    fig.add_trace(go.Scatter(
        x=dates,
        y=cum_benchmark,
        name="Buy & Hold 60/40",
        line=dict(color="#6b7280", width=2, dash="dash"),
    ))
    
    # Strategies
    colors = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444"]
    for (name, returns), color in zip(strategy_returns.items(), colors):
        cum_returns = (1 + returns).cumprod()
        fig.add_trace(go.Scatter(
            x=dates,
            y=cum_returns,
            name=name,
            line=dict(color=color, width=2),
        ))
    
    fig.update_layout(
        title="Cumulative Returns Comparison",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fafafa",
        height=450,
        xaxis=dict(title="Date", gridcolor="#333"),
        yaxis=dict(title="Growth of $1", gridcolor="#333"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    
    return fig


def create_drawdown_chart(
    strategy_returns: dict,
    benchmark_returns: pd.Series,
    dates: pd.Series,
):
    """Create drawdown comparison chart."""
    fig = go.Figure()
    
    # Benchmark drawdown
    benchmark_dd = calculate_drawdown_series(benchmark_returns)
    fig.add_trace(go.Scatter(
        x=dates,
        y=benchmark_dd * 100,
        name="Buy & Hold 60/40",
        line=dict(color="#6b7280", width=1),
        fill="tozeroy",
        fillcolor="rgba(107, 114, 128, 0.3)",
    ))
    
    # Strategy drawdowns
    colors = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444"]
    for (name, returns), color in zip(strategy_returns.items(), colors):
        dd = calculate_drawdown_series(returns)
        fig.add_trace(go.Scatter(
            x=dates,
            y=dd * 100,
            name=name,
            line=dict(color=color, width=2),
        ))
    
    fig.update_layout(
        title="Drawdown Comparison",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fafafa",
        height=350,
        xaxis=dict(title="Date", gridcolor="#333"),
        yaxis=dict(title="Drawdown (%)", gridcolor="#333"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    
    return fig


def create_weights_chart(weights_df: pd.DataFrame, dates: pd.Series):
    """Create stacked area chart of portfolio weights."""
    fig = go.Figure()
    
    colors = {"equity": "#22c55e", "bonds": "#3b82f6", "cash": "#f59e0b", "hedge": "#ef4444"}
    
    for col in ["equity", "bonds", "cash"]:
        if col in weights_df.columns:
            fig.add_trace(go.Scatter(
                x=dates,
                y=weights_df[col] * 100,
                name=col.capitalize(),
                stackgroup="one",
                fillcolor=colors.get(col, "#6b7280"),
                line=dict(width=0),
            ))
    
    fig.update_layout(
        title="Dynamic Portfolio Allocation Over Time",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fafafa",
        height=350,
        xaxis=dict(title="Date", gridcolor="#333"),
        yaxis=dict(title="Allocation (%)", gridcolor="#333", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    
    return fig


def create_risk_return_scatter(comparison_df: pd.DataFrame):
    """Create risk-return scatter plot."""
    fig = go.Figure()
    
    colors = ["#6b7280", "#22c55e", "#3b82f6", "#f59e0b", "#ef4444"]
    
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        fig.add_trace(go.Scatter(
            x=[row["Volatility"] * 100],
            y=[row["Annual Return"] * 100],
            mode="markers+text",
            name=row["Strategy"],
            marker=dict(size=15, color=colors[i % len(colors)]),
            text=[row["Strategy"]],
            textposition="top center",
            textfont=dict(size=10),
        ))
    
    fig.update_layout(
        title="Risk-Return Profile",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fafafa",
        height=400,
        xaxis=dict(title="Volatility (%)", gridcolor="#333"),
        yaxis=dict(title="Annual Return (%)", gridcolor="#333"),
        showlegend=False,
    )
    
    return fig


def create_metrics_table(comparison_df: pd.DataFrame):
    """Format comparison dataframe for display."""
    display_df = comparison_df.copy()
    
    display_df["Total Return"] = display_df["Total Return"].apply(lambda x: f"{x:.1%}")
    display_df["Annual Return"] = display_df["Annual Return"].apply(lambda x: f"{x:.1%}")
    display_df["Volatility"] = display_df["Volatility"].apply(lambda x: f"{x:.1%}")
    display_df["Sharpe"] = display_df["Sharpe"].apply(lambda x: f"{x:.2f}")
    display_df["Sortino"] = display_df["Sortino"].apply(lambda x: f"{x:.2f}")
    display_df["Max Drawdown"] = display_df["Max Drawdown"].apply(lambda x: f"{x:.1%}")
    display_df["Calmar"] = display_df["Calmar"].apply(lambda x: f"{x:.2f}")
    
    return display_df


# =============================================================================
# Main Page
# =============================================================================

def main():
    st.title("💰 Investment Strategies")
    st.markdown("Data-driven strategies based on crash probability predictions")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df, dates, features = load_data()
        predictor = get_predictor()
        predictions = get_predictions(predictor, features)
        
        crash_probs = predictions["crash_probability"]
        market_returns = calculate_market_returns(features)
        bond_returns = calculate_bond_returns(features)
    
    # Sidebar configuration
    st.sidebar.header("Strategy Configuration")
    
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance",
        options=["conservative", "moderate", "aggressive"],
        index=1,
    )
    
    base_equity = st.sidebar.slider(
        "Base Equity Allocation",
        min_value=30, max_value=80, value=60, step=5,
        format="%d%%"
    ) / 100
    
    transaction_cost = st.sidebar.slider(
        "Transaction Cost (bps)",
        min_value=0, max_value=50, value=10, step=5,
    ) / 10000
    
    # Current recommendation
    st.subheader("📊 Current Recommendation")
    
    current_prob = crash_probs.iloc[-1]
    recommendation = get_current_recommendation(current_prob, risk_tolerance)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stance_colors = {
            "Risk-On": "#22c55e",
            "Cautious": "#eab308",
            "Defensive": "#f97316",
            "Maximum Defense": "#ef4444",
        }
        st.markdown(f"""
        <div style="padding: 1.5rem; background: linear-gradient(135deg, {stance_colors[recommendation['stance']]}20, {stance_colors[recommendation['stance']]}40); border-radius: 12px; border-left: 4px solid {stance_colors[recommendation['stance']]};">
            <div style="font-size: 0.9rem; color: #888;">Current Stance</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: {stance_colors[recommendation['stance']]};">{recommendation['stance']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Crash Probability", f"{current_prob:.1%}")
        st.metric("Target Equity", f"{recommendation['target_weights']['equity']:.0%}")
    
    with col3:
        st.metric("Target Bonds", f"{recommendation['target_weights']['bonds']:.0%}")
        st.metric("Target Cash", f"{recommendation['target_weights']['cash']:.0%}")
    
    # Action items
    st.markdown("**Recommended Actions:**")
    for action in recommendation["actions"]:
        st.markdown(f"• {action}")
    
    st.info(f"**Hedging:** {recommendation['hedging']}")
    
    st.markdown("---")
    
    # Strategy backtest
    st.subheader("📈 Strategy Backtest Results")
    
    # Create strategies with custom config
    config = StrategyConfig(
        name=StrategyName.DYNAMIC_RISK,
        base_equity_weight=base_equity,
        transaction_cost=transaction_cost,
    )
    
    # Run backtests
    strategies_to_test = [
        ("Dynamic Risk", DynamicRiskAllocation(config)),
        ("Regime Switch", RegimeSwitching(config)),
        ("Probability Hedge", ProbabilityWeightedHedge(config)),
        ("Momentum Overlay", MomentumRiskOverlay(config)),
    ]
    
    strategy_returns = {}
    for name, strategy in strategies_to_test:
        returns, metrics = strategy.backtest(crash_probs, market_returns, bond_returns)
        strategy_returns[name] = returns
    
    # Benchmark
    benchmark_returns = base_equity * market_returns + (1 - base_equity) * bond_returns
    
    # Comparison table
    comparison_data = []
    
    # Benchmark
    benchmark_metrics = calculate_metrics(benchmark_returns)
    comparison_data.append({
        "Strategy": "Buy & Hold 60/40",
        "Total Return": benchmark_metrics.total_return,
        "Annual Return": benchmark_metrics.annualized_return,
        "Volatility": benchmark_metrics.volatility,
        "Sharpe": benchmark_metrics.sharpe_ratio,
        "Sortino": benchmark_metrics.sortino_ratio,
        "Max Drawdown": benchmark_metrics.max_drawdown,
        "Calmar": benchmark_metrics.calmar_ratio,
    })
    
    # Strategies
    for name, returns in strategy_returns.items():
        metrics = calculate_metrics(returns)
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
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display metrics table
    st.dataframe(
        create_metrics_table(comparison_df),
        use_container_width=True,
        hide_index=True,
    )
    
    # Highlight best performer
    best_sharpe = comparison_df.loc[comparison_df["Sharpe"].idxmax(), "Strategy"]
    best_return = comparison_df.loc[comparison_df["Total Return"].idxmax(), "Strategy"]
    min_dd = comparison_df.loc[comparison_df["Max Drawdown"].idxmax(), "Strategy"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"**Best Sharpe Ratio:** {best_sharpe}")
    with col2:
        st.success(f"**Highest Return:** {best_return}")
    with col3:
        st.success(f"**Smallest Drawdown:** {min_dd}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_returns = create_cumulative_returns_chart(
            strategy_returns, benchmark_returns, dates["Date"]
        )
        st.plotly_chart(fig_returns, use_container_width=True)
    
    with col2:
        fig_scatter = create_risk_return_scatter(comparison_df)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Drawdown chart
    fig_drawdown = create_drawdown_chart(
        strategy_returns, benchmark_returns, dates["Date"]
    )
    st.plotly_chart(fig_drawdown, use_container_width=True)
    
    st.markdown("---")
    
    # Dynamic allocation visualization
    st.subheader("📊 Dynamic Allocation History")
    
    selected_strategy = st.selectbox(
        "Select Strategy to Visualize",
        options=["Dynamic Risk", "Regime Switch", "Probability Hedge"],
        index=0,
    )
    
    strategy_map = {
        "Dynamic Risk": DynamicRiskAllocation(config),
        "Regime Switch": RegimeSwitching(config),
        "Probability Hedge": ProbabilityWeightedHedge(config),
    }
    
    weights_df = get_strategy_weights_history(
        strategy_map[selected_strategy], crash_probs
    )
    weights_df.index = dates["Date"]
    
    fig_weights = create_weights_chart(weights_df, dates["Date"])
    st.plotly_chart(fig_weights, use_container_width=True)
    
    # Strategy explanation
    st.markdown("---")
    st.subheader("📚 Strategy Descriptions")
    
    with st.expander("Dynamic Risk Allocation"):
        st.markdown("""
        **Approach:** Scales equity exposure inversely proportional to crash probability.
        
        As risk increases, the strategy systematically reduces equity and increases bonds/cash.
        The key insight is that you don't wait for confirmation of a crash - you reduce 
        exposure proportionally as warning signs accumulate.
        
        **Formula:** `equity_weight = max_weight - (crash_prob × (max_weight - min_weight))`
        
        **Best for:** Investors who want smooth, continuous risk adjustment without sharp transitions.
        """)
    
    with st.expander("Regime Switching"):
        st.markdown("""
        **Approach:** Binary switching between predefined portfolio allocations based on risk regime.
        
        - **Low Risk (<25%):** Full equity exposure (80%)
        - **Elevated (25-50%):** Balanced allocation (50%)
        - **High Risk (50-75%):** Defensive positioning (25%)
        - **Critical (>75%):** Maximum defense (10%)
        
        **Best for:** Investors who prefer clear rules and potentially more tax-efficient due to less frequent rebalancing.
        """)
    
    with st.expander("Probability-Weighted Hedge"):
        st.markdown("""
        **Approach:** Maintains base equity exposure but adds proportional hedges.
        
        Uses crash probability to size hedge positions (puts, VIX calls, inverse exposure).
        The hedge ratio increases linearly with crash probability up to a maximum allocation.
        
        **Key feature:** Preserves upside participation while providing tail protection.
        The cost is hedge premium decay during low-risk periods.
        
        **Best for:** Investors who want to stay invested but need crash protection.
        """)
    
    with st.expander("Momentum + Risk Overlay"):
        st.markdown("""
        **Approach:** Combines price momentum signals with crash probability overlay.
        
        1. Calculate market momentum (price vs 20-week moving average)
        2. When momentum positive AND risk low: Full exposure
        3. When momentum positive BUT risk high: Reduced exposure
        4. When momentum negative: Exit regardless of risk level
        
        **Best for:** Trend-following investors who want additional crash protection.
        """)


if __name__ == "__main__":
    main()

