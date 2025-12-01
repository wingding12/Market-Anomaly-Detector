"""
Market Anomaly Detector
=======================
An early warning system for detecting potential financial market crashes.

Main Streamlit application entry point.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Market Anomaly Detector",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/Market-Anomaly-Detector",
        "Report a bug": "https://github.com/yourusername/Market-Anomaly-Detector/issues",
        "About": "# Market Anomaly Detector\nAn early warning system for financial market crashes."
    }
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Risk level cards */
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .risk-low { background: linear-gradient(135deg, #22c55e20, #22c55e40); border-left: 4px solid #22c55e; }
    .risk-medium { background: linear-gradient(135deg, #eab30820, #eab30840); border-left: 4px solid #eab308; }
    .risk-high { background: linear-gradient(135deg, #f9731620, #f9731640); border-left: 4px solid #f97316; }
    .risk-critical { background: linear-gradient(135deg, #ef444420, #ef444440); border-left: 4px solid #ef4444; }
    
    /* Metric styling */
    .big-metric {
        font-size: 3rem;
        font-weight: bold;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Strategy card */
    .strategy-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d3d);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #3d3d4d;
    }
    
    /* Action items */
    .action-item {
        padding: 0.75rem 1rem;
        background: #ffffff10;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #6366f1;
    }
    
    /* Feature importance bar */
    .feature-bar {
        height: 24px;
        border-radius: 4px;
        margin-bottom: 4px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Import application modules
# =============================================================================

from src import (
    load_financial_market_data,
    prepare_features,
    get_data_summary,
    CrashPredictor,
    CrashExplainer,
    StrategyEngine,
    StrategyType,
    RiskLevel,
    RISK_COLORS,
    RISK_DESCRIPTIONS,
    get_hedge_recommendations,
    format_explanation_text,
)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "df" not in st.session_state:
        st.session_state.df = None
    if "dates" not in st.session_state:
        st.session_state.dates = None
    if "features" not in st.session_state:
        st.session_state.features = None
    if "predictor" not in st.session_state:
        st.session_state.predictor = None
    if "explainer" not in st.session_state:
        st.session_state.explainer = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = None


@st.cache_resource
def load_predictor():
    """Load the crash predictor (cached)."""
    return CrashPredictor()


@st.cache_resource
def load_explainer():
    """Load the SHAP explainer (cached)."""
    return CrashExplainer()


@st.cache_data
def load_and_prepare_data():
    """Load and prepare the financial data (cached)."""
    df = load_financial_market_data()
    dates, features = prepare_features(df)
    return df, dates, features


@st.cache_data
def get_all_predictions(_predictor, features):
    """Get predictions for all data points (cached)."""
    return _predictor.predict_batch(features)


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render the sidebar with settings and info."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
        st.title("Market Anomaly Detector")
        st.markdown("---")
        
        # Data status
        st.subheader("📊 Data Status")
        
        if st.session_state.data_loaded:
            summary = get_data_summary(st.session_state.df)
            st.success("✅ Data Loaded")
            st.caption(f"**Samples:** {len(st.session_state.features):,}")
            st.caption(f"**Date Range:** {summary['date_range']['start']} to {summary['date_range']['end']}")
        else:
            st.warning("⏳ Loading data...")
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            options=[s.value.capitalize() for s in StrategyType],
            index=2,  # Default: Balanced
            help="Your investment risk tolerance affects strategy recommendations"
        )
        st.session_state.risk_tolerance = StrategyType(risk_tolerance.lower())
        
        st.markdown("---")
        
        # Quick stats
        if st.session_state.predictions is not None:
            st.subheader("📈 Quick Stats")
            preds = st.session_state.predictions
            
            col1, col2 = st.columns(2)
            with col1:
                current_prob = preds["crash_probability"].iloc[-1]
                st.metric("Current Risk", f"{current_prob:.0%}")
            with col2:
                avg_prob = preds["crash_probability"].mean()
                st.metric("Avg Risk", f"{avg_prob:.0%}")
            
            # Risk distribution mini chart
            risk_counts = preds["risk_level"].value_counts()
            st.caption("Risk Distribution")
            for level in ["low", "medium", "high", "critical"]:
                count = risk_counts.get(level, 0)
                pct = count / len(preds) * 100
                st.progress(pct / 100, text=f"{level.capitalize()}: {pct:.0f}%")
        
        st.markdown("---")
        
        # Info
        st.caption("**Model:** XGBoost Classifier")
        st.caption("**Features:** 62 market indicators")
        st.caption("**Data:** Weekly observations")
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.data_loaded = False
            st.rerun()


# =============================================================================
# Main Dashboard
# =============================================================================

def render_risk_gauge(probability: float, risk_level: RiskLevel):
    """Render the main risk gauge display."""
    # Use value lookup for compatibility
    color_map = {
        "low": "#22c55e",
        "medium": "#eab308", 
        "high": "#f97316",
        "critical": "#ef4444",
    }
    color = color_map.get(risk_level.value, "#6366f1")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Main probability display
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <div class="metric-label">Crash Probability</div>
            <div class="big-metric" style="color: {color};">{probability:.1%}</div>
            <div style="font-size: 1.5rem; color: {color}; font-weight: 600; text-transform: uppercase;">
                {risk_level.value} Risk
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar visualization
        st.progress(float(probability))
        
        # Risk description
        desc_map = {
            "low": "Normal market conditions. Continue regular operations.",
            "medium": "Elevated uncertainty. Increase monitoring frequency.",
            "high": "Significant risk detected. Consider reducing exposure.",
            "critical": "High crash probability. Implement defensive positioning.",
        }
        st.info(desc_map.get(risk_level.value, "Unknown risk level"))


def render_allocation_chart(allocations):
    """Render portfolio allocation recommendations."""
    import plotly.graph_objects as go
    
    # Create data for the chart
    labels = [a.asset_class.capitalize() for a in allocations]
    current = [a.current_weight * 100 for a in allocations]
    target = [a.target_weight * 100 for a in allocations]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Current',
        x=labels,
        y=current,
        marker_color='#6366f1',
        opacity=0.6,
    ))
    
    fig.add_trace(go.Bar(
        name='Target',
        x=labels,
        y=target,
        marker_color='#22c55e',
    ))
    
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#fafafa',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Weight (%)", gridcolor='#333'),
        xaxis=dict(gridcolor='#333'),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(explanation, n_features=8):
    """Render feature importance from SHAP."""
    import plotly.graph_objects as go
    
    # Get top contributors
    risk_factors = explanation.top_risk_factors[:n_features//2]
    protective = explanation.top_protective_factors[:n_features//2]
    
    all_factors = risk_factors + protective
    all_factors.sort(key=lambda x: x.shap_value)
    
    features = [f.feature for f in all_factors]
    values = [f.shap_value for f in all_factors]
    colors = ['#ef4444' if v > 0 else '#22c55e' for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#fafafa',
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title="SHAP Value (Impact on Risk)", gridcolor='#333', zeroline=True, zerolinecolor='#666'),
        yaxis=dict(gridcolor='#333'),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_historical_chart(dates, predictions):
    """Render historical crash probability chart."""
    import plotly.graph_objects as go
    
    df_plot = pd.DataFrame({
        'Date': dates['Date'],
        'Probability': predictions['crash_probability'] * 100,
        'Risk': predictions['risk_level'],
    })
    
    fig = go.Figure()
    
    # Add risk zones
    fig.add_hrect(y0=0, y1=25, fillcolor="#22c55e", opacity=0.1, line_width=0)
    fig.add_hrect(y0=25, y1=50, fillcolor="#eab308", opacity=0.1, line_width=0)
    fig.add_hrect(y0=50, y1=75, fillcolor="#f97316", opacity=0.1, line_width=0)
    fig.add_hrect(y0=75, y1=100, fillcolor="#ef4444", opacity=0.1, line_width=0)
    
    # Add probability line
    fig.add_trace(go.Scatter(
        x=df_plot['Date'],
        y=df_plot['Probability'],
        mode='lines',
        name='Crash Probability',
        line=dict(color='#6366f1', width=2),
        fill='tozeroy',
        fillcolor='rgba(99, 102, 241, 0.2)',
    ))
    
    # Add threshold lines
    fig.add_hline(y=50, line_dash="dash", line_color="#f97316", opacity=0.7)
    fig.add_hline(y=75, line_dash="dash", line_color="#ef4444", opacity=0.7)
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#fafafa',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(title="Date", gridcolor='#333'),
        yaxis=dict(title="Crash Probability (%)", gridcolor='#333', range=[0, 100]),
        showlegend=False,
        hovermode='x unified',
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_dashboard():
    """Render the main dashboard."""
    st.title("📉 Market Anomaly Detector")
    st.markdown("Real-time early warning system for financial market crashes")
    st.markdown("---")
    
    # Load data if needed
    if not st.session_state.data_loaded:
        with st.spinner("Loading market data and model..."):
            df, dates, features = load_and_prepare_data()
            predictor = load_predictor()
            
            st.session_state.df = df
            st.session_state.dates = dates
            st.session_state.features = features
            st.session_state.predictor = predictor
            st.session_state.predictions = get_all_predictions(predictor, features)
            st.session_state.data_loaded = True
            st.rerun()
    
    # Get latest prediction
    predictor = st.session_state.predictor
    features = st.session_state.features
    dates = st.session_state.dates
    predictions = st.session_state.predictions
    
    latest_result = predictor.predict_single(features.iloc[[-1]])
    latest_date = dates.iloc[-1]['Date'].strftime('%B %d, %Y')
    
    # Date header
    st.caption(f"📅 Latest Data: **{latest_date}**")
    
    # Main risk display
    render_risk_gauge(latest_result.crash_probability, latest_result.risk_level)
    
    st.markdown("---")
    
    # Two column layout for details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Strategy Recommendation")
        
        # Get strategy recommendation
        risk_tolerance = st.session_state.get("risk_tolerance", StrategyType.BALANCED)
        engine = StrategyEngine(risk_tolerance=risk_tolerance)
        recommendation = engine.get_recommendation(latest_result)
        
        # Strategy summary
        st.markdown(f"""
        <div class="strategy-card">
            <div style="font-size: 1.2rem; font-weight: 600; color: #6366f1; margin-bottom: 0.5rem;">
                {recommendation.recommended_strategy.value.upper()} POSITIONING
            </div>
            <div style="color: #aaa; font-size: 0.9rem;">
                {recommendation.summary}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Allocation chart
        st.caption("**Recommended Allocation**")
        render_allocation_chart(recommendation.allocations)
        
        # Action items
        st.caption("**Action Items**")
        for action in recommendation.actions[:5]:
            st.markdown(f"""
            <div class="action-item">
                {action}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🔍 Risk Drivers")
        
        # Load explainer and get explanation
        with st.spinner("Analyzing risk factors..."):
            try:
                explainer = load_explainer()
                explanation = explainer.explain_single(features.iloc[[-1]])
                
                st.caption("**Feature Impact on Risk (SHAP)**")
                render_feature_importance(explanation)
                
                # Top factors summary
                st.caption("**Key Insights**")
                
                if explanation.top_risk_factors:
                    st.error(f"🔺 **Top Risk Driver:** {explanation.top_risk_factors[0].feature}")
                
                if explanation.top_protective_factors:
                    st.success(f"🛡️ **Top Protection:** {explanation.top_protective_factors[0].feature}")
                    
            except Exception as e:
                st.warning(f"Could not load explanations: {str(e)}")
        
        # Hedge recommendations for high risk
        if latest_result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            st.markdown("")
            st.caption("**Hedge Recommendations**")
            hedges = get_hedge_recommendations(latest_result.risk_level)
            for hedge in hedges[:3]:
                st.markdown(f"• **{hedge.instrument}**: {hedge.action}")
    
    st.markdown("---")
    
    # Historical chart
    st.subheader("📈 Historical Crash Probability")
    render_historical_chart(dates, predictions)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_risk_periods = (predictions["risk_level"].isin(["high", "critical"])).sum()
        st.metric("High Risk Periods", f"{high_risk_periods}", f"{high_risk_periods/len(predictions)*100:.0f}%")
    
    with col2:
        max_prob = predictions["crash_probability"].max()
        st.metric("Max Probability", f"{max_prob:.0%}")
    
    with col3:
        avg_prob = predictions["crash_probability"].mean()
        st.metric("Average Probability", f"{avg_prob:.0%}")
    
    with col4:
        current_prob = predictions["crash_probability"].iloc[-1]
        prev_prob = predictions["crash_probability"].iloc[-2]
        delta = current_prob - prev_prob
        st.metric("Current vs Previous", f"{current_prob:.0%}", f"{delta:+.1%}")


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_dashboard()


if __name__ == "__main__":
    main()
