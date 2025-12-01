"""
Historical Analysis Page
========================

Analyze historical market crash periods and model performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Historical | Market Anomaly Detector",
    page_icon="📜",
    layout="wide",
)

# Import modules
from src import (
    load_financial_market_data,
    prepare_features,
    CrashPredictor,
    detect_market_regime,
)


# =============================================================================
# Data Loading (cached)
# =============================================================================

@st.cache_data
def load_data():
    df = load_financial_market_data()
    dates, features = prepare_features(df)
    return df, dates, features


@st.cache_resource
def load_predictor():
    return CrashPredictor()


@st.cache_data
def get_predictions(_predictor, features):
    return _predictor.predict_batch(features)


# =============================================================================
# Historical Event Analysis
# =============================================================================

HISTORICAL_EVENTS = {
    "Dot-com Bubble (2000-2002)": {
        "start": "2000-03-01",
        "end": "2002-10-31",
        "description": "Technology stock bubble burst, leading to significant market decline."
    },
    "Global Financial Crisis (2007-2009)": {
        "start": "2007-10-01",
        "end": "2009-03-31",
        "description": "Subprime mortgage crisis led to global banking collapse."
    },
    "European Debt Crisis (2010-2012)": {
        "start": "2010-04-01",
        "end": "2012-07-31",
        "description": "Sovereign debt concerns across European nations."
    },
    "China Stock Market Crash (2015-2016)": {
        "start": "2015-06-01",
        "end": "2016-02-28",
        "description": "Chinese market turbulence and global growth concerns."
    },
    "COVID-19 Crash (2020)": {
        "start": "2020-02-01",
        "end": "2020-04-30",
        "description": "Pandemic-induced market selloff and volatility spike."
    },
}


def create_event_analysis_chart(dates, predictions, features, event_name, event_info):
    """Create detailed chart for a historical event."""
    
    start = pd.Timestamp(event_info['start'])
    end = pd.Timestamp(event_info['end'])
    
    # Get data before, during, and after event
    buffer_days = pd.Timedelta(days=180)
    
    mask_period = (dates['Date'] >= start - buffer_days) & (dates['Date'] <= end + buffer_days)
    mask_event = (dates['Date'] >= start) & (dates['Date'] <= end)
    
    period_dates = dates[mask_period]['Date']
    period_probs = predictions[mask_period]['crash_probability'] * 100
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add event shading
    fig.add_vrect(
        x0=start, x1=end,
        fillcolor="rgba(239, 68, 68, 0.2)",
        line_width=0,
        annotation_text=event_name,
        annotation_position="top left",
    )
    
    # Crash probability
    fig.add_trace(
        go.Scatter(
            x=period_dates,
            y=period_probs,
            name='Crash Probability',
            line=dict(color='#ef4444', width=2),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.1)',
        ),
        secondary_y=False,
    )
    
    # Risk thresholds
    fig.add_hline(y=50, line_dash="dash", line_color="#f97316", secondary_y=False)
    fig.add_hline(y=75, line_dash="dash", line_color="#ef4444", secondary_y=False)
    
    # VIX if available
    if 'VIX Index' in features.columns:
        period_vix = features[mask_period]['VIX Index']
        fig.add_trace(
            go.Scatter(
                x=period_dates,
                y=period_vix,
                name='VIX Index',
                line=dict(color='#6366f1', width=1, dash='dot'),
            ),
            secondary_y=True,
        )
    
    fig.update_layout(
        title=f'{event_name}',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#fafafa',
        height=400,
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
    )
    
    fig.update_xaxes(title_text='Date', gridcolor='#333')
    fig.update_yaxes(title_text='Crash Probability (%)', gridcolor='#333', range=[0, 100], secondary_y=False)
    fig.update_yaxes(title_text='VIX', gridcolor='#333', secondary_y=True)
    
    return fig


def calculate_event_statistics(dates, predictions, event_info):
    """Calculate statistics for an event period."""
    start = pd.Timestamp(event_info['start'])
    end = pd.Timestamp(event_info['end'])
    
    mask = (dates['Date'] >= start) & (dates['Date'] <= end)
    event_preds = predictions[mask]
    
    if len(event_preds) == 0:
        return None
    
    probs = event_preds['crash_probability']
    risk_dist = event_preds['risk_level'].value_counts()
    
    return {
        'periods': len(event_preds),
        'mean_prob': probs.mean(),
        'max_prob': probs.max(),
        'min_prob': probs.min(),
        'high_risk_pct': (event_preds['risk_level'].isin(['high', 'critical'])).mean() * 100,
        'critical_pct': (event_preds['risk_level'] == 'critical').mean() * 100,
    }


def create_comparison_chart(dates, predictions):
    """Create comparison chart across all historical events."""
    
    event_stats = []
    for name, info in HISTORICAL_EVENTS.items():
        stats = calculate_event_statistics(dates, predictions, info)
        if stats:
            event_stats.append({
                'Event': name.split('(')[0].strip(),
                'Mean Probability': stats['mean_prob'] * 100,
                'Max Probability': stats['max_prob'] * 100,
                'High Risk %': stats['high_risk_pct'],
            })
    
    if not event_stats:
        return None
    
    df_stats = pd.DataFrame(event_stats)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Mean Probability',
        x=df_stats['Event'],
        y=df_stats['Mean Probability'],
        marker_color='#6366f1',
    ))
    
    fig.add_trace(go.Bar(
        name='Max Probability',
        x=df_stats['Event'],
        y=df_stats['Max Probability'],
        marker_color='#ef4444',
    ))
    
    fig.update_layout(
        title='Crash Probability Across Historical Events',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#fafafa',
        height=400,
        margin=dict(l=60, r=20, t=60, b=100),
        barmode='group',
        xaxis=dict(gridcolor='#333', tickangle=-45),
        yaxis=dict(title='Probability (%)', gridcolor='#333', range=[0, 100]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    
    return fig


def create_regime_timeline(dates, features, predictions):
    """Create market regime timeline."""
    
    if 'VIX Index' not in features.columns:
        return None
    
    regime = detect_market_regime(features['VIX Index'])
    
    df_plot = pd.DataFrame({
        'Date': dates['Date'],
        'Regime': regime.values,
        'Probability': predictions['crash_probability'] * 100,
    })
    
    regime_colors = {0: '#22c55e', 1: '#eab308', 2: '#f97316', 3: '#ef4444'}
    regime_names = {0: 'Calm', 1: 'Normal', 2: 'Elevated', 3: 'Crisis'}
    
    fig = go.Figure()
    
    # Add regime as colored background segments
    for i in range(len(df_plot) - 1):
        fig.add_vrect(
            x0=df_plot['Date'].iloc[i],
            x1=df_plot['Date'].iloc[i+1],
            fillcolor=regime_colors[df_plot['Regime'].iloc[i]],
            opacity=0.2,
            line_width=0,
        )
    
    # Probability line
    fig.add_trace(go.Scatter(
        x=df_plot['Date'],
        y=df_plot['Probability'],
        mode='lines',
        name='Crash Probability',
        line=dict(color='#fafafa', width=2),
    ))
    
    fig.update_layout(
        title='Crash Probability by Market Regime (VIX-based)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#fafafa',
        height=400,
        margin=dict(l=60, r=20, t=60, b=40),
        xaxis=dict(title='Date', gridcolor='#333'),
        yaxis=dict(title='Crash Probability (%)', gridcolor='#333', range=[0, 100]),
    )
    
    return fig


# =============================================================================
# Main Page
# =============================================================================

def main():
    st.title("📜 Historical Analysis")
    st.markdown("Analyze model performance during historical market events")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df, dates, features = load_data()
        predictor = load_predictor()
        predictions = get_predictions(predictor, features)
    
    # Event selection
    st.subheader("🎯 Historical Event Analysis")
    
    selected_event = st.selectbox(
        "Select Historical Event",
        options=list(HISTORICAL_EVENTS.keys()),
        index=1,  # Default to GFC
    )
    
    event_info = HISTORICAL_EVENTS[selected_event]
    
    # Event description
    st.info(f"**{selected_event}**\n\n{event_info['description']}")
    
    # Event chart
    fig_event = create_event_analysis_chart(dates, predictions, features, selected_event, event_info)
    st.plotly_chart(fig_event, use_container_width=True)
    
    # Event statistics
    stats = calculate_event_statistics(dates, predictions, event_info)
    
    if stats:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Periods", stats['periods'])
        with col2:
            st.metric("Mean Prob", f"{stats['mean_prob']:.1%}")
        with col3:
            st.metric("Max Prob", f"{stats['max_prob']:.1%}")
        with col4:
            st.metric("High Risk %", f"{stats['high_risk_pct']:.0f}%")
        with col5:
            st.metric("Critical %", f"{stats['critical_pct']:.0f}%")
    else:
        st.warning("No data available for this event period.")
    
    st.markdown("---")
    
    # Comparison across events
    st.subheader("📊 Cross-Event Comparison")
    
    fig_comparison = create_comparison_chart(dates, predictions)
    if fig_comparison:
        st.plotly_chart(fig_comparison, use_container_width=True)
    else:
        st.warning("Not enough data for comparison chart.")
    
    st.markdown("---")
    
    # Market regime analysis
    st.subheader("🌡️ Market Regime Analysis")
    st.caption("Background colors indicate VIX-based market regime: 🟢 Calm | 🟡 Normal | 🟠 Elevated | 🔴 Crisis")
    
    fig_regime = create_regime_timeline(dates, features, predictions)
    if fig_regime:
        st.plotly_chart(fig_regime, use_container_width=True)
    else:
        st.warning("VIX data not available for regime analysis.")
    
    # Regime statistics
    if 'VIX Index' in features.columns:
        regime = detect_market_regime(features['VIX Index'])
        regime_names = {0: 'Calm', 1: 'Normal', 2: 'Elevated', 3: 'Crisis'}
        
        st.subheader("📈 Statistics by Regime")
        
        regime_df = pd.DataFrame({
            'Regime': regime.values,
            'Probability': predictions['crash_probability'].values,
        })
        
        col1, col2, col3, col4 = st.columns(4)
        
        for i, (idx, name) in enumerate(regime_names.items()):
            regime_data = regime_df[regime_df['Regime'] == idx]['Probability']
            
            with [col1, col2, col3, col4][i]:
                if len(regime_data) > 0:
                    st.markdown(f"**{name}** ({len(regime_data)} periods)")
                    st.metric("Mean Prob", f"{regime_data.mean():.1%}")
                    st.caption(f"Range: {regime_data.min():.1%} - {regime_data.max():.1%}")
                else:
                    st.markdown(f"**{name}**")
                    st.caption("No data")
    
    # High risk periods table
    st.markdown("---")
    st.subheader("⚠️ Top High-Risk Periods")
    
    high_risk_df = pd.DataFrame({
        'Date': dates['Date'],
        'Crash Probability': predictions['crash_probability'],
        'Risk Level': predictions['risk_level'],
    })
    high_risk_df = high_risk_df.sort_values('Crash Probability', ascending=False).head(20)
    high_risk_df['Crash Probability'] = high_risk_df['Crash Probability'].apply(lambda x: f"{x:.1%}")
    high_risk_df['Date'] = high_risk_df['Date'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(
        high_risk_df,
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()

