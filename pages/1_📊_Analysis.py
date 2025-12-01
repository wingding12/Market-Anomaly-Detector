"""
Analysis Page - Detailed Market Analysis
========================================

Provides in-depth analysis tools including:
- Interactive date range selection
- Feature contribution breakdown
- Risk distribution analysis
- Correlation analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Analysis | Market Anomaly Detector",
    page_icon="📊",
    layout="wide",
)

# Import modules
from src import (
    load_financial_market_data,
    prepare_features,
    CrashPredictor,
    CrashExplainer,
    get_feature_category,
    MODEL_FEATURES,
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


@st.cache_resource  
def load_explainer():
    return CrashExplainer()


@st.cache_data
def get_predictions(_predictor, features):
    return _predictor.predict_batch(features)


# =============================================================================
# Visualization Functions
# =============================================================================

def create_probability_timeline(dates, predictions, selected_range=None):
    """Create an interactive probability timeline with zones."""
    df_plot = pd.DataFrame({
        'Date': dates['Date'],
        'Probability': predictions['crash_probability'] * 100,
        'Risk': predictions['risk_level'],
    })
    
    if selected_range:
        mask = (df_plot['Date'] >= selected_range[0]) & (df_plot['Date'] <= selected_range[1])
        df_plot = df_plot[mask]
    
    fig = go.Figure()
    
    # Add risk zone backgrounds
    fig.add_hrect(y0=0, y1=25, fillcolor="rgba(34, 197, 94, 0.15)", 
                  line_width=0, annotation_text="Low Risk", annotation_position="top left")
    fig.add_hrect(y0=25, y1=50, fillcolor="rgba(234, 179, 8, 0.15)", 
                  line_width=0, annotation_text="Medium Risk", annotation_position="top left")
    fig.add_hrect(y0=50, y1=75, fillcolor="rgba(249, 115, 22, 0.15)", 
                  line_width=0, annotation_text="High Risk", annotation_position="top left")
    fig.add_hrect(y0=75, y1=100, fillcolor="rgba(239, 68, 68, 0.15)", 
                  line_width=0, annotation_text="Critical", annotation_position="top left")
    
    # Main probability line
    fig.add_trace(go.Scatter(
        x=df_plot['Date'],
        y=df_plot['Probability'],
        mode='lines',
        name='Crash Probability',
        line=dict(color='#8b5cf6', width=2),
        fill='tozeroy',
        fillcolor='rgba(139, 92, 246, 0.2)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Probability: %{y:.1f}%<extra></extra>',
    ))
    
    # Add moving average
    if len(df_plot) > 10:
        df_plot['MA10'] = df_plot['Probability'].rolling(10).mean()
        fig.add_trace(go.Scatter(
            x=df_plot['Date'],
            y=df_plot['MA10'],
            mode='lines',
            name='10-period MA',
            line=dict(color='#f59e0b', width=2, dash='dot'),
            hovertemplate='MA: %{y:.1f}%<extra></extra>',
        ))
    
    fig.update_layout(
        title='Crash Probability Over Time',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#fafafa',
        height=450,
        margin=dict(l=60, r=20, t=60, b=40),
        xaxis=dict(title='Date', gridcolor='#333', showgrid=True),
        yaxis=dict(title='Crash Probability (%)', gridcolor='#333', range=[0, 100]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
    )
    
    return fig


def create_risk_distribution_chart(predictions):
    """Create a pie chart showing risk distribution."""
    risk_counts = predictions['risk_level'].value_counts()
    
    colors = {
        'low': '#22c55e',
        'medium': '#eab308',
        'high': '#f97316',
        'critical': '#ef4444',
    }
    
    labels = []
    values = []
    chart_colors = []
    
    for level in ['low', 'medium', 'high', 'critical']:
        count = risk_counts.get(level, 0)
        labels.append(level.capitalize())
        values.append(count)
        chart_colors.append(colors[level])
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=chart_colors,
        textinfo='percent+label',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>',
    )])
    
    fig.update_layout(
        title='Risk Level Distribution',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#fafafa',
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    
    return fig


def create_feature_category_importance(explainer, features_sample):
    """Create category-level importance chart."""
    global_imp = explainer.get_global_importance(features_sample)
    
    # Aggregate by category
    category_imp = global_imp.groupby('category')['mean_abs_shap'].sum().sort_values(ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=category_imp.values,
        y=category_imp.index,
        orientation='h',
        marker_color='#6366f1',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
    ))
    
    fig.update_layout(
        title='Feature Category Importance (SHAP)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#fafafa',
        height=400,
        margin=dict(l=150, r=20, t=60, b=40),
        xaxis=dict(title='Mean |SHAP Value|', gridcolor='#333'),
        yaxis=dict(gridcolor='#333'),
    )
    
    return fig


def create_top_features_chart(explainer, features_sample, n_features=15):
    """Create detailed top features importance chart."""
    global_imp = explainer.get_global_importance(features_sample)
    top_features = global_imp.head(n_features).sort_values('mean_abs_shap', ascending=True)
    
    # Color by category
    category_colors = {
        'Commodities & Currencies': '#f59e0b',
        'Volatility': '#ef4444',
        'US Rates': '#3b82f6',
        'European Rates': '#8b5cf6',
        'Italian Bonds': '#ec4899',
        'Japanese Bonds': '#14b8a6',
        'UK Bonds': '#84cc16',
        'Bond Indices': '#06b6d4',
        'Equity Indices': '#22c55e',
        'Futures': '#f97316',
        'Unknown': '#6b7280',
    }
    
    colors = [category_colors.get(cat, '#6b7280') for cat in top_features['category']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_features['mean_abs_shap'],
        y=top_features['feature'],
        orientation='h',
        marker_color=colors,
        hovertemplate='<b>%{y}</b><br>Category: %{customdata}<br>Importance: %{x:.4f}<extra></extra>',
        customdata=top_features['category'],
    ))
    
    fig.update_layout(
        title=f'Top {n_features} Most Important Features',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#fafafa',
        height=500,
        margin=dict(l=150, r=20, t=60, b=40),
        xaxis=dict(title='Mean |SHAP Value|', gridcolor='#333'),
        yaxis=dict(gridcolor='#333'),
    )
    
    return fig


def create_probability_histogram(predictions):
    """Create histogram of crash probabilities."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=predictions['crash_probability'] * 100,
        nbinsx=30,
        marker_color='#6366f1',
        opacity=0.7,
        hovertemplate='Probability: %{x:.0f}%<br>Count: %{y}<extra></extra>',
    ))
    
    # Add vertical lines for thresholds
    for threshold, color, name in [(25, '#22c55e', 'Low'), (50, '#eab308', 'Medium'), (75, '#f97316', 'High')]:
        fig.add_vline(x=threshold, line_dash="dash", line_color=color, 
                      annotation_text=name, annotation_position="top")
    
    fig.update_layout(
        title='Distribution of Crash Probabilities',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#fafafa',
        height=350,
        margin=dict(l=60, r=20, t=60, b=40),
        xaxis=dict(title='Crash Probability (%)', gridcolor='#333', range=[0, 100]),
        yaxis=dict(title='Frequency', gridcolor='#333'),
        bargap=0.05,
    )
    
    return fig


def create_rolling_statistics(dates, predictions, window=20):
    """Create rolling statistics chart."""
    df_plot = pd.DataFrame({
        'Date': dates['Date'],
        'Probability': predictions['crash_probability'] * 100,
    })
    
    df_plot['Rolling_Mean'] = df_plot['Probability'].rolling(window).mean()
    df_plot['Rolling_Std'] = df_plot['Probability'].rolling(window).std()
    df_plot['Upper'] = df_plot['Rolling_Mean'] + 2 * df_plot['Rolling_Std']
    df_plot['Lower'] = df_plot['Rolling_Mean'] - 2 * df_plot['Rolling_Std']
    df_plot['Lower'] = df_plot['Lower'].clip(lower=0)
    df_plot['Upper'] = df_plot['Upper'].clip(upper=100)
    
    fig = go.Figure()
    
    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([df_plot['Date'], df_plot['Date'][::-1]]),
        y=pd.concat([df_plot['Upper'], df_plot['Lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.2)',
        line_color='rgba(0,0,0,0)',
        name='2σ Band',
        hoverinfo='skip',
    ))
    
    # Rolling mean
    fig.add_trace(go.Scatter(
        x=df_plot['Date'],
        y=df_plot['Rolling_Mean'],
        mode='lines',
        name=f'{window}-Period Mean',
        line=dict(color='#6366f1', width=2),
        hovertemplate='Mean: %{y:.1f}%<extra></extra>',
    ))
    
    # Actual probability (thin line)
    fig.add_trace(go.Scatter(
        x=df_plot['Date'],
        y=df_plot['Probability'],
        mode='lines',
        name='Actual',
        line=dict(color='#94a3b8', width=1),
        opacity=0.5,
        hovertemplate='Actual: %{y:.1f}%<extra></extra>',
    ))
    
    fig.update_layout(
        title=f'Rolling Statistics ({window}-Period Window)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#fafafa',
        height=400,
        margin=dict(l=60, r=20, t=60, b=40),
        xaxis=dict(title='Date', gridcolor='#333'),
        yaxis=dict(title='Crash Probability (%)', gridcolor='#333', range=[0, 100]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
    )
    
    return fig


# =============================================================================
# Main Page
# =============================================================================

def main():
    st.title("📊 Detailed Analysis")
    st.markdown("In-depth analysis of market crash probabilities and feature contributions")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df, dates, features = load_data()
        predictor = load_predictor()
        predictions = get_predictions(predictor, features)
    
    # Date range selector
    st.subheader("📅 Date Range Selection")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    min_date = dates['Date'].min().to_pydatetime()
    max_date = dates['Date'].max().to_pydatetime()
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )
    
    with col3:
        st.markdown("")
        st.markdown("")
        if st.button("Reset to Full Range"):
            start_date = min_date
            end_date = max_date
    
    # Convert to datetime
    start_datetime = pd.Timestamp(start_date)
    end_datetime = pd.Timestamp(end_date)
    
    # Filter data
    mask = (dates['Date'] >= start_datetime) & (dates['Date'] <= end_datetime)
    filtered_dates = dates[mask]
    filtered_predictions = predictions[mask]
    filtered_features = features[mask]
    
    st.caption(f"Showing {len(filtered_dates)} data points from {start_date} to {end_date}")
    
    st.markdown("---")
    
    # Main timeline chart
    st.subheader("📈 Crash Probability Timeline")
    fig_timeline = create_probability_timeline(filtered_dates, filtered_predictions)
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Statistics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    probs = filtered_predictions['crash_probability']
    with col1:
        st.metric("Mean", f"{probs.mean():.1%}")
    with col2:
        st.metric("Median", f"{probs.median():.1%}")
    with col3:
        st.metric("Std Dev", f"{probs.std():.1%}")
    with col4:
        st.metric("Min", f"{probs.min():.1%}")
    with col5:
        st.metric("Max", f"{probs.max():.1%}")
    
    st.markdown("---")
    
    # Two-column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Risk Distribution")
        fig_pie = create_risk_distribution_chart(filtered_predictions)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📊 Probability Distribution")
        fig_hist = create_probability_histogram(filtered_predictions)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # Rolling statistics
    st.subheader("📉 Rolling Statistics")
    window = st.slider("Rolling Window (periods)", min_value=5, max_value=50, value=20)
    fig_rolling = create_rolling_statistics(filtered_dates, filtered_predictions, window=window)
    st.plotly_chart(fig_rolling, use_container_width=True)
    
    st.markdown("---")
    
    # Feature importance section
    st.subheader("🔬 Feature Importance Analysis")
    
    with st.spinner("Computing SHAP values (this may take a moment)..."):
        try:
            explainer = load_explainer()
            # Use a sample for speed
            sample_size = min(200, len(filtered_features))
            sample_features = filtered_features.sample(sample_size, random_state=42)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_category = create_feature_category_importance(explainer, sample_features)
                st.plotly_chart(fig_category, use_container_width=True)
            
            with col2:
                n_features = st.slider("Number of features to show", min_value=5, max_value=30, value=15)
                fig_features = create_top_features_chart(explainer, sample_features, n_features)
                st.plotly_chart(fig_features, use_container_width=True)
                
        except Exception as e:
            st.warning(f"Could not compute SHAP analysis: {str(e)}")
    
    # Data export
    st.markdown("---")
    st.subheader("💾 Export Data")
    
    export_df = pd.DataFrame({
        'Date': filtered_dates['Date'],
        'Crash_Probability': filtered_predictions['crash_probability'],
        'Risk_Level': filtered_predictions['risk_level'],
    })
    
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv,
        file_name=f"crash_predictions_{start_date}_to_{end_date}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

