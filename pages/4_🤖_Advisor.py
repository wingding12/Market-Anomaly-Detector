"""
AI Strategy Advisor Page
========================

Interactive chatbot that explains investment strategies in accessible language.
Adapts communication style based on user's experience level.
"""

import streamlit as st
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Advisor | Market Anomaly Detector",
    page_icon="🤖",
    layout="wide",
)

from src import (
    load_financial_market_data,
    prepare_features,
    CrashPredictor,
    CrashExplainer,
)

from src.investment_strategies import get_current_recommendation

from src.strategy_explainer import (
    StrategyExplainer,
    ConversationManager,
    MarketContext,
    UserProfile,
    ExperienceLevel,
    CommunicationStyle,
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


@st.cache_resource
def get_explainer_model():
    return CrashExplainer()


@st.cache_data
def get_predictions(_predictor, features):
    return _predictor.predict_batch(features)


@st.cache_data
def get_latest_explanation(_explainer, features):
    return _explainer.explain_single(features.iloc[[-1]])


# =============================================================================
# Session State
# =============================================================================

def init_session_state():
    """Initialize session state for chat."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = UserProfile()


# =============================================================================
# Market Context Builder
# =============================================================================

def build_market_context(
    predictor,
    explainer_model,
    features,
    risk_tolerance: str = "moderate",
) -> MarketContext:
    """Build market context from current data."""
    
    # Get latest prediction
    result = predictor.predict_single(features.iloc[[-1]])
    
    # Get explanation
    explanation = get_latest_explanation(explainer_model, features)
    
    # Get recommendation
    recommendation = get_current_recommendation(result.crash_probability, risk_tolerance)
    
    # Extract feature names from SHAP
    top_risk = [f.feature for f in explanation.top_risk_factors[:5]]
    top_protective = [f.feature for f in explanation.top_protective_factors[:5]]
    
    # Get VIX if available
    vix_level = None
    if "VIX Index" in features.columns:
        vix_level = float(features["VIX Index"].iloc[-1])
    
    return MarketContext(
        crash_probability=result.crash_probability,
        risk_level=result.risk_level.value,
        top_risk_factors=top_risk,
        top_protective_factors=top_protective,
        recommended_strategy=recommendation["stance"],
        target_equity=recommendation["target_weights"]["equity"],
        target_bonds=recommendation["target_weights"]["bonds"],
        target_cash=recommendation["target_weights"]["cash"],
        vix_level=vix_level,
    )


# =============================================================================
# UI Components
# =============================================================================

def render_chat_message(role: str, content: str):
    """Render a chat message with appropriate styling."""
    if role == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)


def render_quick_actions(explainer: StrategyExplainer):
    """Render quick action buttons."""
    st.markdown("##### Quick Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Current Risk", use_container_width=True):
            return "What is the current market risk?"
        if st.button("💡 What Should I Do?", use_container_width=True):
            return "What should I do now?"
        if st.button("📈 Explain Strategies", use_container_width=True):
            return "What strategies are available?"
    
    with col2:
        if st.button("🔍 Why This Risk?", use_container_width=True):
            return "Why is risk at this level?"
        if st.button("🎯 My Recommendation", use_container_width=True):
            return "Explain my recommendation"
        if st.button("🛡️ How to Hedge?", use_container_width=True):
            return "How do I hedge my portfolio?"
    
    return None


def render_market_status(context: MarketContext):
    """Render compact market status indicator."""
    risk_colors = {
        "low": "#22c55e",
        "medium": "#eab308",
        "high": "#f97316",
        "critical": "#ef4444",
    }
    
    color = risk_colors.get(context.risk_level.lower(), "#6b7280")
    prob_pct = context.crash_probability * 100
    
    st.markdown(f"""
    <div style="padding: 0.75rem; background: linear-gradient(135deg, {color}20, {color}10); 
                border-radius: 8px; border-left: 3px solid {color}; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: {color}; font-weight: 600; text-transform: uppercase;">
                {context.risk_level} Risk
            </span>
            <span style="color: #fafafa; font-weight: bold; font-size: 1.2rem;">
                {prob_pct:.0f}%
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_suggested_questions():
    """Render suggested follow-up questions."""
    suggestions = [
        "What is the Sharpe ratio?",
        "Explain the momentum strategy",
        "When should I buy back in?",
        "What is VIX?",
        "Is my money safe?",
    ]
    
    st.markdown("##### 💬 You could also ask:")
    for suggestion in suggestions[:3]:
        st.markdown(f"- *\"{suggestion}\"*")


# =============================================================================
# Main Page
# =============================================================================

def main():
    init_session_state()
    
    # Header
    st.title("🤖 Strategy Advisor")
    st.markdown("Your AI assistant for understanding market conditions and investment strategies")
    
    # Load data
    with st.spinner("Loading market data..."):
        df, dates, features = load_data()
        predictor = get_predictor()
        explainer_model = get_explainer_model()
    
    # Sidebar configuration
    st.sidebar.header("Communication Style")
    
    experience_options = {
        "Beginner (Plain Language)": ExperienceLevel.BEGINNER,
        "Intermediate (Some Technical)": ExperienceLevel.INTERMEDIATE,
        "Advanced (Full Technical)": ExperienceLevel.ADVANCED,
    }
    
    selected_experience = st.sidebar.selectbox(
        "Your Experience Level",
        options=list(experience_options.keys()),
        index=1,
    )
    
    style_map = {
        ExperienceLevel.BEGINNER: CommunicationStyle.SIMPLE,
        ExperienceLevel.INTERMEDIATE: CommunicationStyle.BALANCED,
        ExperienceLevel.ADVANCED: CommunicationStyle.TECHNICAL,
    }
    
    experience_level = experience_options[selected_experience]
    communication_style = style_map[experience_level]
    
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance",
        options=["conservative", "moderate", "aggressive"],
        index=1,
    )
    
    # Update user profile
    st.session_state.user_profile = UserProfile(
        experience=experience_level,
        style=communication_style,
        risk_tolerance=risk_tolerance,
    )
    
    # Build market context
    market_context = build_market_context(
        predictor, explainer_model, features, risk_tolerance
    )
    
    # Create explainer and conversation manager
    explainer = StrategyExplainer(st.session_state.user_profile, market_context)
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_started = False
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Market status in sidebar
    st.sidebar.markdown("### Current Market")
    render_market_status(market_context)
    
    st.sidebar.markdown(f"**Target Allocation:**")
    st.sidebar.markdown(f"- Equity: {market_context.target_equity*100:.0f}%")
    st.sidebar.markdown(f"- Bonds: {market_context.target_bonds*100:.0f}%")
    st.sidebar.markdown(f"- Cash: {market_context.target_cash*100:.0f}%")
    
    if market_context.vix_level:
        st.sidebar.markdown(f"**VIX:** {market_context.vix_level:.1f}")
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat container
        chat_container = st.container()
        
        # Show greeting if first time
        if not st.session_state.conversation_started:
            convo = ConversationManager(explainer)
            greeting = convo.get_greeting()
            st.session_state.messages.append({"role": "assistant", "content": greeting})
            st.session_state.conversation_started = True
        
        # Display chat history
        with chat_container:
            for message in st.session_state.messages:
                render_chat_message(message["role"], message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask me about market conditions or strategies...")
        
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Create fresh conversation manager with current context
            convo = ConversationManager(explainer)
            
            # Get response
            response = convo.process_message(user_input)
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to show new messages
            st.rerun()
    
    with col2:
        st.markdown("### Quick Actions")
        
        # Quick action buttons
        quick_question = render_quick_actions(explainer)
        
        if quick_question:
            st.session_state.messages.append({"role": "user", "content": quick_question})
            convo = ConversationManager(explainer)
            response = convo.process_message(quick_question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        st.markdown("---")
        render_suggested_questions()
        
        # Strategy cards
        st.markdown("---")
        st.markdown("### Strategy Quick Guide")
        
        with st.expander("📈 Dynamic Risk"):
            st.markdown("*Gradually adjust exposure based on danger level*")
        
        with st.expander("🔄 Regime Switch"):
            st.markdown("*Clear rules for each risk level*")
        
        with st.expander("🛡️ Probability Hedge"):
            st.markdown("*Stay invested with proportional protection*")
        
        with st.expander("📊 Momentum Overlay"):
            st.markdown("*Follow trends with crash protection*")
    
    # Footer
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Communication Style:**")
        style_names = {
            CommunicationStyle.SIMPLE: "🟢 Plain Language",
            CommunicationStyle.BALANCED: "🟡 Balanced",
            CommunicationStyle.TECHNICAL: "🔴 Technical",
        }
        st.markdown(style_names[communication_style])
    
    with col2:
        st.markdown("**Context:**")
        st.markdown(f"📊 {len(features)} data points analyzed")
    
    with col3:
        st.markdown("**Latest Data:**")
        st.markdown(f"📅 {dates['Date'].iloc[-1].strftime('%B %d, %Y')}")


if __name__ == "__main__":
    main()

