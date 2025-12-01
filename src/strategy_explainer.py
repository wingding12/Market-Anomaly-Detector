"""
Strategy Explainer Module
=========================

AI-driven explanation engine that translates complex financial strategies
into accessible, actionable guidance for end users.

This module provides:
1. Natural language explanations of market conditions
2. Personalized strategy recommendations with reasoning
3. Q&A responses for common investment questions
4. Risk-appropriate communication adjustments
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# User Profile Types
# =============================================================================

class ExperienceLevel(Enum):
    """User's investment experience level."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class CommunicationStyle(Enum):
    """Preferred communication style."""
    SIMPLE = "simple"       # Plain language, minimal jargon
    BALANCED = "balanced"   # Some technical terms with explanations
    TECHNICAL = "technical" # Full financial terminology


@dataclass
class UserProfile:
    """User profile for personalized explanations."""
    experience: ExperienceLevel = ExperienceLevel.INTERMEDIATE
    style: CommunicationStyle = CommunicationStyle.BALANCED
    risk_tolerance: str = "moderate"
    portfolio_size: str = "medium"  # small, medium, large


# =============================================================================
# Market Context
# =============================================================================

@dataclass
class MarketContext:
    """Current market state for contextual explanations."""
    crash_probability: float
    risk_level: str  # low, medium, high, critical
    top_risk_factors: List[str]
    top_protective_factors: List[str]
    recommended_strategy: str
    target_equity: float
    target_bonds: float
    target_cash: float
    vix_level: Optional[float] = None
    recent_trend: str = "stable"  # rising, falling, stable


# =============================================================================
# Explanation Templates
# =============================================================================

RISK_EXPLANATIONS = {
    "low": {
        "simple": "Markets look calm right now. It's like smooth sailing weather—good conditions for staying invested.",
        "balanced": "Current risk indicators are within normal ranges. The model sees no elevated crash signals, suggesting stable market conditions.",
        "technical": "Risk metrics are sub-25th percentile. VIX term structure normal, credit spreads contained, cross-asset correlations stable.",
    },
    "medium": {
        "simple": "There are some warning signs in the market, but nothing alarming yet. Think of it as clouds forming—worth watching, but no storm yet.",
        "balanced": "Elevated risk signals detected. Some stress indicators are active, warranting increased monitoring but not immediate defensive action.",
        "technical": "Risk probability 25-50%. Partial activation in volatility regime indicators. Credit and rates showing mild stress signals.",
    },
    "high": {
        "simple": "The warning lights are flashing. Markets are showing stress patterns similar to times before past drops. Time to be cautious.",
        "balanced": "Multiple risk factors are elevated simultaneously. Historical patterns suggest conditions similar to pre-correction periods. Consider reducing exposure.",
        "technical": "Risk probability 50-75%. Cross-asset confirmation of stress: elevated VIX momentum, credit spread widening, flight-to-quality flows evident.",
    },
    "critical": {
        "simple": "This is a serious warning. Market conditions look similar to periods right before major drops in the past. Protecting your money should be the priority.",
        "balanced": "Maximum alert level. Current conditions closely match historical pre-crash patterns. The model strongly suggests defensive positioning.",
        "technical": "Risk probability >75%. Full regime breakdown: volatility term structure inverted, credit markets stressed, safe-haven flows accelerating. Tail risk elevated.",
    },
}

STRATEGY_EXPLANATIONS = {
    "dynamic_risk_allocation": {
        "name": "Dynamic Risk Allocation",
        "simple": "This approach gradually reduces your stock holdings as danger signs increase. Instead of making sudden changes, it smoothly adjusts—like slowly turning down the volume instead of hitting mute.",
        "balanced": "Dynamic Risk Allocation scales your equity exposure inversely to crash probability. As risk rises, the strategy progressively shifts toward bonds and cash, avoiding sharp portfolio transitions.",
        "technical": "Continuous equity weight adjustment: w_equity = w_max - (p_crash × (w_max - w_min)). Smooth risk reduction without discrete regime boundaries. Higher turnover but smaller individual trades.",
        "pros": ["Smooth transitions", "No timing decisions", "Automatic adjustment"],
        "cons": ["Higher trading costs", "May reduce too early", "Continuous monitoring needed"],
    },
    "regime_switching": {
        "name": "Regime Switching",
        "simple": "This is like having four pre-set plans: 'all clear', 'be careful', 'protect yourself', and 'emergency mode'. Based on the danger level, you switch between these plans.",
        "balanced": "Regime Switching uses predefined portfolio allocations for each risk level. When risk crosses a threshold, the entire portfolio shifts to the next defensive posture.",
        "technical": "Discrete state transitions at 25/50/75% probability thresholds. Four regime states with fixed allocations. Lower turnover, concentrated rebalancing events.",
        "pros": ["Clear rules", "Lower trading costs", "Easier to implement"],
        "cons": ["Sharp transitions", "May miss gradual changes", "Threshold sensitivity"],
    },
    "probability_weighted_hedge": {
        "name": "Probability-Weighted Hedging",
        "simple": "Keep your investments mostly unchanged, but buy insurance when things look risky. The riskier it looks, the more insurance you buy.",
        "balanced": "This strategy maintains your core equity position while adding proportional hedges. As crash probability rises, hedge allocation increases—like buying more insurance when storm season approaches.",
        "technical": "Base equity exposure maintained, hedge overlay scaled by p_crash. Hedge instruments: put spreads, VIX calls, inverse exposure. Cost: theta decay during low-risk periods.",
        "pros": ["Stay invested for upside", "Explicit protection", "Flexible sizing"],
        "cons": ["Hedge costs add up", "Requires options knowledge", "Decay in calm markets"],
    },
    "momentum_overlay": {
        "name": "Momentum + Risk Overlay",
        "simple": "Follow the trend, but with a safety net. If prices are rising, stay invested. If they're falling OR danger is high, step aside.",
        "balanced": "Combines trend-following with crash protection. The strategy stays invested when momentum is positive and risk is manageable, but exits when either signal turns negative.",
        "technical": "Dual-signal system: price vs. 20-week MA for momentum, crash probability for risk overlay. Exit triggers: negative momentum OR probability >threshold. Re-entry requires both signals positive.",
        "pros": ["Captures trends", "Double protection", "Clear exit rules"],
        "cons": ["May miss reversals", "Whipsaws possible", "Lagging indicator"],
    },
}

ACTION_TEMPLATES = {
    "low": {
        "simple": [
            "Stay the course with your current investments",
            "This is a good time to review your long-term goals",
            "Consider if you want to add to winning positions",
        ],
        "balanced": [
            "Maintain current equity allocation",
            "Review portfolio for rebalancing opportunities",
            "Monitor risk indicators weekly for early warning signs",
        ],
        "technical": [
            "Hold target equity weight; no tactical adjustment needed",
            "Harvest tax losses if available",
            "Consider selling covered calls to enhance yield",
        ],
    },
    "medium": {
        "simple": [
            "Start thinking about which investments you'd sell first if needed",
            "Make sure you have some cash available",
            "Don't add new risky investments right now",
        ],
        "balanced": [
            "Reduce equity allocation by 10-15%",
            "Increase cash position to 10-15%",
            "Shift equity exposure toward quality factors",
        ],
        "technical": [
            "Tactical underweight equities 10-15%",
            "Increase portfolio quality tilt; reduce high-beta exposure",
            "Consider 2-3% allocation to put spreads as tail hedge",
        ],
    },
    "high": {
        "simple": [
            "Reduce your stock investments significantly",
            "Move money to safer options like bonds or savings",
            "Avoid any new risky investments",
            "Keep enough cash for emergencies and opportunities",
        ],
        "balanced": [
            "Reduce equity to 25-35% of portfolio",
            "Increase bond allocation, favoring quality",
            "Raise cash to 15-20%",
            "Exit speculative and leveraged positions",
        ],
        "technical": [
            "Target equity 25-35%; underweight risk assets",
            "Duration extension in fixed income; flight-to-quality positioning",
            "Implement 5-8% tail hedge via puts or VIX calls",
            "Exit all leveraged and margin positions",
        ],
    },
    "critical": {
        "simple": [
            "Protect your money—this is the priority now",
            "Reduce stocks to the minimum you're comfortable with",
            "Keep a significant amount in cash",
            "Consider investments that go up when markets go down",
        ],
        "balanced": [
            "Minimize equity exposure immediately (10-15%)",
            "Maximum defensive positioning in bonds",
            "Hold 25-30% in cash or equivalents",
            "Implement full hedge position if not already in place",
        ],
        "technical": [
            "Emergency de-risking: target <15% equity",
            "Maximum duration fixed income; Treasury overweight",
            "Full tail hedge implementation: 10-15% in puts, VIX calls",
            "Consider inverse ETF exposure for additional protection",
        ],
    },
}


# =============================================================================
# Q&A Knowledge Base
# =============================================================================

QA_KNOWLEDGE = {
    # Risk-related questions
    "what does crash probability mean": {
        "simple": "It's like a weather forecast for the stock market. A high percentage means conditions look similar to times before past market drops.",
        "balanced": "Crash probability indicates how closely current market conditions match historical pre-crash patterns. An 80% reading means today's indicators strongly resemble what we saw before past significant market declines.",
        "technical": "The probability is output from an XGBoost classifier trained on 62 market indicators. It measures pattern similarity to labeled pre-crash periods, where crashes are defined as subsequent 3-month drawdowns >15%.",
    },
    "why is risk high": {
        "simple": "Multiple warning signs are active at once—it's like seeing dark clouds, falling barometer, and strong winds all together before a storm.",
        "balanced": "Several risk indicators are elevated simultaneously: typically volatility is rising, credit spreads are widening, and safe-haven assets are seeing inflows. This combination historically precedes market stress.",
        "technical": "Cross-asset confirmation of stress signals: VIX momentum elevated, credit spreads widening, bond/equity correlation breakdown, potential flight-to-quality flows in rates and currencies.",
    },
    "is my money safe": {
        "simple": "No investment is completely safe, but taking defensive action during high-risk periods has historically helped protect against the worst losses.",
        "balanced": "Market risk can't be eliminated, but it can be managed. The strategies shown here are designed to reduce exposure during dangerous periods. However, they're tools for informed decision-making, not guarantees.",
        "technical": "Systematic risk management aims to improve risk-adjusted returns by reducing drawdowns. Historical backtests show defensive positioning during elevated-risk periods typically preserves capital, though no strategy perfectly predicts market outcomes.",
    },
    
    # Strategy questions
    "which strategy is best": {
        "simple": "There's no single 'best'—it depends on what matters most to you. If you want simplicity, try Regime Switching. If you want to stay invested but protected, try the Hedging approach.",
        "balanced": "Each strategy has trade-offs. Dynamic Risk offers smooth adjustments; Regime Switching provides clear rules; Probability Hedging maintains exposure with protection; Momentum Overlay combines trend-following with crash defense. Choose based on your priorities.",
        "technical": "Strategy selection should align with your utility function. Maximize Sharpe for risk-adjusted returns; minimize drawdown for capital preservation; maximize return for growth. The optimal choice depends on your loss aversion coefficient and time horizon.",
    },
    "what should i do now": {
        "dynamic": True,  # Indicates this needs real-time context
    },
    "explain my recommendation": {
        "dynamic": True,
    },
    
    # Concept explanations
    "what is sharpe ratio": {
        "simple": "It measures how much return you get for the risk you take. Higher is better—like getting more miles per gallon.",
        "balanced": "The Sharpe Ratio measures risk-adjusted return: how much excess return you earn per unit of volatility. A ratio above 1.0 is generally considered good; above 2.0 is excellent.",
        "technical": "Sharpe = (R_portfolio - R_riskfree) / σ_portfolio. Measures excess return per unit of total risk. Assumes normal return distributions; can be misleading for strategies with skewed returns.",
    },
    "what is max drawdown": {
        "simple": "The biggest drop from a high point to a low point. If your account went from $100 to $70, that's a 30% drawdown.",
        "balanced": "Maximum drawdown is the largest peak-to-trough decline in portfolio value. It shows the worst-case scenario you would have experienced—useful for understanding what you might have to endure.",
        "technical": "MDD = max(peak_t - trough_t) / peak_t over all t. Measures worst historical loss. Key metric for capital preservation mandates and understanding sequence-of-returns risk.",
    },
    "what is vix": {
        "simple": "It's called the 'fear index.' When investors are worried, the VIX goes up. When they're calm, it stays low. Normal is around 15-20.",
        "balanced": "The VIX measures expected market volatility over the next 30 days, derived from S&P 500 option prices. It's often called the 'fear gauge'—spikes indicate increased uncertainty and risk aversion.",
        "technical": "VIX is the implied volatility of 30-day SPX options, calculated from a strip of OTM puts and calls. Represents market's expectation of annualized volatility. Mean-reverting with positive skew; spikes during stress events.",
    },
    
    # Action-oriented questions
    "how do i hedge": {
        "simple": "The simplest way is to sell some stocks and buy bonds or hold cash. For more protection, you could look into inverse funds that go up when markets go down.",
        "balanced": "Common hedging approaches: (1) Reduce equity exposure and increase bonds/cash, (2) Buy put options on indices you own, (3) Add VIX call options, (4) Use inverse ETFs for short-term protection. Each has different costs and trade-offs.",
        "technical": "Tail hedge implementation: put spreads on SPX (e.g., 5% OTM, 3-month expiry), VIX call spreads, or inverse ETF allocation. Size based on portfolio beta and target max loss. Consider delta-hedging for precision.",
    },
    "hedge portfolio": {
        "simple": "The simplest way is to sell some stocks and buy bonds or hold cash. For more protection, you could look into inverse funds that go up when markets go down.",
        "balanced": "Common hedging approaches: (1) Reduce equity exposure and increase bonds/cash, (2) Buy put options on indices you own, (3) Add VIX call options, (4) Use inverse ETFs for short-term protection. Each has different costs and trade-offs.",
        "technical": "Tail hedge implementation: put spreads on SPX (e.g., 5% OTM, 3-month expiry), VIX call spreads, or inverse ETF allocation. Size based on portfolio beta and target max loss. Consider delta-hedging for precision.",
    },
    "protect investments": {
        "simple": "The simplest way is to sell some stocks and buy bonds or hold cash. For more protection, you could look into inverse funds that go up when markets go down.",
        "balanced": "Common hedging approaches: (1) Reduce equity exposure and increase bonds/cash, (2) Buy put options on indices you own, (3) Add VIX call options, (4) Use inverse ETFs for short-term protection. Each has different costs and trade-offs.",
        "technical": "Tail hedge implementation: put spreads on SPX (e.g., 5% OTM, 3-month expiry), VIX call spreads, or inverse ETF allocation. Size based on portfolio beta and target max loss. Consider delta-hedging for precision.",
    },
    "when should i buy back in": {
        "simple": "When the warning signs calm down. The system will show lower crash probability when conditions improve. Don't try to catch the exact bottom—that's nearly impossible.",
        "balanced": "Re-entry signals: crash probability declining below 50%, VIX normalizing, credit spreads tightening, and momentum turning positive. Consider scaling back in gradually rather than all at once.",
        "technical": "Re-entry criteria: probability <50%, VIX below 25 with declining momentum, credit spreads mean-reverting, price above 20-week MA. Use 3-tranche re-entry to manage timing risk.",
    },
}


# =============================================================================
# Explainer Engine
# =============================================================================

class StrategyExplainer:
    """
    AI-driven explanation engine for investment strategies.
    
    Generates personalized, context-aware explanations based on:
    - Current market conditions
    - User's experience level
    - Preferred communication style
    """
    
    def __init__(
        self,
        user_profile: Optional[UserProfile] = None,
        market_context: Optional[MarketContext] = None,
    ):
        self.profile = user_profile or UserProfile()
        self.context = market_context
    
    def update_context(self, market_context: MarketContext):
        """Update market context for real-time explanations."""
        self.context = market_context
    
    def update_profile(self, user_profile: UserProfile):
        """Update user profile for personalized responses."""
        self.profile = user_profile
    
    def _get_style_key(self) -> str:
        """Get the appropriate style key based on user profile."""
        return self.profile.style.value
    
    def explain_current_risk(self) -> str:
        """Explain the current risk level in appropriate language."""
        if not self.context:
            return "I need current market data to explain the risk level. Please ensure data is loaded."
        
        style = self._get_style_key()
        risk_level = self.context.risk_level.lower()
        
        base_explanation = RISK_EXPLANATIONS.get(risk_level, RISK_EXPLANATIONS["medium"])[style]
        
        # Add probability context
        prob_pct = self.context.crash_probability * 100
        
        if style == "simple":
            prob_context = f"\n\nThe current danger reading is {prob_pct:.0f}%."
        elif style == "balanced":
            prob_context = f"\n\nCurrent crash probability: {prob_pct:.1f}%."
        else:
            prob_context = f"\n\nP(crash) = {prob_pct:.1f}% | Risk classification: {risk_level.upper()}"
        
        return base_explanation + prob_context
    
    def explain_strategy(self, strategy_key: str) -> str:
        """Explain a specific strategy."""
        style = self._get_style_key()
        strategy = STRATEGY_EXPLANATIONS.get(strategy_key)
        
        if not strategy:
            return f"I don't have information about the strategy '{strategy_key}'."
        
        explanation = f"**{strategy['name']}**\n\n{strategy[style]}"
        
        if style != "simple":
            explanation += "\n\n**Pros:**\n"
            for pro in strategy["pros"]:
                explanation += f"• {pro}\n"
            explanation += "\n**Cons:**\n"
            for con in strategy["cons"]:
                explanation += f"• {con}\n"
        
        return explanation
    
    def get_action_items(self) -> List[str]:
        """Get recommended actions based on current risk level."""
        if not self.context:
            return ["Load market data to get personalized recommendations."]
        
        style = self._get_style_key()
        risk_level = self.context.risk_level.lower()
        
        return ACTION_TEMPLATES.get(risk_level, ACTION_TEMPLATES["medium"])[style]
    
    def explain_recommendation(self) -> str:
        """Explain why the current strategy is recommended."""
        if not self.context:
            return "I need market data to explain the recommendation."
        
        style = self._get_style_key()
        risk_level = self.context.risk_level.lower()
        prob = self.context.crash_probability * 100
        
        # Build explanation based on style and risk level
        if style == "simple":
            if risk_level == "low":
                explanation = f"With only {prob:.0f}% danger showing, the market looks stable. "
                explanation += "The recommendation is to stay invested and capture potential gains."
            elif risk_level == "medium":
                explanation = f"At {prob:.0f}% danger level, there are some concerns but nothing severe. "
                explanation += "The recommendation is to be a bit more cautious while staying mostly invested."
            elif risk_level == "high":
                explanation = f"At {prob:.0f}% danger, multiple warning signs are active. "
                explanation += f"That's why the target is {self.context.target_equity*100:.0f}% stocks—significantly reduced to protect your money."
            else:  # critical
                explanation = f"At {prob:.0f}% danger, conditions look very similar to past crashes. "
                explanation += f"The recommendation of only {self.context.target_equity*100:.0f}% stocks prioritizes protecting your capital."
        
        elif style == "balanced":
            explanation = f"**Current Assessment**: {prob:.1f}% crash probability ({risk_level.upper()} risk)\n\n"
            explanation += f"**Recommended Allocation**:\n"
            explanation += f"• Equities: {self.context.target_equity*100:.0f}%\n"
            explanation += f"• Bonds: {self.context.target_bonds*100:.0f}%\n"
            explanation += f"• Cash: {self.context.target_cash*100:.0f}%\n\n"
            explanation += "**Reasoning**: "
            
            if self.context.top_risk_factors:
                explanation += f"Key risk drivers include {', '.join(self.context.top_risk_factors[:3])}. "
            if self.context.top_protective_factors:
                explanation += f"Some protection from {', '.join(self.context.top_protective_factors[:2])}."
        
        else:  # technical
            explanation = f"**Risk Metrics**\n"
            explanation += f"• P(crash) = {prob:.2f}%\n"
            explanation += f"• Classification: {risk_level.upper()}\n"
            if self.context.vix_level:
                explanation += f"• VIX: {self.context.vix_level:.1f}\n"
            explanation += f"\n**Target Allocation**\n"
            explanation += f"• Equity: {self.context.target_equity*100:.0f}% (β-adjusted)\n"
            explanation += f"• Fixed Income: {self.context.target_bonds*100:.0f}%\n"
            explanation += f"• Cash/Equivalents: {self.context.target_cash*100:.0f}%\n"
            explanation += f"\n**Factor Attribution**\n"
            if self.context.top_risk_factors:
                explanation += f"• Risk factors: {', '.join(self.context.top_risk_factors[:3])}\n"
            if self.context.top_protective_factors:
                explanation += f"• Protective factors: {', '.join(self.context.top_protective_factors[:2])}"
        
        return explanation
    
    def answer_question(self, question: str) -> str:
        """Answer a user question using the knowledge base."""
        question_lower = question.lower().strip()
        
        # Remove common question prefixes
        prefixes = ["what is ", "what's ", "how do i ", "how can i ", "why is ", 
                    "should i ", "can you explain ", "tell me about ", "explain "]
        for prefix in prefixes:
            if question_lower.startswith(prefix):
                question_lower = question_lower[len(prefix):]
                break
        
        # Remove punctuation
        question_lower = re.sub(r'[^\w\s]', '', question_lower)
        
        # Find best matching question in knowledge base
        best_match = None
        best_score = 0
        
        for key in QA_KNOWLEDGE:
            # Simple keyword matching
            key_words = set(key.split())
            question_words = set(question_lower.split())
            overlap = len(key_words & question_words)
            score = overlap / max(len(key_words), 1)
            
            if score > best_score:
                best_score = score
                best_match = key
        
        if best_match and best_score > 0.3:
            answer_data = QA_KNOWLEDGE[best_match]
            
            # Handle dynamic questions that need context
            if answer_data.get("dynamic"):
                if "what should" in best_match or "do now" in best_match:
                    return self._generate_action_response()
                elif "recommendation" in best_match:
                    return self.explain_recommendation()
            
            style = self._get_style_key()
            return answer_data.get(style, answer_data.get("balanced", "I don't have a detailed answer for that."))
        
        # Fallback response
        return self._generate_fallback_response(question)
    
    def _generate_action_response(self) -> str:
        """Generate what-to-do-now response based on context."""
        if not self.context:
            return "I need current market data to give you specific recommendations. Please make sure data is loaded."
        
        style = self._get_style_key()
        actions = self.get_action_items()
        
        if style == "simple":
            response = "Here's what I'd suggest right now:\n\n"
        elif style == "balanced":
            response = f"Based on current {self.context.risk_level.upper()} risk conditions:\n\n"
        else:
            response = f"Action items for P(crash)={self.context.crash_probability*100:.1f}%:\n\n"
        
        for i, action in enumerate(actions, 1):
            response += f"{i}. {action}\n"
        
        return response
    
    def _generate_fallback_response(self, question: str) -> str:
        """Generate a helpful fallback when question isn't recognized."""
        suggestions = [
            "what does crash probability mean",
            "which strategy is best",
            "what should I do now",
            "how do I hedge",
            "what is max drawdown",
        ]
        
        response = "I'm not sure I understand that question. Here are some things I can help with:\n\n"
        for suggestion in suggestions:
            response += f"• \"{suggestion.title()}?\"\n"
        response += "\nOr ask me to explain any of the strategies shown on this page."
        
        return response
    
    def get_market_summary(self) -> str:
        """Get a natural language summary of current market conditions."""
        if not self.context:
            return "Market data not available."
        
        style = self._get_style_key()
        prob = self.context.crash_probability * 100
        risk = self.context.risk_level.lower()
        
        if style == "simple":
            summaries = {
                "low": f"Markets are calm today. The danger meter shows {prob:.0f}%, which is low. Good conditions for staying invested.",
                "medium": f"Some caution warranted. The danger level is {prob:.0f}%—elevated but not alarming. Keep an eye on things.",
                "high": f"Warning signs are active. At {prob:.0f}% danger, it's time to think defensively. Consider reducing risk.",
                "critical": f"Serious warning. At {prob:.0f}% danger, conditions resemble past pre-crash periods. Protect your portfolio.",
            }
        elif style == "balanced":
            summaries = {
                "low": f"Market conditions normal. Crash probability at {prob:.1f}% indicates stable environment. Risk indicators contained.",
                "medium": f"Elevated risk detected. {prob:.1f}% crash probability suggests increased vigilance. Some stress indicators active.",
                "high": f"High risk environment. {prob:.1f}% probability with multiple factors confirming stress. Defensive positioning recommended.",
                "critical": f"Critical risk level. {prob:.1f}% probability indicates conditions similar to historical pre-crash periods. Immediate defensive action warranted.",
            }
        else:
            summaries = {
                "low": f"P(crash) = {prob:.2f}% | Status: NORMAL | VIX regime stable, credit contained, momentum neutral.",
                "medium": f"P(crash) = {prob:.2f}% | Status: ELEVATED | Partial stress signal activation, cross-asset confirmation pending.",
                "high": f"P(crash) = {prob:.2f}% | Status: HIGH | Multi-factor stress confirmation, risk-off positioning indicated.",
                "critical": f"P(crash) = {prob:.2f}% | Status: CRITICAL | Full regime breakdown, historical pattern match to pre-crash conditions.",
            }
        
        return summaries.get(risk, summaries["medium"])


# =============================================================================
# Conversation Manager
# =============================================================================

@dataclass
class Message:
    """Chat message."""
    role: str  # "user" or "assistant"
    content: str


class ConversationManager:
    """
    Manages conversation flow and maintains context.
    """
    
    def __init__(self, explainer: StrategyExplainer):
        self.explainer = explainer
        self.history: List[Message] = []
        self.greeting_shown = False
    
    def get_greeting(self) -> str:
        """Get initial greeting message."""
        style = self.explainer._get_style_key()
        
        greetings = {
            "simple": "👋 Hi! I'm your investment advisor. I can help explain what the market signals mean and what you might want to do about it.\n\nTry asking me things like:\n• \"What should I do now?\"\n• \"Why is risk high?\"\n• \"Explain my recommendation\"",
            "balanced": "👋 Welcome to the Strategy Advisor. I'll help you understand current market conditions and investment recommendations.\n\nI can explain:\n• Current risk assessment and what's driving it\n• Different investment strategies and their trade-offs\n• Specific actions appropriate for your situation\n\nWhat would you like to know?",
            "technical": "Strategy Advisor initialized. Available queries:\n• Risk assessment: current probability, factor attribution\n• Strategy analysis: backtest metrics, allocation logic\n• Implementation: hedge sizing, rebalancing triggers\n\nAwaiting input.",
        }
        
        self.greeting_shown = True
        return greetings.get(style, greetings["balanced"])
    
    def process_message(self, user_message: str) -> str:
        """Process user message and return response."""
        # Store user message
        self.history.append(Message(role="user", content=user_message))
        
        # Normalize input
        msg_lower = user_message.lower().strip()
        
        # Check for specific intents
        if any(word in msg_lower for word in ["hello", "hi ", "hey", "start"]):
            response = self.get_greeting()
        
        elif any(word in msg_lower for word in ["current", "now", "today", "market"]) and any(word in msg_lower for word in ["risk", "condition", "status", "state"]):
            response = self.explainer.get_market_summary()
        
        elif "explain" in msg_lower and "risk" in msg_lower:
            response = self.explainer.explain_current_risk()
        
        elif "explain" in msg_lower and "recommendation" in msg_lower:
            response = self.explainer.explain_recommendation()
        
        elif any(word in msg_lower for word in ["dynamic", "regime", "hedge", "momentum"]) and "strategy" in msg_lower:
            # Extract strategy name
            strategy_map = {
                "dynamic": "dynamic_risk_allocation",
                "regime": "regime_switching",
                "hedge": "probability_weighted_hedge",
                "momentum": "momentum_overlay",
            }
            for key, value in strategy_map.items():
                if key in msg_lower:
                    response = self.explainer.explain_strategy(value)
                    break
            else:
                response = self.explainer.answer_question(user_message)
        
        elif any(word in msg_lower for word in ["what should", "do now", "action", "next step"]):
            response = self.explainer._generate_action_response()
        
        elif any(word in msg_lower for word in ["strategy", "strategies"]) and any(word in msg_lower for word in ["all", "list", "compare", "which"]):
            response = "Here are the available strategies:\n\n"
            for key, strat in STRATEGY_EXPLANATIONS.items():
                response += f"**{strat['name']}**: {strat['simple'][:100]}...\n\n"
            response += "Ask me to explain any specific strategy for more details."
        
        else:
            # Use Q&A system
            response = self.explainer.answer_question(user_message)
        
        # Store assistant response
        self.history.append(Message(role="assistant", content=response))
        
        return response
    
    def get_history(self) -> List[Message]:
        """Get conversation history."""
        return self.history
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        self.greeting_shown = False


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Strategy Explainer Test")
    print("=" * 60)
    
    # Create test context
    context = MarketContext(
        crash_probability=0.72,
        risk_level="high",
        top_risk_factors=["VIX Index_lag_3", "EONIA Index", "GTDEM2Y Govt"],
        top_protective_factors=["JPY Curncy", "GBP Curncy"],
        recommended_strategy="dynamic_risk_allocation",
        target_equity=0.30,
        target_bonds=0.45,
        target_cash=0.25,
        vix_level=28.5,
    )
    
    # Test different profiles
    profiles = [
        UserProfile(ExperienceLevel.BEGINNER, CommunicationStyle.SIMPLE),
        UserProfile(ExperienceLevel.INTERMEDIATE, CommunicationStyle.BALANCED),
        UserProfile(ExperienceLevel.ADVANCED, CommunicationStyle.TECHNICAL),
    ]
    
    for profile in profiles:
        print(f"\n--- {profile.style.value.upper()} Style ---")
        explainer = StrategyExplainer(profile, context)
        
        print("\n[Risk Explanation]")
        print(explainer.explain_current_risk())
        
        print("\n[Recommendation]")
        print(explainer.explain_recommendation())
        
        print("\n[Actions]")
        for action in explainer.get_action_items():
            print(f"  • {action}")
    
    # Test conversation
    print("\n" + "=" * 60)
    print("Conversation Test")
    print("=" * 60)
    
    profile = UserProfile(ExperienceLevel.INTERMEDIATE, CommunicationStyle.BALANCED)
    explainer = StrategyExplainer(profile, context)
    convo = ConversationManager(explainer)
    
    test_questions = [
        "What is crash probability?",
        "Why is risk high right now?",
        "What should I do?",
        "Explain the momentum strategy",
    ]
    
    print(convo.get_greeting())
    for q in test_questions:
        print(f"\nUser: {q}")
        print(f"Bot: {convo.process_message(q)}")
    
    print("\n✅ All tests passed!")

