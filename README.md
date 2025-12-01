# Market Anomaly Detector

A quantitative early warning system that identifies elevated crash risk in global financial markets. Built for portfolio managers, risk analysts, and quantitative researchers who need systematic tools for monitoring market stress conditions.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)

---

## The Problem

Market crashes don't happen in isolation. They're preceded by observable patterns across asset classes—rising volatility, yield curve inversions, credit spread widening, and cross-market correlations breaking down. The challenge is synthesizing these signals in real-time.

Traditional risk metrics like VaR and standard deviation assume normal distributions and fail spectacularly during tail events. This system takes a different approach: supervised learning on historical crash periods to identify the multi-factor conditions that precede market dislocations.

---

## What This System Does

**Crash Probability Scoring**: Outputs a 0-100% probability that current market conditions resemble historical pre-crash environments. The model was trained on data spanning the dot-com bust, the 2008 financial crisis, European sovereign debt concerns, and the COVID-19 selloff.

**Risk Attribution**: SHAP-based decomposition shows exactly which factors are driving the current risk assessment. When the model flags elevated risk, you can see whether it's volatility momentum, rate differentials, or equity market structure causing the signal.

**Strategy Recommendations**: Translates probability scores into actionable portfolio positioning—from maintaining full exposure during benign conditions to implementing defensive hedges when risk is critical.

**Investment Strategy Backtesting**: Four distinct systematic strategies with full performance analytics. Compare approaches like dynamic risk scaling, regime switching, and probability-weighted hedging against traditional buy-and-hold.

**Historical Event Analysis**: Examine how the model would have performed during specific market events. Compare predicted probabilities against actual market outcomes to build confidence in the signals.

---

## The Data Pipeline

### Input Features

The model ingests 62 market indicators spanning:

| Category              | Description                    | Key Indicators                          |
| --------------------- | ------------------------------ | --------------------------------------- |
| **Volatility Regime** | Fear gauges and their momentum | VIX Index, VIX 1-3 week lags            |
| **Rate Environment**  | Yield levels and curve shape   | US 2Y/10Y/30Y, German Bunds, EONIA      |
| **Credit Conditions** | Risk appetite in fixed income  | High yield spreads, IG spreads, EM debt |
| **Equity Markets**    | Global equity performance      | MSCI World, regional indices, futures   |
| **Currency Stress**   | Safe haven flows               | JPY, Gold, DXY                          |
| **Commodities**       | Economic activity proxies      | Crude oil, Baltic Dry Index             |

### Feature Engineering

Raw market data undergoes several transformations:

1. **Lag Feature Construction**: VIX and MSCI World indices get 1, 2, and 3-week lags. Volatility momentum (the rate of change in fear) often matters more than absolute levels.

2. **Missing Data Handling**: Forward-fill with backward-fill fallback. Markets don't always trade synchronously across time zones.

3. **Schema Validation**: Strict feature ordering ensures the model receives inputs in the exact format it was trained on. Misaligned features produce garbage predictions.

The data loader handles the Bloomberg terminal export format used in the included dataset, with parsers for the multi-row header structure common in institutional data feeds.

---

## Understanding the Risk Framework

### Why VIX Momentum Dominates

The model's most important feature is `VIX Index_lag_3`—the VIX level from three weeks prior. This captures volatility regime shifts. Markets don't crash from calm conditions; they crash after volatility has already been elevated and building.

When VIX spikes from 12 to 25, the initial move often isn't the dangerous part. It's the sustained elevation that signals deteriorating risk appetite and potential for cascade effects.

### Cross-Asset Confirmation

Single-asset signals generate false positives. The model achieves better precision by requiring confirmation across:

- **Rates**: Flight-to-quality flows into Treasuries and Bunds
- **Credit**: Spread widening in high yield and emerging markets
- **Currencies**: Yen strength as carry trades unwind
- **Equity structure**: Futures basis, regional divergences

### Risk Level Interpretation

| Probability | Classification | What It Means                                                                             |
| ----------- | -------------- | ----------------------------------------------------------------------------------------- |
| 0-25%       | Low            | Normal market functioning. Volatility contained, correlations stable.                     |
| 25-50%      | Elevated       | Some stress indicators active. Worth monitoring but not actionable alone.                 |
| 50-75%      | High           | Multiple factors signaling risk. Consider reducing beta exposure.                         |
| 75-100%     | Critical       | Market conditions resemble historical pre-crash periods. Defensive positioning warranted. |

These thresholds aren't arbitrary—they're calibrated against historical drawdown severity.

---

## Investment Strategies

The system implements four systematic approaches to translate crash probabilities into portfolio positions. Each strategy addresses a different investment philosophy and risk profile.

### Strategy 1: Dynamic Risk Allocation

The most intuitive approach—scale equity exposure inversely to crash probability.

```
equity_weight = max_weight - (crash_probability × (max_weight - min_weight))
```

At 0% probability, you're at maximum equity (90%). At 100% probability, you're at minimum (10%). The key insight: don't wait for crash confirmation. Reduce exposure proportionally as warning signs accumulate.

**Characteristics:**

- Smooth, continuous adjustments
- No sharp portfolio transitions
- Higher turnover but smaller individual trades
- Best for: Investors who prefer gradual risk management

### Strategy 2: Regime Switching

Binary approach with predefined allocations for each risk regime:

| Regime      | Probability | Equity | Bonds | Cash |
| ----------- | ----------- | ------ | ----- | ---- |
| Risk-On     | 0-25%       | 80%    | 15%   | 5%   |
| Cautious    | 25-50%      | 50%    | 35%   | 15%  |
| Defensive   | 50-75%      | 25%    | 45%   | 30%  |
| Max Defense | 75-100%     | 10%    | 40%   | 50%  |

**Characteristics:**

- Clear, rule-based transitions
- Lower turnover (trades only on regime changes)
- Potentially more tax-efficient
- Best for: Investors who want simple, transparent rules

### Strategy 3: Probability-Weighted Hedging

Maintains base equity exposure while layering proportional hedges. The philosophy: stay invested for upside capture, but size tail protection based on risk levels.

```
hedge_allocation = crash_probability × max_hedge_weight
```

Hedge instruments could include:

- Put options or put spreads on equity indices
- VIX call options
- Inverse ETFs for tactical exposure
- Gold as a flight-to-quality asset

**Characteristics:**

- Preserves upside participation
- Explicit cost (hedge premium decay)
- Better during prolonged elevated-risk periods
- Best for: Investors who can't afford to miss rallies but need crash protection

### Strategy 4: Momentum + Risk Overlay

Combines trend-following with crash probability as an override signal.

Logic:

1. Calculate momentum: is price above 20-week moving average?
2. If momentum negative → exit regardless of crash probability
3. If momentum positive AND risk low → full exposure
4. If momentum positive BUT risk high → reduced exposure

**Characteristics:**

- Avoids fighting sustained downtrends
- Benefits from momentum's historical edge
- Can miss sharp reversal rallies
- Best for: Trend-followers who want additional crash protection

### Performance Metrics

Each strategy is evaluated using:

| Metric            | Description                                                   |
| ----------------- | ------------------------------------------------------------- |
| **Sharpe Ratio**  | Risk-adjusted return (excess return ÷ volatility)             |
| **Sortino Ratio** | Downside-adjusted return (penalizes only negative volatility) |
| **Max Drawdown**  | Worst peak-to-trough decline                                  |
| **Calmar Ratio**  | Return ÷ max drawdown—measures recovery efficiency            |
| **Win Rate**      | Percentage of positive periods                                |
| **Profit Factor** | Gross gains ÷ gross losses                                    |

The Strategy page lets you backtest all four approaches against a 60/40 benchmark with customizable parameters.

---

## Application Structure

```
├── app.py                        # Main dashboard
├── pages/
│   ├── 1_📊_Analysis.py          # Deep-dive analytics
│   ├── 2_📜_Historical.py        # Event studies
│   ├── 3_💰_Strategies.py        # Investment strategy backtester
│   └── 4_🤖_Advisor.py           # AI strategy advisor chatbot
└── src/
    ├── data_loader.py            # Data ingestion pipeline
    ├── feature_engineering.py    # Transformations
    ├── predictor.py              # Model inference
    ├── explainer.py              # SHAP attribution
    ├── strategy_engine.py        # Portfolio recommendations
    ├── investment_strategies.py  # Systematic trading strategies
    └── strategy_explainer.py     # AI explanation engine
```

### Dashboard (Main Page)

The primary interface shows:

- Current crash probability with trend
- Strategy recommendation based on risk tolerance setting
- Feature contribution waterfall
- 20-year probability history

### Analysis Page

For users who want to dig deeper:

- Custom date range selection
- Rolling statistics with confidence bands
- Risk regime distribution
- Category-level feature importance

### Historical Page

Event-specific analysis covering:

- Dot-com bubble (2000-2002)
- Global Financial Crisis (2007-2009)
- European Debt Crisis (2010-2012)
- China selloff (2015-2016)
- COVID-19 crash (2020)

Compare model predictions against actual outcomes to understand signal lead times and accuracy.

### Strategies Page

Full investment strategy backtesting:

- **Current Recommendation**: Real-time positioning advice based on latest probability and your risk tolerance
- **Strategy Comparison**: Side-by-side performance of all four strategies against 60/40 benchmark
- **Cumulative Returns**: Visualize growth trajectories and identify when strategies outperformed
- **Drawdown Analysis**: Compare worst-case scenarios across approaches
- **Dynamic Allocation Chart**: See how weights change over time for each strategy
- **Risk-Return Scatter**: Visual comparison of return vs volatility trade-offs

### Advisor Page

Interactive AI assistant that explains market conditions and strategies in plain language:

- **Conversational Interface**: Ask questions naturally—"What should I do now?" or "Why is risk high?"
- **Adaptive Communication**: Three modes (Simple/Balanced/Technical) that adjust explanations to your experience level
- **Context-Aware Responses**: Answers incorporate current market state and your risk tolerance
- **Quick Action Buttons**: One-click access to common questions
- **Strategy Explainer**: Get detailed breakdowns of any strategy's logic and trade-offs

The advisor translates complex quantitative signals into actionable guidance without requiring deep financial expertise.

---

## AI Strategy Advisor

The Advisor module provides intelligent explanations tailored to the user's background. It's designed to make sophisticated risk analytics accessible to a broader audience.

### Communication Styles

| Style         | Audience                           | Example Output                                                                                                                   |
| ------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Simple**    | New investors, non-financial users | "Markets look risky right now. The danger level is 72%—similar to conditions before past drops."                                 |
| **Balanced**  | Intermediate investors, advisors   | "Current crash probability: 72%. Multiple risk factors are elevated, warranting defensive positioning."                          |
| **Technical** | Quants, portfolio managers         | "P(crash) = 72.0%. Cross-asset stress confirmation: VIX momentum +, credit spreads widening, rates signaling flight-to-quality." |

### What the Advisor Can Explain

**Market Conditions**

- Current risk level and what's driving it
- Why certain factors are flagged as concerning
- How today compares to historical stress periods

**Investment Strategies**

- How each strategy works and when it's appropriate
- Trade-offs between different approaches
- Why a specific allocation is recommended

**Financial Concepts**

- Sharpe ratio, max drawdown, VIX, and other metrics
- Hedging approaches and their costs
- When to re-enter after defensive positioning

**Actionable Guidance**

- Specific steps based on current risk level
- Position sizing recommendations
- Hedging instrument suggestions

### Example Interactions

**User**: "What should I do now?"

**Advisor** (Balanced mode, High risk):

> Based on current HIGH risk conditions:
>
> 1. Reduce equity to 25-35% of portfolio
> 2. Increase bond allocation, favoring quality
> 3. Raise cash to 15-20%
> 4. Exit speculative and leveraged positions

**User**: "Explain the momentum strategy"

**Advisor**:

> **Momentum + Risk Overlay**: Combines trend-following with crash protection.
> The strategy stays invested when momentum is positive and risk is manageable,
> but exits when either signal turns negative.
>
> **Pros:** Captures trends, double protection, clear exit rules
> **Cons:** May miss reversals, whipsaws possible

---

## Getting Started

### Requirements

- Python 3.9+
- ~500MB disk space for dependencies

### Installation

```bash
git clone https://github.com/yourusername/Market-Anomaly-Detector.git
cd Market-Anomaly-Detector

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Running Locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### Deployment

Works out of the box with Streamlit Cloud:

1. Push to GitHub
2. Connect repo at share.streamlit.io
3. Point to `app.py`

For containerized deployment:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Model Details

### Training Approach

The XGBoost classifier was trained on weekly observations from 1999-2021, with crash periods labeled based on subsequent 3-month drawdowns exceeding 15%.

| Parameter     | Value                       |
| ------------- | --------------------------- |
| Algorithm     | XGBoost (gradient boosting) |
| Trees         | 200                         |
| Max Depth     | 5                           |
| Learning Rate | 0.05                        |
| Objective     | Binary cross-entropy        |

### Performance Characteristics

The model trades off precision and recall. It will generate false positives—periods flagged as high risk that don't result in crashes. This is intentional. In risk management, missing a crash (false negative) is far more costly than unnecessary hedging (false positive).

Expect the model to spend roughly 10-15% of time in "high" or "critical" states during normal market conditions.

---

## Practical Usage Notes

**Don't use this as a timing tool.** The model identifies elevated risk environments, not precise crash dates. Markets can stay stressed for months before dislocating, or they can resolve without incident.

**Combine with fundamental analysis.** A high probability reading during earnings season volatility means something different than the same reading during a liquidity crisis.

**Calibrate to your portfolio.** The strategy recommendations assume a generic balanced portfolio. Actual positioning should account for your specific exposures, hedging costs, and risk tolerance.

**Watch for regime changes.** The model was trained on historical data. Novel market structures (cryptocurrency integration, central bank balance sheet expansion) may create patterns outside the training distribution.

### Interpreting Strategy Backtests

When evaluating strategy performance:

- **Sharpe Ratio > 0.5** is generally considered acceptable; above 1.0 is strong
- **Max Drawdown** matters more for capital preservation mandates—compare against your loss tolerance
- **Win Rate** alone is misleading; a 40% win rate with 2:1 profit factor is excellent
- **Calmar Ratio** shows how efficiently you recover from losses—higher is better

Remember that backtests are optimistic. Real-world implementation faces:

- Execution slippage during volatile periods
- Bid-ask spreads that increase during stress
- Funding costs for leveraged positions
- Tax implications of frequent rebalancing

The strategies page uses weekly rebalancing and 10bps transaction costs as defaults. Adjust these parameters to match your actual trading environment.

---

## Data Sources

The included `FinancialMarketData.csv` contains 1,149 weekly observations of 57 market indicators, covering April 1999 through April 2021. Data sourced from Bloomberg terminal exports.

For production use, you'd want to integrate live data feeds. The `data_loader` module is designed to be extensible—add new loaders for your preferred data vendor.

---

## References

Technical background on the approaches used:

- Liu et al., "Isolation Forest" (2008) - Anomaly detection foundations
- Lundberg & Lee, "SHAP Values" (2017) - Model interpretability
- Unit8, "Financial Transaction Anomaly Detection" - Applied ML in finance
- Investopedia, "Market Anomalies" - Financial context

---

## Limitations

This is a research tool, not investment advice. Model outputs should inform human decision-making, not replace it.

- Historical patterns may not repeat
- Black swan events are by definition unpredictable
- Transaction costs and implementation slippage aren't modeled
- The training data ends in 2021

Always consult qualified financial advisors for actual investment decisions.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Pull requests welcome. Areas that would benefit from contribution:

- Additional data source integrations (Bloomberg API, Refinitiv, Alpha Vantage)
- Alternative model architectures (LSTM, transformer, ensemble methods)
- Options-based hedging calculators with Greeks
- Real-time alerting system (email/Slack notifications)
- Transaction cost modeling improvements

Open an issue first to discuss significant changes.
