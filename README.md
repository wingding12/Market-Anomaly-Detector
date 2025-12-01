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

**Historical Backtesting**: Examine how the model would have performed during specific market events. Compare predicted probabilities against actual market outcomes to build confidence in the signals.

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

## Application Structure

```
├── app.py                      # Main dashboard
├── pages/
│   ├── 1_📊_Analysis.py        # Deep-dive analytics
│   └── 2_📜_Historical.py      # Event studies
└── src/
    ├── data_loader.py          # Data ingestion pipeline
    ├── feature_engineering.py  # Transformations
    ├── predictor.py            # Model inference
    ├── explainer.py            # SHAP attribution
    └── strategy_engine.py      # Portfolio recommendations
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

- Additional data source integrations
- Alternative model architectures (LSTM, transformer)
- Enhanced backtesting framework
- Real-time alerting system

Open an issue first to discuss significant changes.
