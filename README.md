# 📉 Market Anomaly Detector

An early warning system for detecting potential financial market crashes using machine learning. This application analyzes historical market data, identifies anomalous patterns, and provides actionable investment strategies for risk mitigation.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 Features

- **Real-time Market Analysis**: Monitor current market conditions and crash probability
- **Historical Backtesting**: Analyze past market data to validate prediction accuracy
- **Explainable AI**: SHAP-based explanations for understanding prediction drivers
- **Investment Strategies**: Automated risk mitigation recommendations
- **Interactive Dashboard**: Beautiful, intuitive Streamlit-powered interface

---

## 🏗️ Project Structure

```
Market-Anomaly-Detector/
├── app.py                    # Main Streamlit entry point
├── requirements.txt          # Python dependencies
├── xgb_weights.pkl          # Pre-trained XGBoost model
├── FinancialMarketData.csv  # Historical market data (1999-2021)
├── README.md                 # This file
│
├── .streamlit/
│   └── config.toml          # Streamlit theme & configuration
│
├── src/                     # Core source modules
│   ├── __init__.py
│   ├── feature_schema.py    # Feature definitions & validation
│   ├── data_loader.py       # Data fetching & CSV handling
│   ├── feature_engineering.py # Feature extraction pipeline
│   ├── model_utils.py       # Model loading utilities
│   ├── predictor.py         # Prediction wrapper
│   ├── explainer.py         # SHAP explainability
│   ├── strategy_engine.py   # Investment recommendations
│   └── backtester.py        # Historical analysis
│
├── pages/                   # Streamlit multi-page app
│   ├── 1_Dashboard.py       # Main monitoring dashboard
│   └── 2_Historical.py      # Historical analysis page
│
├── data/                    # User data & cache
│   └── (user-uploaded CSVs)
│
└── models/                  # Model artifacts
    └── (saved models)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Market-Anomaly-Detector.git
   cd Market-Anomaly-Detector
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   Navigate to `http://localhost:8501`

---

## 📊 How It Works

### Anomaly Detection Approach

This system uses an **XGBoost classifier** trained on historical market data to identify conditions that precede market crashes. The approach is inspired by [anomaly detection techniques in financial transactions](https://unit8.com/resources/a-guide-to-building-a-financial-transaction-anomaly-detector/).

### Key Components

1. **Data Ingestion**: Fetches market data via Yahoo Finance API or accepts user-uploaded CSV files
2. **Feature Engineering**: Extracts technical indicators and market signals
3. **Crash Prediction**: XGBoost model outputs crash probability (0-100%)
4. **Explainability**: SHAP values reveal which features drive predictions
5. **Strategy Engine**: Converts predictions into actionable investment advice

### Risk Levels

| Level       | Crash Probability | Recommended Action    |
| ----------- | ----------------- | --------------------- |
| 🟢 Low      | 0-25%             | Normal operations     |
| 🟡 Medium   | 25-50%            | Increase monitoring   |
| 🟠 High     | 50-75%            | Reduce exposure       |
| 🔴 Critical | 75-100%           | Defensive positioning |

---

## 🛠️ Development

### Project Phases

- [x] **Phase 1**: Foundation & Data Layer ✅

  - [x] Step 1.1: Project structure setup
  - [x] Step 1.2: Inspect model input requirements
  - [x] Step 1.3: Data fetching module
  - [x] Step 1.4: Feature engineering pipeline
  - [x] Step 1.5: Integration testing

- [x] **Phase 2**: Model Integration ✅
- [ ] **Phase 3**: Strategy Engine
- [ ] **Phase 4**: Streamlit UI - Core
- [ ] **Phase 5**: Streamlit UI - Visualizations
- [ ] **Phase 6**: Historical Analysis
- [ ] **Phase 7**: Polish & Deployment

---

## 📈 Data Sources

The application supports multiple data sources:

- **Included Dataset**: `FinancialMarketData.csv` with 1,149 weekly observations (1999-2021)
- **CSV Upload**: Custom datasets matching the required feature schema
- **Pre-loaded Samples**: Demo data for testing

### Data Format

The model expects **62 features** covering global financial markets:

| Category                 | Features | Examples                                            |
| ------------------------ | -------- | --------------------------------------------------- |
| Commodities & Currencies | 8        | Gold (XAU), Dollar Index (DXY), JPY, GBP, WTI Crude |
| Volatility               | 4        | VIX Index + 3 lag periods                           |
| US Rates                 | 5        | 30Y, 10Y, 2Y Treasury, 3M T-Bill, 1M LIBOR          |
| European Rates           | 4        | German Bunds, EONIA                                 |
| Global Bonds             | 9        | Italian, Japanese, UK government bonds              |
| Bond Indices             | 9        | Bloomberg Aggregate, MBS, Corporate, High Yield     |
| Equity Indices           | 13       | MSCI USA, Europe, Japan, EM, World + 3 lags         |
| Futures                  | 10       | S&P 500, Nasdaq, Euro Stoxx, Gold, Brent            |

See `src/feature_schema.py` for the complete feature specification.

---

## 🔬 Model Information

### Pre-trained Model

The included `xgb_weights.pkl` is a pre-trained XGBoost binary classifier that predicts market crash conditions.

| Property      | Value                         |
| ------------- | ----------------------------- |
| Algorithm     | XGBoost (Gradient Boosting)   |
| Type          | Binary Classification         |
| Output        | Crash / No Crash              |
| Features      | 62 (56 base + 6 lag features) |
| Estimators    | 200 trees                     |
| Max Depth     | 5                             |
| Learning Rate | 0.05                          |
| Objective     | binary:logistic               |

### Top Predictive Features

| Feature         | Importance | Description               |
| --------------- | ---------- | ------------------------- |
| VIX Index_lag_3 | 35.4%      | VIX momentum (3-week lag) |
| EONIA Index     | 8.8%       | Euro overnight rate       |
| GTDEM2Y Govt    | 8.6%       | German 2-year yield       |
| ES1 Index       | 8.2%       | S&P 500 futures           |
| MXJP Index      | 7.2%       | MSCI Japan                |
| NQ1 Index       | 6.4%       | Nasdaq futures            |

The model heavily relies on **volatility momentum** (VIX lags) as the primary crash indicator.

---

## 📚 References

- [A Guide to Building a Financial Transaction Anomaly Detector](https://unit8.com/resources/a-guide-to-building-a-financial-transaction-anomaly-detector/)
- [Trading with Market Anomalies - Investopedia](https://www.investopedia.com/articles/financial-theory/11/trading-with-market-anomalies.asp)
- [Anomaly Detection Algorithms - Built In](https://builtin.com/machine-learning/anomaly-detection-algorithms)
- [Anomaly Detection with Unsupervised ML - Medium](https://medium.com/simform-engineering/anomaly-detection-with-unsupervised-machine-learning-3bcf4c431aff)

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It should not be used as the sole basis for investment decisions. Financial markets are inherently unpredictable, and past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<p align="center">
  Built with ❤️ for safer investing
</p>
