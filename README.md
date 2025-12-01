# 📉 Market Anomaly Detector

An early warning system for detecting potential financial market crashes using machine learning. This application analyzes historical market data, identifies anomalous patterns, and provides actionable investment strategies for risk mitigation.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🖥️ Screenshots

### Main Dashboard

- Real-time crash probability gauge with color-coded risk levels
- Strategy recommendations with portfolio allocation charts
- SHAP-based feature importance analysis
- Historical probability timeline

### Analysis Page

- Interactive date range selection
- Rolling statistics with confidence bands
- Risk distribution pie charts
- Feature category importance breakdown

### Historical Page

- Analysis of major market events (2000-2020)
- Cross-event comparison charts
- VIX-based market regime detection
- High-risk period identification

---

## 🎯 Features

- **Real-time Market Analysis**: Monitor current market conditions and crash probability
- **Historical Backtesting**: Analyze past market data to validate prediction accuracy
- **Explainable AI**: SHAP-based explanations for understanding prediction drivers
- **Investment Strategies**: Automated risk mitigation recommendations
- **Interactive Dashboard**: Beautiful, intuitive Streamlit-powered interface
- **Multi-page App**: Dedicated pages for detailed analysis and historical review

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
│   ├── config.toml          # Streamlit theme & configuration
│   └── secrets.toml.example # Secrets template
│
├── src/                     # Core source modules
│   ├── __init__.py          # Package exports
│   ├── feature_schema.py    # Feature definitions & validation
│   ├── data_loader.py       # Data fetching & CSV handling
│   ├── feature_engineering.py # Feature extraction pipeline
│   ├── model_utils.py       # Model loading utilities
│   ├── predictor.py         # Prediction wrapper
│   ├── explainer.py         # SHAP explainability
│   └── strategy_engine.py   # Investment recommendations
│
├── pages/                   # Streamlit multi-page app
│   ├── 1_📊_Analysis.py     # Detailed analysis page
│   └── 2_📜_Historical.py   # Historical events page
│
├── data/                    # User data & cache
│   └── .gitkeep
│
└── models/                  # Model artifacts
    └── .gitkeep
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

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Deploy!

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Environment Variables

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` for any API keys or sensitive configuration.

---

## 📊 How It Works

### Anomaly Detection Approach

This system uses an **XGBoost classifier** trained on historical market data to identify conditions that precede market crashes. The approach is inspired by [anomaly detection techniques in financial transactions](https://unit8.com/resources/a-guide-to-building-a-financial-transaction-anomaly-detector/).

### Key Components

1. **Data Ingestion**: Loads market data from CSV with 62 financial indicators
2. **Feature Engineering**: Computes lag features for VIX and MSCI World
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
- [x] **Phase 2**: Model Integration ✅
- [x] **Phase 3**: Strategy Engine ✅
- [x] **Phase 4**: Streamlit UI - Core ✅
- [x] **Phase 5**: Streamlit UI - Visualizations ✅
- [x] **Phase 6**: Historical Analysis ✅
- [x] **Phase 7**: Polish & Deployment ✅

### Running Tests

```bash
# Test data loading
python -m src.data_loader

# Test predictions
python -m src.predictor

# Test explanations
python -m src.explainer

# Test strategies
python -m src.strategy_engine
```

---

## 📈 Data Sources

The application uses the included dataset:

- **FinancialMarketData.csv**: 1,149 weekly observations (1999-2021)
- **62 features** covering global financial markets

### Data Format

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

The included `xgb_weights.pkl` is a pre-trained XGBoost binary classifier.

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

## 🙏 Acknowledgments

- XGBoost team for the excellent gradient boosting library
- Streamlit team for the amazing web framework
- SHAP library for explainable AI capabilities

---

<p align="center">
  Built with ❤️ for safer investing
</p>
