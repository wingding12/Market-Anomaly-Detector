"""
Feature Schema for Market Anomaly Detector
==========================================

This module defines the feature schema expected by the XGBoost crash prediction model.
The model expects 62 features total: 56 base market indicators + 6 lag features.

Feature Categories:
- Commodities & Currencies (8 features)
- Volatility (1 feature + 3 lags = 4 total)
- US Rates & Yields (6 features)
- European Rates (4 features)
- Italian Bonds (3 features)
- Japanese Bonds (3 features)
- UK Bonds (3 features)
- Bond Indices (9 features)
- Equity Indices (10 features + 3 lags = 13 total)
- Futures (8 features)
"""

from typing import List, Dict, Tuple


# =============================================================================
# Base Features (56 total) - Direct market indicators
# =============================================================================

COMMODITIES_CURRENCIES: List[str] = [
    "XAU BGNL Curncy",   # Gold in EUR
    "DXY Curncy",        # US Dollar Index
    "JPY Curncy",        # Japanese Yen
    "GBP Curncy",        # British Pound
    "Cl1 Comdty",        # WTI Crude Oil Front Month
    "CRY Index",         # Commodity Research Bureau Index
    "BDIY Index",        # Baltic Dry Index
    "ECSURPUS Index",    # US Economic Surprise Index
]

VOLATILITY: List[str] = [
    "VIX Index",         # CBOE Volatility Index
]

US_RATES: List[str] = [
    "USGG30YR Index",    # US 30-Year Treasury Yield
    "GT10 Govt",         # US 10-Year Treasury
    "USGG2YR Index",     # US 2-Year Treasury Yield
    "USGG3M Index",      # US 3-Month Treasury Bill
    "US0001M Index",     # US 1-Month LIBOR
]

EUROPEAN_RATES: List[str] = [
    "GTDEM30Y Govt",     # German 30-Year Bund
    "GTDEM10Y Govt",     # German 10-Year Bund
    "GTDEM2Y Govt",      # German 2-Year Schatz
    "EONIA Index",       # Euro Overnight Index Average
]

ITALIAN_BONDS: List[str] = [
    "GTITL30YR Corp",    # Italian 30-Year Bond
    "GTITL10YR Corp",    # Italian 10-Year Bond
    "GTITL2YR Corp",     # Italian 2-Year Bond
]

JAPANESE_BONDS: List[str] = [
    "GTJPY30YR Corp",    # Japanese 30-Year Bond
    "GTJPY10YR Corp",    # Japanese 10-Year Bond
    "GTJPY2YR Corp",     # Japanese 2-Year Bond
]

UK_BONDS: List[str] = [
    "GTGBP30Y Govt",     # UK 30-Year Gilt
    "GTGBP20Y Govt",     # UK 20-Year Gilt
    "GTGBP2Y Govt",      # UK 2-Year Gilt
]

BOND_INDICES: List[str] = [
    "LUMSTRUU Index",    # Bloomberg US MBS Total Return
    "LMBITR Index",      # Bloomberg US Municipal Bond
    "LUACTRUU Index",    # Bloomberg US Corporate Total Return
    "LF98TRUU Index",    # Bloomberg US Corporate High Yield
    "LG30TRUU Index",    # Bloomberg US Long Government/Credit
    "LP01TREU Index",    # Bloomberg Pan-European Aggregate
    "EMUSTRUU Index",    # Bloomberg Euro Aggregate
    "LF94TRUU Index",    # Bloomberg Global High Yield
    "LEGATRUU Index",    # Bloomberg Global Aggregate
]

EQUITY_INDICES: List[str] = [
    "MXUS Index",        # MSCI USA
    "MXEU Index",        # MSCI Europe
    "MXJP Index",        # MSCI Japan
    "MXBR Index",        # MSCI Brazil
    "MXRU Index",        # MSCI Russia
    "MXIN Index",        # MSCI India
    "MXCN Index",        # MSCI China
    "MXWO Index",        # MSCI World
    "MXWD Index",        # MSCI World
    "HFRXGL Index",      # HFRX Global Hedge Fund Index
]

FUTURES: List[str] = [
    "RX1 Comdty",        # Euro-Bund Future
    "TY1 Comdty",        # US 10-Year Treasury Note Future
    "GC1 Comdty",        # Gold Future
    "CO1 Comdty",        # Brent Crude Future
    "ES1 Index",         # S&P 500 E-mini Future
    "VG1 Index",         # Euro Stoxx 50 Future
    "NQ1 Index",         # Nasdaq 100 E-mini Future
    "TP1 Index",         # Topix Future
    "DU1 Comdty",        # Euro-Schatz Future
    "TU2 Comdty",        # US 2-Year Treasury Note Future
]


# =============================================================================
# Lag Features (6 total) - Computed from base features
# =============================================================================

LAG_FEATURES: List[str] = [
    "VIX Index_lag_1",   # VIX 1-period lag
    "VIX Index_lag_2",   # VIX 2-period lag
    "VIX Index_lag_3",   # VIX 3-period lag
    "MXWO Index_lag_1",  # MSCI World 1-period lag
    "MXWO Index_lag_2",  # MSCI World 2-period lag
    "MXWO Index_lag_3",  # MSCI World 3-period lag
]

# Features that need lag computation
LAG_SOURCE_FEATURES: Dict[str, int] = {
    "VIX Index": 3,      # 3 lag periods needed
    "MXWO Index": 3,     # 3 lag periods needed
}


# =============================================================================
# Complete Feature Lists
# =============================================================================

# All base features in order expected by model
BASE_FEATURES: List[str] = (
    COMMODITIES_CURRENCIES[:1] +  # XAU BGNL Curncy
    [COMMODITIES_CURRENCIES[7]] +  # ECSURPUS Index
    COMMODITIES_CURRENCIES[6:7] +  # BDIY Index
    COMMODITIES_CURRENCIES[5:6] +  # CRY Index
    COMMODITIES_CURRENCIES[1:5] +  # DXY, JPY, GBP, Cl1
    VOLATILITY +                   # VIX Index
    US_RATES +                     # US rates
    EUROPEAN_RATES +               # European rates
    ITALIAN_BONDS +                # Italian bonds
    JAPANESE_BONDS +               # Japanese bonds
    UK_BONDS +                     # UK bonds
    BOND_INDICES +                 # Bond indices
    EQUITY_INDICES +               # Equity indices
    FUTURES                        # Futures
)

# All features in exact order expected by the model
MODEL_FEATURES: List[str] = [
    "XAU BGNL Curncy",
    "ECSURPUS Index",
    "BDIY Index",
    "CRY Index",
    "DXY Curncy",
    "JPY Curncy",
    "GBP Curncy",
    "Cl1 Comdty",
    "VIX Index",
    "USGG30YR Index",
    "GT10 Govt",
    "USGG2YR Index",
    "USGG3M Index",
    "US0001M Index",
    "GTDEM30Y Govt",
    "GTDEM10Y Govt",
    "GTDEM2Y Govt",
    "EONIA Index",
    "GTITL30YR Corp",
    "GTITL10YR Corp",
    "GTITL2YR Corp",
    "GTJPY30YR Corp",
    "GTJPY10YR Corp",
    "GTJPY2YR Corp",
    "GTGBP30Y Govt",
    "GTGBP20Y Govt",
    "GTGBP2Y Govt",
    "LUMSTRUU Index",
    "LMBITR Index",
    "LUACTRUU Index",
    "LF98TRUU Index",
    "LG30TRUU Index",
    "LP01TREU Index",
    "EMUSTRUU Index",
    "LF94TRUU Index",
    "MXUS Index",
    "MXEU Index",
    "MXJP Index",
    "MXBR Index",
    "MXRU Index",
    "MXIN Index",
    "MXCN Index",
    "MXWO Index",
    "MXWD Index",
    "LEGATRUU Index",
    "HFRXGL Index",
    "RX1 Comdty",
    "TY1 Comdty",
    "GC1 Comdty",
    "CO1 Comdty",
    "ES1 Index",
    "VG1 Index",
    "NQ1 Index",
    "TP1 Index",
    "DU1 Comdty",
    "TU2 Comdty",
    # Lag features
    "VIX Index_lag_1",
    "VIX Index_lag_2",
    "VIX Index_lag_3",
    "MXWO Index_lag_1",
    "MXWO Index_lag_2",
    "MXWO Index_lag_3",
]

# Total feature count
N_FEATURES: int = 62
N_BASE_FEATURES: int = 56
N_LAG_FEATURES: int = 6


# =============================================================================
# Feature Importance (from trained model)
# =============================================================================

# Top features by importance (feature_importances_ from model)
TOP_FEATURES_BY_IMPORTANCE: List[Tuple[str, float]] = [
    ("VIX Index_lag_3", 0.3537),      # Most important - VIX momentum
    ("EONIA Index", 0.0882),          # Euro overnight rate
    ("GTDEM2Y Govt", 0.0856),         # German 2Y yield
    ("ES1 Index", 0.0819),            # S&P 500 futures
    ("MXJP Index", 0.0718),           # MSCI Japan
    ("NQ1 Index", 0.0637),            # Nasdaq futures
    ("GTJPY2YR Corp", 0.0583),        # Japanese 2Y
    ("GTGBP2Y Govt", 0.0392),         # UK 2Y gilt
    ("JPY Curncy", 0.0298),           # Japanese Yen
    ("GTGBP30Y Govt", 0.0230),        # UK 30Y gilt
]


# =============================================================================
# CSV Parsing Configuration
# =============================================================================

CSV_CONFIG: Dict = {
    "header_row": 5,           # Row index containing column headers
    "skip_rows": [6],          # Rows to skip after header
    "date_column": "Ticker",   # Original column name for dates
    "date_column_rename": "Date",
    "drop_columns": ["Unnamed: 0", "Unnamed: 1"],
    "extra_columns": ["LLL1 Index"],  # Columns in CSV not used by model
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_feature_category(feature_name: str) -> str:
    """Return the category of a given feature."""
    if feature_name in COMMODITIES_CURRENCIES:
        return "Commodities & Currencies"
    elif feature_name in VOLATILITY or "VIX" in feature_name:
        return "Volatility"
    elif feature_name in US_RATES:
        return "US Rates"
    elif feature_name in EUROPEAN_RATES:
        return "European Rates"
    elif feature_name in ITALIAN_BONDS:
        return "Italian Bonds"
    elif feature_name in JAPANESE_BONDS:
        return "Japanese Bonds"
    elif feature_name in UK_BONDS:
        return "UK Bonds"
    elif feature_name in BOND_INDICES:
        return "Bond Indices"
    elif feature_name in EQUITY_INDICES or "MXWO" in feature_name:
        return "Equity Indices"
    elif feature_name in FUTURES:
        return "Futures"
    else:
        return "Unknown"


def validate_features(feature_names: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that the provided features match model requirements.
    
    Returns:
        Tuple of (is_valid, missing_features)
    """
    required = set(MODEL_FEATURES)
    provided = set(feature_names)
    missing = required - provided
    return len(missing) == 0, list(missing)


if __name__ == "__main__":
    # Print summary when run directly
    print("=" * 60)
    print("Market Anomaly Detector - Feature Schema")
    print("=" * 60)
    print(f"\nTotal Features Required: {N_FEATURES}")
    print(f"  - Base Features: {N_BASE_FEATURES}")
    print(f"  - Lag Features: {N_LAG_FEATURES}")
    print(f"\nTop 10 Features by Importance:")
    for feat, imp in TOP_FEATURES_BY_IMPORTANCE:
        print(f"  {feat}: {imp:.4f}")

