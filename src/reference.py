"""
Static reference data: geography, FX rates and merchant categories.

Kept separate from the simulator because the *consumer* also needs it
(merchant-category risk, home-country lookups) – exactly like a real scoring
service that joins the event stream against slowly-changing reference tables.
"""
from __future__ import annotations

# (city, ISO-3 country, lat, lon, currency, traffic weight)
CITIES: list[tuple[str, str, float, float, str, float]] = [
    ("New York", "USA", 40.71, -74.01, "USD", 9), ("Los Angeles", "USA", 34.05, -118.24, "USD", 6),
    ("Chicago", "USA", 41.88, -87.63, "USD", 4), ("Miami", "USA", 25.76, -80.19, "USD", 3),
    ("Toronto", "CAN", 43.65, -79.38, "CAD", 3), ("Mexico City", "MEX", 19.43, -99.13, "MXN", 3),
    ("São Paulo", "BRA", -23.55, -46.63, "BRL", 3), ("London", "GBR", 51.51, -0.13, "GBP", 6),
    ("Manchester", "GBR", 53.48, -2.24, "GBP", 2), ("Paris", "FRA", 48.86, 2.35, "EUR", 4),
    ("Berlin", "DEU", 52.52, 13.40, "EUR", 4), ("Madrid", "ESP", 40.42, -3.70, "EUR", 2),
    ("Amsterdam", "NLD", 52.37, 4.90, "EUR", 2), ("Dubai", "ARE", 25.20, 55.27, "AED", 2),
    ("Lagos", "NGA", 6.52, 3.38, "NGN", 2), ("Johannesburg", "ZAF", -26.20, 28.05, "ZAR", 1),
    ("Mumbai", "IND", 19.08, 72.88, "INR", 4), ("Bengaluru", "IND", 12.97, 77.59, "INR", 3),
    ("Singapore", "SGP", 1.35, 103.82, "SGD", 3), ("Tokyo", "JPN", 35.68, 139.69, "JPY", 4),
    ("Seoul", "KOR", 37.57, 126.98, "KRW", 2), ("Sydney", "AUS", -33.87, 151.21, "AUD", 3),
    ("Hong Kong", "HKG", 22.32, 114.17, "HKD", 2), ("Moscow", "RUS", 55.76, 37.62, "RUB", 1),
    ("Bucharest", "ROU", 44.43, 26.10, "RON", 1),
]

# Units of USD per 1 unit of currency (static snapshot; production would stream FX).
FX_TO_USD: dict[str, float] = {
    "USD": 1.0, "CAD": 0.74, "MXN": 0.058, "BRL": 0.20, "GBP": 1.27, "EUR": 1.09, "AED": 0.27, "NGN": 0.0011,
    "ZAR": 0.054, "INR": 0.012, "SGD": 0.74, "JPY": 0.0068, "KRW": 0.00075, "AUD": 0.66, "HKD": 0.128,
    "RUB": 0.011, "RON": 0.22,
}

# category -> (inherent risk 0-1, typical amount multiplier, online share, allowed txn types)
MERCHANT_CATEGORIES: dict[str, dict] = {
    "Grocery":          {"risk": 0.05, "amount": 0.6, "online": 0.15, "types": ["POS", "ECOM"]},
    "Restaurants":      {"risk": 0.05, "amount": 0.5, "online": 0.10, "types": ["POS"]},
    "Fuel":             {"risk": 0.10, "amount": 0.6, "online": 0.00, "types": ["POS"]},
    "Pharmacy":         {"risk": 0.05, "amount": 0.4, "online": 0.20, "types": ["POS", "ECOM"]},
    "Fashion":          {"risk": 0.20, "amount": 1.2, "online": 0.55, "types": ["POS", "ECOM"]},
    "Electronics":      {"risk": 0.45, "amount": 3.0, "online": 0.60, "types": ["POS", "ECOM"]},
    "Travel & Airlines": {"risk": 0.40, "amount": 4.0, "online": 0.85, "types": ["ECOM"]},
    "Hotels":           {"risk": 0.25, "amount": 3.0, "online": 0.60, "types": ["POS", "ECOM"]},
    "Ride Hailing":     {"risk": 0.10, "amount": 0.3, "online": 1.00, "types": ["ECOM"]},
    "Streaming & Apps": {"risk": 0.20, "amount": 0.2, "online": 1.00, "types": ["ECOM"]},
    "Digital Goods":    {"risk": 0.60, "amount": 0.4, "online": 1.00, "types": ["ECOM"]},
    "Gift Cards":       {"risk": 0.80, "amount": 1.5, "online": 0.80, "types": ["ECOM", "POS"]},
    "Crypto Exchange":  {"risk": 0.85, "amount": 4.0, "online": 1.00, "types": ["ECOM", "TRANSFER"]},
    "Gambling":         {"risk": 0.75, "amount": 1.5, "online": 1.00, "types": ["ECOM"]},
    "Utilities":        {"risk": 0.05, "amount": 1.0, "online": 0.90, "types": ["ECOM", "TRANSFER"]},
    "ATM Cash":         {"risk": 0.35, "amount": 1.6, "online": 0.00, "types": ["ATM"]},
    "P2P Wallet":       {"risk": 0.40, "amount": 1.0, "online": 1.00, "types": ["P2P"]},
}

CATEGORY_RISK = {k: v["risk"] for k, v in MERCHANT_CATEGORIES.items()}
HIGH_RISK_CATEGORIES = {k for k, v in CATEGORY_RISK.items() if v >= 0.6}
CITY_INDEX = {c[0]: c for c in CITIES}

# Per-card attack episodes (one compromised card at a time) ...
EPISODE_TYPES = ["card_testing", "account_takeover", "impossible_travel", "velocity_burst", "high_amount_anomaly"]
# ... plus coordinated rings: many cards driven from a few shared devices (detected by the graph job, not rules).
FRAUD_TYPES = EPISODE_TYPES + ["fraud_ring"]
RING_COLLECTOR_CATEGORIES = ["P2P Wallet", "Gift Cards", "Crypto Exchange"]
