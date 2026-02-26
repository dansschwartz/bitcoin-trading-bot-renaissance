"""KNOWN DEAD ENDS — confirmed failures. DO NOT repropose without new evidence."""

KNOWN_DEAD_ENDS = [
    {"id": "DE-001", "desc": "Deep learning on raw non-normalized multi-year prices",
     "reason": "44x price range makes absolute features meaningless", "applies": ["linguist","cryptographer"]},
    {"id": "DE-002", "desc": "Hard +/-1 binary labels for 5-min direction",
     "reason": "Coin flip at 5-min. Use soft labels: tanh(return*100)", "applies": ["linguist"]},
    {"id": "DE-003", "desc": "MSE loss when model predicts near zero",
     "reason": "Gradient dead zone. Use Huber loss.", "applies": ["linguist"]},
    {"id": "DE-004", "desc": "Training >100 epochs when early stopping fires at 13-22",
     "reason": "More data/features >> more compute", "applies": ["linguist"]},
    {"id": "DE-005", "desc": "Sentiment analysis from Twitter/social media",
     "reason": "Noise, manipulation, lags price. Negative EV.", "applies": ["linguist","cryptographer"]},
    {"id": "DE-006", "desc": "Genetic algorithms for strategy optimization",
     "reason": "Catastrophic overfitting on historical data", "applies": ["mathematician","linguist"]},
    {"id": "DE-007", "desc": "Pure technical analysis patterns (H&S, cup-handle)",
     "reason": "Not statistically predictive when quantified", "applies": ["cryptographer"]},
    {"id": "DE-008", "desc": "Microsecond latency optimization",
     "reason": "Exchange processing 5-50ms, 5-min cycles. Speed irrelevant.", "applies": ["systems_engineer"]},
    {"id": "DE-009", "desc": "Global normalization across multi-year data",
     "reason": "Non-stationary. Per-WINDOW standardization only.", "applies": ["linguist","physicist"]},
    {"id": "DE-010", "desc": "Paid data before free sources exhausted",
     "reason": "Binance derivatives data is FREE. Cross-asset is FREE.", "applies": ["all"]},
    {"id": "DE-011", "desc": "Shuffled/random time series splits",
     "reason": "Massive lookahead bias. Walk-forward ONLY.", "applies": ["linguist"]},
    {"id": "DE-012", "desc": "Genetic/adaptive weight mutation at runtime",
     "reason": "Grade D in audit. Erratic sizing. Weights must be stable.", "applies": ["mathematician","systems_engineer"]},
]

def is_dead_end(description: str) -> list:
    """Check if a proposal matches known dead ends. Returns list of matching IDs."""
    d = description.lower()
    kw = {
        "DE-001": ["raw price", "unnormalized", "absolute price"],
        "DE-002": ["binary label", "hard label", "+1/-1"],
        "DE-003": ["mse loss", "mean squared error"],
        "DE-004": ["more epochs", "train longer"],
        "DE-005": ["sentiment", "twitter", "social media", "reddit"],
        "DE-006": ["genetic algorithm", "evolutionary", "ga optim"],
        "DE-007": ["head and shoulders", "chart pattern", "technical pattern"],
        "DE-008": ["microsecond", "nanosecond", "latency optim"],
        "DE-009": ["global normalization", "global z-score"],
        "DE-010": ["glassnode", "coinglass", "paid data"],
        "DE-011": ["shuffle", "random split", "random train"],
        "DE-012": ["genetic weight", "adaptive mutation", "evolve weights"],
    }
    return [de_id for de_id, words in kw.items() if any(w in d for w in words)]

def get_dead_ends_for(researcher: str) -> list:
    return [de for de in KNOWN_DEAD_ENDS if researcher in de["applies"] or "all" in de["applies"]]
