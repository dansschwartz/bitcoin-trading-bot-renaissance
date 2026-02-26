"""Shared atoms — wrap shared/queries.py and shared/data_loader.py as queryable atoms."""
from knowledge.registry import KB, Atom, Pair, Regime, Scale, AType, Affects


def _weekly_performance(**kwargs) -> dict:
    """Retrieve weekly performance summary from database."""
    from knowledge.shared.queries import weekly_performance
    return weekly_performance(**kwargs)

def _model_accuracy(**kwargs) -> dict:
    """Retrieve model accuracy matrix from database."""
    from knowledge.shared.queries import model_accuracy_matrix
    df = model_accuracy_matrix(**kwargs)
    return df.to_dict() if not df.empty else {"error": "no data"}

def _correlation_matrix(**kwargs) -> dict:
    """Retrieve 6x6 return correlation matrix."""
    from knowledge.shared.queries import correlation_matrix
    df = correlation_matrix(**kwargs)
    return df.to_dict() if not df.empty else {"error": "no data"}

def _devil_summary(**kwargs) -> dict:
    """Retrieve Devil Tracker cost summary."""
    from knowledge.shared.queries import devil_tracker_summary
    return devil_tracker_summary(**kwargs)

def _regime_history(**kwargs) -> dict:
    """Retrieve regime distribution and transitions."""
    from knowledge.shared.queries import regime_history
    return regime_history(**kwargs)

def _load_pair_returns(pair: str = "BTC-USD") -> dict:
    """Load returns for a pair from training data."""
    try:
        from knowledge.shared.data_loader import load_pair_csv, get_returns
        df = load_pair_csv(pair, nrows=2000)
        r = get_returns(df)
        return {"pair": pair, "n_bars": len(r), "mean": round(float(r.mean()), 6),
                "std": round(float(r.std()), 6), "skew": round(float(r.skew()), 4),
                "kurt": round(float(r.kurtosis()), 4)}
    except Exception as e:
        return {"error": str(e)}

def _aligned_returns_summary() -> dict:
    """Load aligned returns and compute summary statistics."""
    try:
        from knowledge.shared.data_loader import get_aligned_returns
        df = get_aligned_returns(nrows=2000)
        return {"n_bars": len(df), "pairs": list(df.columns),
                "correlation": df.corr().to_dict()}
    except Exception as e:
        return {"error": str(e)}

def _dead_end_check(description: str = "") -> dict:
    """Check proposal against known dead ends."""
    from knowledge.shared.dead_ends import is_dead_end, get_dead_ends_for
    matches = is_dead_end(description)
    return {"description": description, "dead_end_matches": matches,
            "is_dead_end": len(matches) > 0}


KB.register_many([
    Atom("shared.weekly_performance", "Weekly Performance Query", "shared",
         _weekly_performance.__doc__, atype=AType.QUERY|AType.COMPUTED,
         affects=Affects.MODELS|Affects.SIGNALS|Affects.SIZING, formula=_weekly_performance),
    Atom("shared.model_accuracy", "Model Accuracy Matrix", "shared",
         _model_accuracy.__doc__, atype=AType.QUERY|AType.COMPUTED,
         affects=Affects.MODELS, formula=_model_accuracy),
    Atom("shared.correlation_matrix", "Return Correlation Matrix", "shared",
         _correlation_matrix.__doc__, atype=AType.QUERY|AType.COMPUTED,
         affects=Affects.SIZING|Affects.RISK, formula=_correlation_matrix),
    Atom("shared.devil_summary", "Devil Tracker Summary", "shared",
         _devil_summary.__doc__, atype=AType.QUERY|AType.COMPUTED,
         affects=Affects.COST|Affects.EXECUTION, formula=_devil_summary),
    Atom("shared.regime_history", "Regime History Query", "shared",
         _regime_history.__doc__, atype=AType.QUERY|AType.COMPUTED,
         affects=Affects.REGIME, formula=_regime_history),
    Atom("shared.pair_returns", "Load Pair Returns", "shared",
         _load_pair_returns.__doc__, atype=AType.QUERY,
         affects=Affects.FEATURES, formula=_load_pair_returns),
    Atom("shared.aligned_returns", "Aligned Multi-Pair Returns", "shared",
         _aligned_returns_summary.__doc__, atype=AType.QUERY,
         affects=Affects.SIZING|Affects.RISK, formula=_aligned_returns_summary),
    Atom("shared.dead_end_check", "Dead End Checker", "shared",
         _dead_end_check.__doc__, atype=AType.QUERY,
         affects=Affects.MODELS|Affects.FEATURES|Affects.SIGNALS, formula=_dead_end_check),
    # Reference atoms
    Atom("shared.feature_groups", "Feature Group Map", "shared",
         "98-dim feature vector groups: candle_shape(0-5), returns(5-12), sma(12-20), "
         "ema(20-26), technical(26-31), volatility(31-35), volume(35-40), momentum(40-43), "
         "microstructure(43-46), cross_asset(46-61), derivatives(61-68), padding(68-98).",
         atype=AType.REFERENCE, affects=Affects.FEATURES,
         value={"candle_shape":[0,5],"returns":[5,12],"sma":[12,20],"ema":[20,26],
                "technical":[26,31],"volatility":[31,35],"volume":[35,40],"momentum":[40,43],
                "microstructure":[43,46],"cross_asset":[46,61],"derivatives":[61,68],"padding":[68,98]}),
    Atom("shared.pair_universe", "Trading Pair Universe", "shared",
         "6 core pairs: BTC-USD, ETH-USD, SOL-USD, DOGE-USD, AVAX-USD, LINK-USD. "
         "Dynamic universe adds ~70-90 Binance USDT pairs with >$2M daily volume.",
         atype=AType.REFERENCE, affects=Affects.SIGNALS,
         value={"core": ["BTC-USD","ETH-USD","SOL-USD","DOGE-USD","AVAX-USD","LINK-USD"],
                "dynamic_min_volume": 2_000_000}),
])
