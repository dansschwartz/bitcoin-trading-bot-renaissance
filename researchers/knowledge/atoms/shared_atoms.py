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


# ── Audit query wrappers ──

def _audit_model_accuracy(**kwargs) -> dict:
    """Per-model live accuracy from decision_audit_log + ml_predictions."""
    from knowledge.shared.queries import audit_model_accuracy
    from knowledge._base import PROJECT_ROOT
    return audit_model_accuracy(str(PROJECT_ROOT / "data" / "renaissance_bot.db"), **kwargs)

def _audit_signal_effectiveness(**kwargs) -> dict:
    """Signal separation analysis from decision_audit_log."""
    from knowledge.shared.queries import audit_signal_effectiveness
    from knowledge._base import PROJECT_ROOT
    return audit_signal_effectiveness(str(PROJECT_ROOT / "data" / "renaissance_bot.db"), **kwargs)

def _audit_regime_performance(**kwargs) -> dict:
    """P&L and accuracy by regime from decision_audit_log."""
    from knowledge.shared.queries import audit_regime_performance
    from knowledge._base import PROJECT_ROOT
    return audit_regime_performance(str(PROJECT_ROOT / "data" / "renaissance_bot.db"), **kwargs)

def _audit_sizing_chain(**kwargs) -> dict:
    """Sizing chain analysis from decision_audit_log."""
    from knowledge.shared.queries import audit_sizing_chain_analysis
    from knowledge._base import PROJECT_ROOT
    return audit_sizing_chain_analysis(str(PROJECT_ROOT / "data" / "renaissance_bot.db"), **kwargs)

def _audit_cost_vs_edge(**kwargs) -> dict:
    """Devil cost analysis from decision_audit_log."""
    from knowledge.shared.queries import audit_cost_vs_edge
    from knowledge._base import PROJECT_ROOT
    return audit_cost_vs_edge(str(PROJECT_ROOT / "data" / "renaissance_bot.db"), **kwargs)

def _audit_confluence(**kwargs) -> dict:
    """Confluence boost effectiveness from decision_audit_log."""
    from knowledge.shared.queries import audit_confluence_effectiveness
    from knowledge._base import PROJECT_ROOT
    return audit_confluence_effectiveness(str(PROJECT_ROOT / "data" / "renaissance_bot.db"), **kwargs)

def _audit_feature_health(**kwargs) -> dict:
    """Feature vector health metrics from decision_audit_log."""
    from knowledge.shared.queries import audit_feature_health
    from knowledge._base import PROJECT_ROOT
    return audit_feature_health(str(PROJECT_ROOT / "data" / "renaissance_bot.db"), **kwargs)


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
    # ── Audit query atoms ──
    Atom("shared.audit_model_accuracy", "Audit: Model Accuracy", "shared",
         _audit_model_accuracy.__doc__, atype=AType.QUERY|AType.COMPUTED,
         affects=Affects.MODELS, formula=_audit_model_accuracy),
    Atom("shared.audit_signal_effectiveness", "Audit: Signal Effectiveness", "shared",
         _audit_signal_effectiveness.__doc__, atype=AType.QUERY|AType.COMPUTED,
         affects=Affects.SIGNALS, formula=_audit_signal_effectiveness),
    Atom("shared.audit_regime_performance", "Audit: Regime Performance", "shared",
         _audit_regime_performance.__doc__, atype=AType.QUERY|AType.COMPUTED,
         affects=Affects.REGIME|Affects.SIZING, formula=_audit_regime_performance),
    Atom("shared.audit_sizing_chain", "Audit: Sizing Chain", "shared",
         _audit_sizing_chain.__doc__, atype=AType.QUERY|AType.COMPUTED,
         affects=Affects.SIZING, formula=_audit_sizing_chain),
    Atom("shared.audit_cost_vs_edge", "Audit: Cost vs Edge", "shared",
         _audit_cost_vs_edge.__doc__, atype=AType.QUERY|AType.COMPUTED,
         affects=Affects.COST|Affects.EXECUTION, formula=_audit_cost_vs_edge),
    Atom("shared.audit_confluence", "Audit: Confluence Effectiveness", "shared",
         _audit_confluence.__doc__, atype=AType.QUERY|AType.COMPUTED,
         affects=Affects.SIGNALS, formula=_audit_confluence),
    Atom("shared.audit_feature_health", "Audit: Feature Health", "shared",
         _audit_feature_health.__doc__, atype=AType.QUERY|AType.COMPUTED,
         affects=Affects.FEATURES, formula=_audit_feature_health),
])

# ── Regime Config Registry atoms ──
try:
    from knowledge.regime_registry import REGIME_CONFIG

    KB.register(Atom(
        "shared.regime_config_manifest", "Regime Config Registry Manifest", "shared",
        "Complete listing of all regime-conditional parameters.\n"
        "Use: print(REGIME_CONFIG.manifest())\n"
        "Query: REGIME_CONFIG.get('sizing.regime_scalar', regime='trending')\n"
        "All entries: REGIME_CONFIG.get_all(ParamType.SIZING, regime=Regime.TRENDING)",
        atype=AType.REFERENCE,
        affects=Affects.SIZING | Affects.SIGNALS | Affects.RISK,
        value=REGIME_CONFIG,
    ))

    KB.register(Atom(
        "shared.regime_config_propose", "Propose Regime Config Change", "shared",
        "Structured way to propose regime parameter changes.\n"
        "Instead of: 'edit position_sizer.py line 47'\n"
        "Use: REGIME_CONFIG.propose_change(\n"
        "    key='sizing.regime_scalar', regime='trending',\n"
        "    current_value=1.20, proposed_value=1.05,\n"
        "    rationale='Audit data shows trending returns don't justify 1.2x'\n"
        ")",
        atype=AType.REFERENCE,
        affects=Affects.SIZING | Affects.SIGNALS,
        formula=REGIME_CONFIG.propose_change,
    ))
except ImportError:
    pass
