"""Linguist's knowledge atoms — ML calibration, ensemble analysis, feature engineering."""
import numpy as np
from knowledge.registry import KB, Atom, Pair, Regime, Scale, AType, Affects


def vc_dimension_check(n_params: int, n_samples: int) -> dict:
    """Is the model overparameterized? Rule of thumb: need n_samples >= 10 * VC_dim.
    For neural nets, VC_dim ~ O(n_params * log(n_params)).
    If ratio < 5, high overfitting risk."""
    vc_approx = n_params * np.log(max(n_params, 2))
    ratio = n_samples / max(vc_approx, 1)
    return {"vc_dim_approx": round(vc_approx, 0), "n_samples": n_samples,
            "ratio": round(ratio, 2), "overparameterized": ratio < 5,
            "safe": ratio >= 10, "recommendation": "reduce model" if ratio < 5 else "OK"}


def rademacher_complexity(predictions, labels, n_trials: int = 100) -> dict:
    """Empirical Rademacher complexity — overfitting diagnostic.
    Random label accuracy near actual accuracy = model is memorizing, not learning."""
    preds = np.array(predictions)
    labs = np.array(labels)
    actual_acc = float(np.mean((preds > 0.5) == labs))
    random_accs = []
    for _ in range(n_trials):
        random_labels = np.random.choice([0, 1], size=len(labs))
        random_accs.append(float(np.mean((preds > 0.5) == random_labels)))
    mean_random = float(np.mean(random_accs))
    gap = actual_acc - mean_random
    return {"actual_accuracy": round(actual_acc, 4),
            "random_label_accuracy": round(mean_random, 4),
            "gap": round(gap, 4),
            "overfitting_risk": gap < 0.02}


def ensemble_correlation(predictions_dict) -> dict:
    """Pairwise correlation between model predictions.
    High correlation (>0.8) means models are redundant.
    Ideal ensemble: accuracy > 0.5 AND correlation < 0.5."""
    names = list(predictions_dict.keys())
    preds = np.array([predictions_dict[n] for n in names])
    corr = np.corrcoef(preds)
    n = len(names)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    avg_corr = float(np.mean(corr[mask]))
    pairs = {}
    for i in range(n):
        for j in range(i+1, n):
            pairs[f"{names[i]}_vs_{names[j]}"] = round(float(corr[i, j]), 4)
    return {"avg_correlation": round(avg_corr, 4),
            "pairwise": pairs,
            "too_correlated": avg_corr > 0.8,
            "good_diversity": avg_corr < 0.5}


def condorcet_effective(n_models: int, avg_correlation: float) -> dict:
    """Effective jury size for ensemble via Condorcet theorem.
    With correlation rho, effective N = N / (1 + (N-1)*rho).
    6 models at rho=0.7 -> only 1.5 effective models."""
    n_eff = n_models / (1 + (n_models - 1) * avg_correlation)
    benefit = n_eff > 1.5
    return {"n_models": n_models, "avg_correlation": round(avg_correlation, 4),
            "n_effective": round(n_eff, 2),
            "beneficial": benefit,
            "recommendation": "add diverse model" if n_eff < 2 else "ensemble OK"}


def platt_calibrate(scores, labels) -> dict:
    """Platt scaling: fit sigmoid to convert raw scores to calibrated probabilities.
    P(y=1|s) = 1 / (1 + exp(A*s + B)). Fit A, B by maximum likelihood."""
    from scipy.optimize import minimize
    s = np.array(scores, dtype=float)
    y = np.array(labels, dtype=float)
    def neg_ll(params):
        a, b = params
        p = 1.0 / (1.0 + np.exp(a * s + b))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    res = minimize(neg_ll, [0.0, 0.0], method='Nelder-Mead')
    A, B = res.x
    calibrated = 1.0 / (1.0 + np.exp(A * s + B))
    return {"A": round(float(A), 4), "B": round(float(B), 4),
            "calibrated_mean": round(float(np.mean(calibrated)), 4),
            "calibrated_std": round(float(np.std(calibrated)), 4)}


def isotonic_calibrate(scores, labels) -> dict:
    """Isotonic regression calibration — non-parametric, monotonic.
    Better than Platt when relationship is non-sigmoidal."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds='clip')
    s = np.array(scores)
    y = np.array(labels)
    ir.fit(s, y)
    calibrated = ir.predict(s)
    return {"calibrated_mean": round(float(np.mean(calibrated)), 4),
            "calibrated_std": round(float(np.std(calibrated)), 4),
            "n_breakpoints": len(ir.X_thresholds_) if hasattr(ir, 'X_thresholds_') else 0}


def expected_calibration_error(predicted_probs, actual_outcomes, n_bins: int = 10) -> dict:
    """ECE: how well-calibrated are the model's confidence scores?
    Perfect calibration: ECE = 0. ECE > 0.05 needs recalibration."""
    probs = np.array(predicted_probs)
    outcomes = np.array(actual_outcomes)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        avg_conf = float(np.mean(probs[mask]))
        avg_acc = float(np.mean(outcomes[mask]))
        count = int(mask.sum())
        ece += count * abs(avg_conf - avg_acc)
        bin_data.append({"bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                         "confidence": round(avg_conf, 4),
                         "accuracy": round(avg_acc, 4),
                         "count": count})
    ece /= max(len(probs), 1)
    return {"ece": round(ece, 4), "bins": bin_data,
            "well_calibrated": ece < 0.05, "needs_recalibration": ece > 0.10}


def feature_staleness(feature_series, lag: int = 1) -> dict:
    """Detect stale features via autocorrelation.
    AC(1) > 0.99 means feature barely changes — likely stale/broken."""
    s = np.array(feature_series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 10:
        return {"ac1": 0.0, "stale": False}
    ac1 = float(np.corrcoef(s[:-lag], s[lag:])[0, 1])
    unique_ratio = len(np.unique(s)) / max(len(s), 1)
    return {"ac1": round(ac1, 4), "unique_ratio": round(unique_ratio, 4),
            "stale": ac1 > 0.99 or unique_ratio < 0.01,
            "low_variance": unique_ratio < 0.05}


def feature_importance_gain(importances_dict) -> dict:
    """Analyze LightGBM feature importances (gain-based).
    Top 10% of features typically carry 80%+ of information."""
    names = list(importances_dict.keys())
    values = np.array([importances_dict[n] for n in names])
    total = values.sum()
    if total == 0:
        return {"error": "all importances zero"}
    sorted_idx = np.argsort(values)[::-1]
    cumulative = np.cumsum(values[sorted_idx]) / total
    top_10_pct = max(1, len(names) // 10)
    top_10_fraction = float(cumulative[min(top_10_pct, len(cumulative)-1)])
    return {"top_features": [(names[i], round(float(values[i]/total), 4)) for i in sorted_idx[:10]],
            "top_10pct_explains": round(top_10_fraction, 4),
            "n_features": len(names),
            "concentrated": top_10_fraction > 0.8}


KB.register_many([
    Atom("ling.vc_dimension", "VC Dimension Check", "linguist",
         vc_dimension_check.__doc__, atype=AType.FORMULA, affects=Affects.MODELS, formula=vc_dimension_check),
    Atom("ling.rademacher", "Rademacher Complexity", "linguist",
         rademacher_complexity.__doc__, atype=AType.FORMULA, affects=Affects.MODELS, formula=rademacher_complexity),
    Atom("ling.ensemble_corr", "Ensemble Correlation", "linguist",
         ensemble_correlation.__doc__, atype=AType.FORMULA, affects=Affects.MODELS|Affects.SIGNALS, formula=ensemble_correlation),
    Atom("ling.condorcet", "Condorcet Effective Jury", "linguist",
         condorcet_effective.__doc__, atype=AType.FORMULA, affects=Affects.MODELS, formula=condorcet_effective),
    Atom("ling.platt_calibrate", "Platt Scaling", "linguist",
         platt_calibrate.__doc__, atype=AType.FORMULA, affects=Affects.MODELS|Affects.SIGNALS, formula=platt_calibrate),
    Atom("ling.isotonic_calibrate", "Isotonic Calibration", "linguist",
         isotonic_calibrate.__doc__, atype=AType.FORMULA, affects=Affects.MODELS, formula=isotonic_calibrate),
    Atom("ling.ece", "Expected Calibration Error", "linguist",
         expected_calibration_error.__doc__, atype=AType.FORMULA, affects=Affects.MODELS|Affects.SIGNALS, formula=expected_calibration_error),
    Atom("ling.feature_staleness", "Feature Staleness Detector", "linguist",
         feature_staleness.__doc__, atype=AType.FORMULA, affects=Affects.FEATURES, formula=feature_staleness),
    Atom("ling.feature_importance", "Feature Importance Analysis", "linguist",
         feature_importance_gain.__doc__, atype=AType.FORMULA, affects=Affects.FEATURES|Affects.MODELS, formula=feature_importance_gain),
    # Thresholds
    Atom("ling.accuracy_benchmarks", "Model Accuracy Benchmarks", "linguist",
         "Random=50%, minimum viable=51.5%, good=53%, excellent=55%. "
         "More than 55% at 5-min resolution is suspicious (likely overfit).",
         atype=AType.THRESHOLD, affects=Affects.MODELS,
         value={"random": 0.50, "minimum": 0.515, "good": 0.53, "excellent": 0.55, "suspicious": 0.60}),
    Atom("ling.ensemble_rules", "Ensemble Decision Rules", "linguist",
         "Majority vote for direction. Confidence = agreement fraction. "
         "Weight LightGBM 2x in sideways regimes. Weight DL 2x in trending.",
         atype=AType.REFERENCE, affects=Affects.MODELS|Affects.SIGNALS,
         value={"lgbm_boost_regime": "sideways", "dl_boost_regime": "trending", "boost_factor": 2.0}),
    # Crypto-specific
    Atom("ling.crypto_features", "Crypto-Specific Feature Groups", "linguist",
         "Active: candle_shape(0-5), returns(5-12), sma(12-20), ema(20-26), technical(26-31), "
         "volatility(31-35), volume(35-40), momentum(40-43), microstructure(43-46). "
         "Inactive: cross_asset(46-61), derivatives(61-68). Padding(68-98).",
         atype=AType.REFERENCE, affects=Affects.FEATURES, crypto_specific=True,
         value={"active_dims": 46, "total_dims": 98, "inactive_start": 46}),
])
