"""Physicist's knowledge atoms — dynamical systems, data quality, phase transitions."""
import numpy as np
from knowledge.registry import KB, Atom, Pair, Regime, Scale, AType, Affects


def hurst_exponent(series, method: str = "dfa") -> dict:
    """Hurst exponent H: H=0.5 random walk, H>0.5 trending, H<0.5 mean-reverting.
    Uses Detrended Fluctuation Analysis (DFA) for robustness.
    For crypto 5-min bars: H typically 0.45-0.55 (near random)."""
    s = np.array(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 100:
        return {"H": 0.5, "error": "too few samples", "ci": [0, 1]}
    if method == "dfa":
        y = np.cumsum(s - np.mean(s))
        scales = np.unique(np.logspace(np.log10(4), np.log10(len(s)//4), 20).astype(int))
        flucts = []
        for sc in scales:
            segments = len(y) // sc
            if segments < 1:
                continue
            f2 = 0.0
            for seg in range(segments):
                chunk = y[seg*sc:(seg+1)*sc]
                x = np.arange(sc)
                coef = np.polyfit(x, chunk, 1)
                trend = np.polyval(coef, x)
                f2 += np.mean((chunk - trend)**2)
            flucts.append(np.sqrt(f2 / segments))
        if len(flucts) < 3:
            return {"H": 0.5, "error": "insufficient scale range"}
        log_s = np.log(scales[:len(flucts)])
        log_f = np.log(np.array(flucts))
        H = float(np.polyfit(log_s, log_f, 1)[0])
    else:
        # R/S method fallback
        lags = range(2, min(100, len(s)//2))
        rs_vals = []
        for lag in lags:
            rs_sub = []
            for start in range(0, len(s) - lag, lag):
                chunk = s[start:start+lag]
                mean_chunk = np.mean(chunk)
                devs = np.cumsum(chunk - mean_chunk)
                R = np.max(devs) - np.min(devs)
                S = np.std(chunk)
                if S > 0:
                    rs_sub.append(R / S)
            if rs_sub:
                rs_vals.append((lag, np.mean(rs_sub)))
        if len(rs_vals) < 3:
            return {"H": 0.5, "error": "insufficient data"}
        log_n = np.log([v[0] for v in rs_vals])
        log_rs = np.log([v[1] for v in rs_vals])
        H = float(np.polyfit(log_n, log_rs, 1)[0])

    regime = "trending" if H > 0.55 else "mean_reverting" if H < 0.45 else "random_walk"
    return {"H": round(H, 4), "regime": regime, "trending": H > 0.55,
            "mean_reverting": H < 0.45, "method": method}


def hurst_multiscale(series, scales=None) -> dict:
    """H at multiple timescales to detect scale-dependent behavior.
    A market can be mean-reverting at 5-min but trending at 4-hour."""
    s = np.array(series, dtype=float)
    s = s[~np.isnan(s)]
    if scales is None:
        scales = [1, 6, 12, 48, 288]  # 5m, 30m, 1h, 4h, 1d
    results = {}
    for sc in scales:
        if sc >= len(s) // 4:
            continue
        agg = np.array([np.mean(s[i:i+sc]) for i in range(0, len(s)-sc, sc)])
        if len(agg) < 50:
            continue
        h = hurst_exponent(agg, method="dfa")
        results[f"scale_{sc}"] = h.get("H", 0.5)
    return {"by_scale": results, "scale_dependent": max(results.values(), default=0.5) - min(results.values(), default=0.5) > 0.15 if results else False}


def lyapunov_exponent(series) -> dict:
    """Largest Lyapunov exponent for prediction horizon estimation.
    Lambda > 0: chaotic (predictability decays exponentially).
    Prediction horizon ~ 1/lambda time steps."""
    s = np.array(series, dtype=float)
    s = s[~np.isnan(s)]
    n = len(s)
    if n < 100:
        return {"lambda": 0.0, "error": "too few samples"}
    embedding_dim = 3
    tau = 1
    vectors = np.array([s[i:i+embedding_dim*tau:tau] for i in range(n - embedding_dim*tau)])
    divergences = []
    for i in range(len(vectors) - 1):
        dists = np.linalg.norm(vectors - vectors[i], axis=1)
        dists[i] = float('inf')
        j = np.argmin(dists)
        if j < len(vectors) - 1:
            d0 = max(dists[j], 1e-12)
            d1 = np.linalg.norm(vectors[min(i+1, len(vectors)-1)] - vectors[min(j+1, len(vectors)-1)])
            if d1 > 0:
                divergences.append(np.log(d1 / d0))
    lam = float(np.mean(divergences)) if divergences else 0.0
    horizon = 1.0 / max(abs(lam), 1e-6)
    return {"lambda": round(lam, 4), "chaotic": lam > 0.05,
            "prediction_horizon_bars": round(horizon, 1)}


def tail_index(returns) -> dict:
    """Hill estimator for tail index alpha.
    alpha < 2: infinite variance (heavy tails). alpha < 3: infinite kurtosis.
    Crypto typically alpha ~ 2.5-3.5 (heavier than Gaussian)."""
    r = np.abs(np.array(returns, dtype=float))
    r = r[r > 0]
    r = np.sort(r)[::-1]
    if len(r) < 50:
        return {"alpha": 4.0, "error": "too few samples"}
    k = max(int(len(r) * 0.05), 10)
    threshold = r[k]
    tail = r[:k]
    alpha = float(k / np.sum(np.log(tail / threshold)))
    return {"alpha": round(alpha, 4), "k_tail": k, "threshold": round(float(threshold), 6),
            "infinite_variance": alpha < 2, "heavy_tails": alpha < 3.5,
            "gaussian_like": alpha > 4}


def realized_vol_ratio(df, fast_bars: int = 6, slow_bars: int = 288) -> dict:
    """Ratio of fast-scale to slow-scale realized volatility.
    Ratio > 1: microstructure noise or regime change. Ratio < 0.8: mean-reverting intraday."""
    close = np.array(df["close"] if hasattr(df, "columns") else df, dtype=float)
    returns = np.diff(np.log(close))
    if len(returns) < slow_bars:
        return {"ratio": 1.0, "error": "insufficient data"}
    fast_vol = np.std(returns[-fast_bars:]) * np.sqrt(fast_bars)
    slow_vol = np.std(returns[-slow_bars:]) * np.sqrt(slow_bars)
    ratio = fast_vol / max(slow_vol, 1e-12)
    return {"ratio": round(float(ratio), 4), "fast_vol": round(float(fast_vol), 6),
            "slow_vol": round(float(slow_vol), 6),
            "microstructure_noise": ratio > 1.5, "mean_reverting": ratio < 0.8}


def adf_test(series) -> dict:
    """Augmented Dickey-Fuller test for stationarity.
    p < 0.05: stationary (good for mean reversion). p > 0.10: unit root (trending)."""
    from scipy import stats
    s = np.array(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 30:
        return {"stationary": False, "error": "too few samples"}
    ds = np.diff(s)
    s_lagged = s[:-1]
    n = len(ds)
    X = np.column_stack([s_lagged, np.ones(n)])
    beta = np.linalg.lstsq(X, ds, rcond=None)[0]
    residuals = ds - X @ beta
    se = np.sqrt(float(np.sum(residuals**2) / (n - 2)) / float(np.sum((s_lagged - np.mean(s_lagged))**2)))
    t_stat = float(beta[0] / max(se, 1e-12))
    critical = {1: -3.43, 5: -2.86, 10: -2.57}
    return {"t_stat": round(t_stat, 4), "critical_values": critical,
            "stationary_1pct": t_stat < critical[1],
            "stationary_5pct": t_stat < critical[5],
            "stationary_10pct": t_stat < critical[10]}


def kpss_test(series) -> dict:
    """KPSS test — complementary to ADF. H0: stationary.
    If ADF rejects AND KPSS rejects: difference-stationary (trend).
    If ADF rejects AND KPSS doesn't reject: stationary."""
    s = np.array(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 30:
        return {"stationary": True, "error": "too few samples"}
    n = len(s)
    e = s - np.mean(s)
    cumsum = np.cumsum(e)
    s2 = float(np.sum(e**2) / n)
    stat = float(np.sum(cumsum**2) / (n**2 * max(s2, 1e-12)))
    critical = {10: 0.347, 5: 0.463, 1: 0.739}
    return {"kpss_stat": round(stat, 4), "critical_values": critical,
            "stationary": stat < critical[5]}


def percolation_fraction(corr_matrix, threshold: float = 0.5) -> dict:
    """Fraction of pairs above correlation threshold — cascade risk proxy.
    At percolation threshold (~0.5), connected cluster spans all assets."""
    c = np.abs(np.array(corr_matrix))
    n = c.shape[0]
    np.fill_diagonal(c, 0)
    above = c > threshold
    fraction = float(above.sum()) / (n * (n - 1))
    max_corr = float(c.max())
    mean_corr = float(c.sum() / (n * (n - 1)))
    return {"fraction_above": round(fraction, 4), "threshold": threshold,
            "max_corr": round(max_corr, 4), "mean_abs_corr": round(mean_corr, 4),
            "cascade_risk": fraction > 0.5}


def eigenvalue_spectrum(corr_matrix, n_samples: int) -> dict:
    """Eigenvalue spectrum with Marchenko-Pastur comparison.
    Eigenvalues above MP upper bound carry signal; below are noise."""
    c = np.array(corr_matrix)
    n = c.shape[0]
    q = n / n_samples
    lp = (1 + np.sqrt(q))**2
    lm = (1 - np.sqrt(q))**2
    eig = np.sort(np.linalg.eigvalsh(c))[::-1]
    signal = [float(e) for e in eig if e > lp]
    noise = [float(e) for e in eig if e <= lp]
    return {"eigenvalues": np.round(eig, 4).tolist(),
            "mp_bounds": [round(lm, 4), round(lp, 4)],
            "n_signal": len(signal), "n_noise": len(noise),
            "explained_by_signal": round(sum(signal) / sum(eig), 4) if sum(eig) > 0 else 0}


def mahalanobis_outlier(features, threshold: float = 3.0) -> dict:
    """Multivariate outlier detection using Mahalanobis distance.
    Points with D > threshold are anomalies in feature space."""
    X = np.array(features, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mu = np.mean(X, axis=0)
    cov = np.cov(X.T)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
    diff = X - mu
    d = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
    outliers = np.where(d > threshold)[0]
    return {"n_outliers": len(outliers), "outlier_indices": outliers.tolist(),
            "max_distance": round(float(d.max()), 4),
            "mean_distance": round(float(d.mean()), 4),
            "fraction_outlier": round(len(outliers) / len(d), 4)}


def check_data_quality(pair: str, df) -> dict:
    """Comprehensive data quality check: gaps, anomalies, staleness."""
    import pandas as pd
    if hasattr(df, 'index') and isinstance(df.index, pd.DatetimeIndex):
        dt = df.index.to_series().diff()
        expected = pd.Timedelta(minutes=5)
        gaps = dt[dt > expected * 2]
        gap_count = len(gaps)
        max_gap = str(gaps.max()) if len(gaps) > 0 else "0"
    else:
        gap_count = 0
        max_gap = "unknown"
    close = np.array(df["close"] if "close" in df.columns else df, dtype=float)
    returns = np.diff(np.log(close[close > 0]))
    spikes = np.abs(returns) > 0.05  # >5% in 5min
    zero_vol = (np.array(df["volume"] if "volume" in df.columns else [1]) == 0).sum()
    return {"pair": pair, "n_bars": len(df), "gaps": gap_count, "max_gap": max_gap,
            "price_spikes_5pct": int(spikes.sum()), "zero_volume_bars": int(zero_vol),
            "quality_score": round(1.0 - (gap_count + int(spikes.sum())) / max(len(df), 1), 4)}


KB.register_many([
    Atom("phys.hurst", "Hurst Exponent (DFA)", "physicist",
         hurst_exponent.__doc__, atype=AType.FORMULA, affects=Affects.REGIME|Affects.SIGNALS,
         scales=Scale.ALL, formula=hurst_exponent),
    Atom("phys.hurst_multiscale", "Multi-Scale Hurst", "physicist",
         hurst_multiscale.__doc__, atype=AType.FORMULA, affects=Affects.REGIME, formula=hurst_multiscale),
    Atom("phys.lyapunov", "Lyapunov Exponent", "physicist",
         lyapunov_exponent.__doc__, atype=AType.FORMULA, affects=Affects.MODELS, formula=lyapunov_exponent),
    Atom("phys.tail_index", "Hill Tail Index", "physicist",
         tail_index.__doc__, atype=AType.FORMULA, affects=Affects.RISK|Affects.SIZING, formula=tail_index),
    Atom("phys.realized_vol_ratio", "Realized Vol Ratio", "physicist",
         realized_vol_ratio.__doc__, atype=AType.FORMULA, affects=Affects.REGIME|Affects.SIGNALS, formula=realized_vol_ratio),
    Atom("phys.adf_test", "ADF Stationarity Test", "physicist",
         adf_test.__doc__, atype=AType.FORMULA, affects=Affects.REGIME|Affects.FEATURES, formula=adf_test),
    Atom("phys.kpss_test", "KPSS Stationarity Test", "physicist",
         kpss_test.__doc__, atype=AType.FORMULA, affects=Affects.REGIME, formula=kpss_test),
    Atom("phys.percolation", "Percolation Fraction", "physicist",
         percolation_fraction.__doc__, atype=AType.FORMULA, affects=Affects.RISK, formula=percolation_fraction),
    Atom("phys.eigenvalue_spectrum", "Eigenvalue Spectrum", "physicist",
         eigenvalue_spectrum.__doc__, atype=AType.FORMULA, affects=Affects.SIZING|Affects.RISK, formula=eigenvalue_spectrum),
    Atom("phys.mahalanobis", "Mahalanobis Outlier Detection", "physicist",
         mahalanobis_outlier.__doc__, atype=AType.FORMULA, affects=Affects.FEATURES|Affects.RISK, formula=mahalanobis_outlier),
    Atom("phys.data_quality", "Data Quality Check", "physicist",
         check_data_quality.__doc__, atype=AType.COMPUTED, affects=Affects.FEATURES, formula=check_data_quality),
    # Crypto-specific physics
    Atom("phys.liquidation_cascade", "Liquidation Cascade Model", "physicist",
         "BTC drop -> altcoin correlation spike -> forced liquidations -> positive feedback. "
         "Detection: OI declining + price declining + correlation > 0.95.",
         atype=AType.REFERENCE, affects=Affects.RISK, crypto_specific=True,
         value={"correlation_threshold": 0.95, "oi_decline_trigger": -0.02}),
    Atom("phys.hurst_crypto_benchmark", "Crypto Hurst Benchmarks", "physicist",
         "BTC 5-min: H ~ 0.48-0.52. ETH: 0.46-0.50. Altcoins: more variable. "
         "H < 0.45 = strong mean reversion signal. H > 0.55 = momentum.",
         atype=AType.REFERENCE, affects=Affects.REGIME,
         value={"btc_range": [0.48, 0.52], "eth_range": [0.46, 0.50], "signal_threshold": 0.05}),
])
