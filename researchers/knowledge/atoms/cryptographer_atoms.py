"""Cryptographer's knowledge atoms — HMM, spectral, sequence analysis."""
import numpy as np
from knowledge.registry import KB, Atom, Pair, Regime, Scale, AType, Affects


def hmm_bic(log_likelihood: float, n_states: int, n_features: int, n_samples: int) -> dict:
    """Bayesian Information Criterion for HMM model selection.
    BIC = -2*LL + k*ln(n). Lower is better.
    k = n_states^2 + n_states*n_features*2 - 1 (transitions + means + variances)."""
    k = n_states**2 + n_states * n_features * 2 - 1
    bic = -2 * log_likelihood + k * np.log(n_samples)
    aic = -2 * log_likelihood + 2 * k
    return {"bic": round(bic, 2), "aic": round(aic, 2), "n_params": k,
            "samples_per_param": round(n_samples / k, 1)}


def hmm_transition_eigenvalues(transition_matrix) -> dict:
    """Eigenvalues of HMM transition matrix reveal mixing time and state stickiness.
    Second-largest eigenvalue lambda_2 controls mixing: t_mix ~ -1/ln(|lambda_2|).
    High stickiness (lambda_2 > 0.95) means slow regime changes."""
    T = np.array(transition_matrix)
    eig = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
    lam2 = float(eig[1]) if len(eig) > 1 else 0.0
    mixing_time = -1.0 / np.log(max(abs(lam2), 1e-10)) if abs(lam2) < 1 else float('inf')
    stickiness = np.mean(np.diag(T))
    return {"eigenvalues": np.round(eig, 4).tolist(),
            "lambda_2": round(lam2, 4),
            "mixing_time_steps": round(mixing_time, 1),
            "avg_stickiness": round(float(stickiness), 4),
            "too_sticky": lam2 > 0.95,
            "too_fast": lam2 < 0.5}


def snr_estimate(signal_series) -> dict:
    """Signal-to-noise ratio estimate using autocorrelation method.
    SNR = var(signal) / var(noise). For crypto features: SNR < 0.01 is common."""
    s = np.array(signal_series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 10:
        return {"snr": 0.0, "error": "too few samples"}
    ac1 = float(np.corrcoef(s[:-1], s[1:])[0, 1])
    total_var = float(np.var(s))
    signal_var = max(ac1 * total_var, 0)
    noise_var = total_var - signal_var
    snr = signal_var / max(noise_var, 1e-12)
    return {"snr": round(snr, 6), "ac1": round(ac1, 4),
            "signal_var": round(signal_var, 8), "noise_var": round(noise_var, 8),
            "usable": snr > 0.01, "strong": snr > 0.1}


def spectral_density(series, bar_minutes: int = 5) -> dict:
    """Power spectral density with crypto cycle identification.
    Key frequencies: 8h (funding), 24h (daily), 168h (weekly)."""
    s = np.array(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 64:
        return {"error": "need >= 64 samples"}
    s = s - np.mean(s)
    freqs = np.fft.rfftfreq(len(s), d=bar_minutes * 60)
    psd = np.abs(np.fft.rfft(s))**2 / len(s)
    crypto_cycles = {
        "8h_funding": 1.0 / (8 * 3600),
        "24h_daily": 1.0 / (24 * 3600),
        "168h_weekly": 1.0 / (168 * 3600),
    }
    detected = {}
    for name, target_freq in crypto_cycles.items():
        idx = np.argmin(np.abs(freqs - target_freq))
        if idx > 0 and idx < len(psd):
            local_power = float(psd[idx])
            bg_power = float(np.median(psd[max(0, idx-5):idx+5]))
            detected[name] = {"power": round(local_power, 4),
                              "ratio_to_background": round(local_power / max(bg_power, 1e-12), 2)}
    peak_idx = np.argsort(psd[1:])[-5:] + 1
    dominant = [{"freq_hz": round(float(freqs[i]), 8),
                 "period_hours": round(1.0 / max(float(freqs[i]), 1e-12) / 3600, 2),
                 "power": round(float(psd[i]), 4)} for i in peak_idx[::-1]]
    return {"crypto_cycles": detected, "dominant_frequencies": dominant}


def ngram_entropy_rate(returns, n: int = 3, bins: int = 5) -> dict:
    """Sequence predictability via n-gram entropy rate.
    Discretize returns into bins, compute H(X_n | X_{n-1}...X_1).
    Low entropy rate = more predictable."""
    r = np.array(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < n + 10:
        return {"error": "too few samples"}
    quantiles = np.percentile(r, np.linspace(0, 100, bins + 1))
    discretized = np.digitize(r, quantiles[1:-1])
    from collections import Counter
    ngrams = [tuple(discretized[i:i+n]) for i in range(len(discretized) - n + 1)]
    prefix = [ng[:-1] for ng in ngrams]
    prefix_counts = Counter(prefix)
    ngram_counts = Counter(ngrams)
    h_cond = 0.0
    total = len(ngrams)
    for ng, count in ngram_counts.items():
        p_ng = count / total
        p_prefix = prefix_counts[ng[:-1]] / total
        h_cond -= p_ng * np.log2(p_ng / p_prefix)
    h_max = np.log2(bins)
    return {"entropy_rate": round(h_cond, 4), "h_max": round(h_max, 4),
            "normalized": round(h_cond / max(h_max, 1e-6), 4),
            "predictable": h_cond / max(h_max, 1e-6) < 0.85}


def cusum_detector(series, k: float = 0.5, h: float = 4.0) -> dict:
    """CUSUM online change point detection.
    k = allowance (slack), h = decision interval (threshold).
    Returns indices where regime changes detected."""
    s = np.array(series, dtype=float)
    s = (s - np.mean(s)) / max(np.std(s), 1e-12)
    S_pos, S_neg = 0.0, 0.0
    changes = []
    for i in range(len(s)):
        S_pos = max(0, S_pos + s[i] - k)
        S_neg = max(0, S_neg - s[i] - k)
        if S_pos > h:
            changes.append({"index": i, "direction": "up", "magnitude": round(float(S_pos), 2)})
            S_pos = 0.0
        if S_neg > h:
            changes.append({"index": i, "direction": "down", "magnitude": round(float(S_neg), 2)})
            S_neg = 0.0
    return {"change_points": changes, "n_changes": len(changes),
            "avg_interval": round(len(s) / max(len(changes), 1), 1)}


def cross_correlation_peak(x, y, max_lag: int = 20) -> dict:
    """Find optimal lead-lag between two series with confidence.
    Positive lag means x leads y. Uses normalized cross-correlation."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    x = (x - np.mean(x)) / max(np.std(x), 1e-12)
    y = (y - np.mean(y)) / max(np.std(y), 1e-12)
    n = min(len(x), len(y))
    lags = range(-max_lag, max_lag + 1)
    cc = []
    for lag in lags:
        if lag >= 0:
            c = float(np.mean(x[:n-lag] * y[lag:n])) if n > lag else 0
        else:
            c = float(np.mean(x[-lag:n] * y[:n+lag])) if n > -lag else 0
        cc.append(c)
    cc = np.array(cc)
    best_idx = np.argmax(np.abs(cc))
    best_lag = list(lags)[best_idx]
    noise_floor = 2.0 / np.sqrt(n)
    return {"best_lag": best_lag, "correlation": round(float(cc[best_idx]), 4),
            "noise_floor": round(noise_floor, 4),
            "significant": abs(cc[best_idx]) > noise_floor,
            "all_lags": {int(l): round(float(c), 4) for l, c in zip(lags, cc) if abs(c) > noise_floor}}


def granger_causality(x, y, max_lag: int = 5) -> dict:
    """Does x Granger-cause y? Tests if past x improves y prediction over past y alone.
    Uses OLS F-test. Returns p-values per lag."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = min(len(x), len(y))
    if n < max_lag + 20:
        return {"error": "too few samples"}
    results = {}
    for lag in range(1, max_lag + 1):
        Y = y[lag:n]
        X_restricted = np.column_stack([y[lag-j:n-j] for j in range(1, lag + 1)])
        X_full = np.column_stack([X_restricted] + [x[lag-j:n-j] for j in range(1, lag + 1)])
        X_restricted = np.column_stack([X_restricted, np.ones(len(Y))])
        X_full = np.column_stack([X_full, np.ones(len(Y))])
        rss_r = float(np.sum((Y - X_restricted @ np.linalg.lstsq(X_restricted, Y, rcond=None)[0])**2))
        rss_f = float(np.sum((Y - X_full @ np.linalg.lstsq(X_full, Y, rcond=None)[0])**2))
        df1 = lag
        df2 = len(Y) - X_full.shape[1]
        f_stat = ((rss_r - rss_f) / df1) / (rss_f / max(df2, 1))
        from scipy import stats
        p_val = float(1 - stats.f.cdf(f_stat, df1, df2))
        results[f"lag_{lag}"] = {"f_stat": round(f_stat, 4), "p_value": round(p_val, 4),
                                  "significant": p_val < 0.05}
    return results


KB.register_many([
    Atom("crypto.hmm_bic", "HMM BIC Model Selection", "cryptographer",
         hmm_bic.__doc__, atype=AType.FORMULA, affects=Affects.REGIME|Affects.MODELS, formula=hmm_bic),
    Atom("crypto.hmm_transition", "HMM Transition Eigenvalues", "cryptographer",
         hmm_transition_eigenvalues.__doc__, atype=AType.FORMULA, affects=Affects.REGIME, formula=hmm_transition_eigenvalues),
    Atom("crypto.snr_estimate", "Signal-to-Noise Ratio", "cryptographer",
         snr_estimate.__doc__, atype=AType.FORMULA, affects=Affects.FEATURES|Affects.SIGNALS, formula=snr_estimate),
    Atom("crypto.spectral_density", "Power Spectral Density", "cryptographer",
         spectral_density.__doc__, atype=AType.FORMULA, affects=Affects.FEATURES|Affects.SIGNALS,
         scales=Scale.ALL, formula=spectral_density, crypto_specific=True),
    Atom("crypto.ngram_entropy", "N-gram Entropy Rate", "cryptographer",
         ngram_entropy_rate.__doc__, atype=AType.FORMULA, affects=Affects.FEATURES, formula=ngram_entropy_rate),
    Atom("crypto.cusum", "CUSUM Change Point Detector", "cryptographer",
         cusum_detector.__doc__, atype=AType.FORMULA, affects=Affects.REGIME|Affects.SIGNALS, formula=cusum_detector),
    Atom("crypto.cross_correlation", "Cross-Correlation Peak", "cryptographer",
         cross_correlation_peak.__doc__, atype=AType.FORMULA, affects=Affects.SIGNALS, formula=cross_correlation_peak),
    Atom("crypto.granger", "Granger Causality Test", "cryptographer",
         granger_causality.__doc__, atype=AType.FORMULA, affects=Affects.SIGNALS|Affects.FEATURES, formula=granger_causality),
    # Thresholds
    Atom("crypto.snr_thresholds", "SNR Thresholds for Features", "cryptographer",
         "SNR < 0.001: pure noise. 0.001-0.01: marginal. 0.01-0.1: usable. > 0.1: strong signal.",
         atype=AType.THRESHOLD, affects=Affects.FEATURES,
         value={"noise": 0.001, "marginal": 0.01, "usable": 0.1, "strong": 1.0}),
    Atom("crypto.cycle_frequencies", "Crypto Cycle Frequencies", "cryptographer",
         "Key cycles: 8h (funding), 24h (daily volume), 168h (weekly), 720h (monthly options expiry)",
         atype=AType.REFERENCE, affects=Affects.SIGNALS, crypto_specific=True,
         value={"funding_8h": 1/(8*3600), "daily_24h": 1/(24*3600), "weekly_168h": 1/(168*3600)}),
    Atom("crypto.regime_benchmarks", "HMM Regime Quality Benchmarks", "cryptographer",
         "Good HMM: 3-5 states, mixing time 20-100 bars, stickiness 0.85-0.95, BIC decreasing",
         atype=AType.REFERENCE, affects=Affects.REGIME,
         value={"ideal_states": [3, 5], "mixing_time": [20, 100], "stickiness": [0.85, 0.95]}),
    # Crypto-specific
    Atom("crypto.funding_settlement", "Funding Settlement Pattern", "cryptographer",
         "Funding settled every 8h at 00:00/08:00/16:00 UTC. Pre-settlement: crowded side unwinds 30-60min before.",
         atype=AType.REFERENCE, affects=Affects.SIGNALS|Affects.EXECUTION, crypto_specific=True),
    Atom("crypto.wash_detector", "Wash Trading Detector", "cryptographer",
         "Volume spike > 5x normal with < 0.1% price change = likely wash trading. Common on MEXC altcoins.",
         atype=AType.THRESHOLD, affects=Affects.FEATURES, crypto_specific=True,
         value={"volume_spike_ratio": 5.0, "max_price_change": 0.001}),
])
