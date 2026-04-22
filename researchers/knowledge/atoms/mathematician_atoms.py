"""Mathematician's knowledge atoms."""
import numpy as np
from knowledge.registry import KB, Atom, Pair, Regime, Scale, AType, Affects

def kelly_optimal(p: float, b: float = 1.0) -> dict:
    """Kelly criterion: f* = (p*b - q) / b.
    Returns optimal, half, quarter Kelly with growth rates.
    ALWAYS use fractional Kelly due to estimation error.
    For crypto with high vol: quarter Kelly (c=0.25) is standard."""
    q = 1.0 - p
    if b <= 0 or p <= 0 or p >= 1: return {"f_star": 0, "error": "invalid"}
    f = (p * b - q) / b
    def g(f_): return p * np.log(1 + f_ * b) + q * np.log(max(1e-10, 1 - f_)) if f_ > 0 else 0
    return {"f_star": round(f,4), "half_kelly": round(f/2,4), "quarter_kelly": round(f/4,4),
            "growth_full": round(g(f),6), "growth_half": round(g(f/2),6),
            "edge_bps": round((p*b-q)*10000, 1)}

def kelly_multi_asset(mu, cov) -> dict:
    """f* = Sigma^{-1} * mu for N correlated assets. The mathematical heart of portfolio construction.
    REQUIRES cleaned covariance (use marchenko_pastur_clean first)."""
    mu, cov = np.array(mu), np.array(cov)
    try:
        f = np.linalg.inv(cov) @ mu
        pvar = float(f @ cov @ f)
        return {"fractions": np.round(f,4).tolist(), "half_kelly": np.round(f/2,4).tolist(),
                "leverage": round(float(np.sum(np.abs(f))),2), "variance": round(pvar,6),
                "growth": round(float(mu @ f - 0.5 * pvar),6)}
    except np.linalg.LinAlgError:
        return {"error": "Singular matrix -- use shrinkage"}

def effective_n_bets(corr_matrix) -> dict:
    """How many independent bets from N correlated assets?
    N_eff = N/(1+(N-1)*rho_bar). With 6 pairs at rho=0.7 -> only 1.2 independent bets.
    We THINK we have 6 trades but really ~1.2."""
    c = np.array(corr_matrix)
    n = c.shape[0]
    mask = np.triu(np.ones_like(c, dtype=bool), k=1)
    avg_rho = float(np.mean(c[mask]))
    n_eff = n / (1 + (n-1) * avg_rho)
    eig = np.linalg.eigvalsh(c); eig = eig[eig > 0]
    n_eff_e = float(np.sum(eig)**2 / np.sum(eig**2))
    return {"n_assets": n, "avg_correlation": round(avg_rho,4),
            "n_eff": round(n_eff_e,2), "eigenvalues": sorted(np.round(eig,4).tolist(), reverse=True),
            "danger": n_eff_e < 2}

def hoeffding_sample_size(win_rate=0.54, margin=0.02, confidence=0.95) -> dict:
    """Min trades to confirm win rate +/-margin. At 54% +/-2% 95%: need 4,612 trades.
    At 100 trades/week -> 46 weeks -> 11 months."""
    alpha = 1 - confidence
    n = int(np.ceil(np.log(2/alpha) / (2 * margin**2)))
    return {"min_n": n, "weeks_at_100_pw": round(n/100,1), "months": round(n/100/4.33,1)}

def marchenko_pastur_clean(corr_matrix, n_samples: int) -> dict:
    """Clean correlation matrix with RMT. Eigenvalues in MP bulk [lambda_-,lambda_+] are noise.
    Shrink noise to 1.0, keep signal eigenvalues."""
    c = np.array(corr_matrix)
    n = c.shape[0]; q = n / n_samples
    lp = (1 + np.sqrt(q))**2; lm = (1 - np.sqrt(q))**2
    eig, vec = np.linalg.eigh(c)
    cleaned = eig.copy(); noise = eig <= lp; cleaned[noise] = 1.0
    result = vec @ np.diag(cleaned) @ vec.T
    d = np.sqrt(np.diag(result)); result = result / np.outer(d, d)
    return {"cleaned_matrix": np.round(result,4).tolist(),
            "mp_bounds": [round(lm,4), round(lp,4)],
            "signal_count": int((~noise).sum()), "noise_count": int(noise.sum())}

def entropy(probs) -> dict:
    """Shannon entropy H = -sum p*log2(p). For regime detector: high H = confused."""
    p = np.array(probs, dtype=float); p = p[p>0]; p = p/p.sum()
    h = float(-np.sum(p * np.log2(p))); hmax = float(np.log2(len(p)))
    return {"H": round(h,4), "H_max": round(hmax,4), "normalized": round(h/max(hmax,1e-6),4),
            "confused": h/max(hmax,1e-6) > 0.85}

def kl_divergence(p, q) -> dict:
    """D_KL(P||Q) for distribution drift detection.
    > 0.1 nats: investigate. > 0.5 nats: regime shift."""
    p, q = np.array(p, dtype=float), np.array(q, dtype=float)
    p, q = p/p.sum(), q/q.sum()
    mask = (p > 0) & (q > 0)
    dkl = float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
    return {"D_KL": round(dkl,6), "investigate": dkl > 0.1, "regime_shift": dkl > 0.5}

def wasserstein_1d(a, b) -> dict:
    """Earth mover's distance between empirical distributions.
    Better than KL for comparing return distributions (defined even with no overlap)."""
    from scipy.stats import wasserstein_distance
    w = wasserstein_distance(a, b)
    return {"W1": round(w,6)}

KB.register_many([
    Atom("math.kelly_optimal", "Kelly Optimal Fraction", "mathematician",
         kelly_optimal.__doc__, atype=AType.FORMULA, affects=Affects.SIZING, formula=kelly_optimal,
         see_also=["math.kelly_multi_asset","math.hoeffding_sample_size"]),
    Atom("math.kelly_multi_asset", "Multi-Asset Kelly", "mathematician",
         kelly_multi_asset.__doc__, atype=AType.FORMULA, affects=Affects.SIZING, formula=kelly_multi_asset),
    Atom("math.effective_n_bets", "Effective Independent Bets", "mathematician",
         effective_n_bets.__doc__, atype=AType.FORMULA, affects=Affects.SIZING|Affects.RISK, formula=effective_n_bets),
    Atom("math.hoeffding_sample_size", "Sample Size for Win Rate CI", "mathematician",
         hoeffding_sample_size.__doc__, atype=AType.FORMULA, affects=Affects.SIZING|Affects.MODELS, formula=hoeffding_sample_size),
    Atom("math.marchenko_pastur_clean", "RMT Correlation Cleaning", "mathematician",
         marchenko_pastur_clean.__doc__, atype=AType.FORMULA, affects=Affects.SIZING, scales=Scale.SLOW, formula=marchenko_pastur_clean),
    Atom("math.entropy", "Shannon Entropy", "mathematician",
         entropy.__doc__, atype=AType.FORMULA, affects=Affects.REGIME|Affects.SIZING, formula=entropy),
    Atom("math.kl_divergence", "KL Divergence", "mathematician",
         kl_divergence.__doc__, atype=AType.FORMULA, affects=Affects.MODELS|Affects.REGIME, formula=kl_divergence),
    Atom("math.wasserstein", "Wasserstein Distance", "mathematician",
         wasserstein_1d.__doc__, atype=AType.FORMULA, affects=Affects.MODELS, formula=wasserstein_1d),
    Atom("math.kelly_fraction_by_regime", "Regime-Dependent Kelly", "mathematician",
         "Kelly fraction multiplier: low_vol=0.50, trending=0.40, high_vol=0.25, transition=0.15",
         atype=AType.THRESHOLD, affects=Affects.SIZING,
         value={"low_volatility":0.50, "trending":0.40, "high_volatility":0.25, "transition":0.15}),
    Atom("math.significance_table", "Sample Sizes for Confidence", "mathematician",
         "Pre-computed: 95%+/-1%->18444, 95%+/-2%->4612, 95%+/-3%->2050, 95%+/-5%->738, 90%+/-2%->3381",
         atype=AType.REFERENCE, affects=Affects.MODELS,
         value={"95_1pct":18444, "95_2pct":4612, "95_3pct":2050, "95_5pct":738, "90_2pct":3381}),
])
