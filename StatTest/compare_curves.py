import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import make_splrep
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import pymc as pm
import argparse


def load_languages(directory: str) -> dict[str:pd.DataFrame]:
    """
    Load all CSV files in directory.
    Returns a dict: {language_name: dataframe}
    """
    files = glob.glob(os.path.join(directory, "*.csv"))
    data = {}
    for f in files:
        lang = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        df = df[['D_order', 'D_structure']].dropna()
        data[lang] = df
    return data


def monotone_spline_fit(x: iter, y: iter, smoothing=1.0) -> callable:
    """
    Monotone decreasing fit:
    (1) Perform isotonic regression on y to enforce decreasing monotonicity.
    (2) Fit smoothing spline to the monotone output.
    Returns a callable function f(x_eval).
    """
    iso = IsotonicRegression(increasing=False)
    y_iso = iso.fit_transform(x, y)

    # Smoothing spline applied to isotonic output
    return make_splrep(x, y_iso, s=smoothing)


def bootstrap_curves(x: iter, y: iter, b=2000, smoothing=1.0, grid=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap monotone fits. This creates multiple bootstrapped curves for a single language.
    Returns matrix of shape (B, len(grid)) with fitted curves.

    Args:
        x: the values on the x-axis
        y: the corresponding values on the y-axis
        b: number of bootstrap resamples
        smoothing: The smoothing condition for the spline fit
        grid: a 1D array of x-values

    Returns:
        the grid used and the bootstrap curves generated
    """
    if grid is None:
        grid = np.linspace(min(x), max(x), 200)

    boot_curves = np.zeros((b, len(grid)))

    n = len(x)
    # desc is the Prefix for the progressbar
    for b_i in tqdm(range(b), desc="Bootstrapping"):
        # Select n values between 0 and n-1, with replacement
        idx = np.random.choice(n, n, replace=True)
        xb, yb = x[idx], y[idx]
        f = monotone_spline_fit(xb, yb, smoothing=smoothing)
        boot_curves[b_i] = f(grid)

    return grid, boot_curves


def global_envelope_test(language_curves: dict[str,np.ndarray], n_perm=2000):
    """
    Permutation Global Envelope Test

    Args:
        language_curves: dict {lang: curve_values_on_common_grid}
        n_perm: the number of permutations
    Returns:
         envelope bands + whether curves fall outside the envelope under a permuted null.
    """
    langs = list(language_curves.keys())
    # np.stack combines a sequence of arrays along a new axis, effectively creating a higher-dimensional array
    curves = np.stack([language_curves[lang] for lang in langs])
    n_languages, _ = curves.shape

    # Null hypothesis: curves are exchangeable across languages
    perm_stats = []

    # Test statistic: L2 norm from mean curve
    observed_mean = curves.mean(axis=0) # This is the mean across languages
    # This gives one sqrt(diff**2) per language; it's not normalized by length
    observed_stats = np.array([
        np.sqrt(((curves[i] - observed_mean)**2).sum())
        for i in range(n_languages)
    ])

    # Permutations
    for _ in tqdm(range(n_perm), desc="Permutation Envelope"):
        perm = np.random.permutation(n_languages)
        # This effectively reassigns the curves to new languages
        perm_curves = curves[perm]
        perm_mean = perm_curves.mean(axis=0)
        assert perm_mean == observed_mean, 'After permuting the languages the mean changed'
        stat = np.array([
            np.sqrt(((perm_curves[i] - perm_mean)**2).sum())
            for i in range(n_languages)
        ])
        perm_stats.append(stat)

    perm_stats = np.array(perm_stats)  # shape (n_perm, L)

    # Envelope: compute p-values for each language
    p_vals = []
    for i in range(n_languages):
        p = (np.sum(perm_stats[:, i] >= observed_stats[i]) + 1) / (n_perm + 1)
        p_vals.append(p)

    # noinspection SpellCheckingInspection
    return {
        "languages": langs,
        "pvals": p_vals,
        "observed_stats": observed_stats,
        "perm_stats": perm_stats
    }


# ============================================================
# 5. Functional ANOVA (Permutation)
# ============================================================

def functional_anova(language_curves, n_perm=2000):
    """
    Simple permutation ANOVA:
    Compute variance between languages vs. within.
    """
    langs = list(language_curves.keys())
    curves = np.stack([language_curves[k] for k in langs])
    an_l, an_m = curves.shape

    grand_mean = curves.mean(axis=0)
    between = sum(((curves[i] - grand_mean) ** 2).sum() for i in range(an_l))

    perm_values = []
    for _ in tqdm(range(n_perm), desc="Functional ANOVA permutations"):
        # permute curve labels
        perm = np.random.permutation(an_l)
        perm_curves = curves[perm]
        perm_mean = perm_curves.mean(axis=0)
        stat = sum(((perm_curves[i] - perm_mean) ** 2).sum() for i in range(an_l))
        perm_values.append(stat)

    perm_values = np.array(perm_values)
    p_val = (np.sum(perm_values >= between) + 1) / (1 + n_perm)

    return {"between_stat": between, "perm_stats": perm_values, "p": p_val}


# ============================================================
# 6. Hierarchical Gaussian Process confirmatory analysis
# ============================================================

def hierarchical_gp(language_curves, grid):
    """
    Bayesian hierarchical GP model:
    f_i(x) = f_shared(x) + g_i(x)
    """
    langs = list(language_curves.keys())
    curves = np.stack([language_curves[k] for k in langs])
    an_l, an_m = curves.shape

    with pm.Model() as model:
        # Shared GP
        l_shared = pm.Gamma("ℓ_shared", alpha=2, beta=1)
        eta_shared = pm.HalfNormal("η_shared", sigma=1)
        cov_shared = eta_shared**2 * pm.gp.cov.ExpQuad(1, l_shared)
        gp_shared = pm.gp.Latent(cov_func=cov_shared)
        f_shared = gp_shared.prior("f_shared", X=grid[:, None])

        # Language deviations GP
        l_dev = pm.Gamma("ℓ_dev", alpha=2, beta=1)
        eta_dev = pm.HalfNormal("η_dev", sigma=1)
        cov_dev = eta_dev**2 * pm.gp.cov.ExpQuad(1, l_dev)

        g = []
        for i in range(an_l):
            gp_dev = pm.gp.Latent(cov_func=cov_dev)
            g_i = gp_dev.prior(f"g_{i}", X=grid[:, None])
            g.append(g_i)

        # Noise
        a_sigma = pm.HalfNormal("σ", sigma=0.1)

        # Observed curves
        for i in range(an_l):
            pm.Normal(f"y_{i}", mu=f_shared + g[i], sigma=a_sigma, observed=curves[i])

        trace = pm.sample(2000, tune=2000, target_accept=0.9)

    return trace


# ============================================================
# 7. Main pipeline
# ============================================================

def run_pipeline(directory):
    data = load_languages(directory)
    languages = list(data.keys())

    # Common grid
    all_x = np.concatenate([data[lang].D_order.values for lang in languages])
    grid = np.linspace(min(all_x), max(all_x), 200)

    # Fit monotone splines and bootstrap
    boot = {}
    for lang in languages:
        df = data[lang]
        x = df.D_order.values
        y = df.D_structure.values

        f = monotone_spline_fit(x, y, smoothing=1.0)
        curve = f(grid)

        # Bootstrap
        _, boot_curves = bootstrap_curves(x, y, b=2000, smoothing=1.0, grid=grid)

        boot[lang] = {
            "curve": curve,
            "boot": boot_curves
        }

    # Global envelope test
    language_curves = {lang: boot[lang]["curve"] for lang in languages}
    envelope = global_envelope_test(language_curves, n_perm=2000)

    # Functional ANOVA
    f_anova = functional_anova(language_curves, n_perm=2000)

    # Hierarchical GP
    trace = hierarchical_gp(language_curves, grid)

    return {
        "boot": boot,
        "envelope": envelope,
        "anova": f_anova,
        "gp_trace": trace,
        "grid": grid
    }


# ============================================================
# 8. Run from command line
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Full statistical pipeline for monotone curves.")
    parser.add_argument("directory", help="Directory containing CSV files")
    args = parser.parse_args()

    results = run_pipeline(args.directory)

    print("Done. Results dictionary contains bootstrap, envelope test, ANOVA, and GP trace.")
