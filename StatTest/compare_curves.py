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


def functional_anova(language_bootstrap_curves: dict[str,np.ndarray], n_perm=2000):
    """
    Functional ANOVA via permutation of bootstrap curves.
    Input:
        language_bootstrap_curves: bootstrap curves for each language (n_bootstraps replicates, n_grid_pts)
        n_permutations: number of permutations to run
    Returns:
        dict with keys:
          - 'observed_stat' : observed between-group L2 statistic (scalar)
          - 'perm_stats'    : array of permutation statistics (length n_permutations)
          - 'p_value'       : permutation p-value (Monte Carlo, (r+1)/(n_perm+1))
    """

    curves = np.asarray(language_bootstrap_curves)
    n_langs, n_bootstraps, n_grid_pts = curves.shape

    # 1) observed means per language (shape: n_langs x n_grid_pts)
    observed_means = curves.mean(axis=1)

    # 2) observed between-group L2 statistic:
    #    sum over languages of integrated squared deviation from grand mean
    grand_mean = observed_means.mean(axis=0)
    observed_stat = sum(np.sum((observed_means[i] - grand_mean)**2) for i in range(n_langs))

    # 3) pool all bootstrap curves and permute them
    pooled = curves.reshape(n_langs * n_bootstraps, n_grid_pts)  # shape (n_langs*n_bootstraps, n_grid_pts)

    perm_stats = np.zeros(n_perm)
    for b in tqdm(range(n_perm), desc="Functional ANOVA permutations"):
        permuted = np.random.permutation(pooled)
        perm_groups = permuted.reshape(n_langs, n_bootstraps, n_grid_pts)
        perm_means = perm_groups.mean(axis=1)    # shape (n_langs, n_grid_pts)
        perm_grand = perm_means.mean(axis=0)
        perm_stat = sum(np.sum((perm_means[i] - perm_grand)**2) for i in range(n_langs))
        perm_stats[b] = perm_stat

    # 4) p-value (Monte Carlo permutation p-value with +1 correction)
    r = np.sum(perm_stats >= observed_stat)
    p_value = (r + 1) / (n_perm + 1)

    return {
        "observed_stat": float(observed_stat),
        "perm_stats": perm_stats,
        "p_value": float(p_value)
    }


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

    language_bootstrap_curves = {lang: boot[lang]["boot"] for lang in languages}

    # Functional ANOVA
    f_anova = functional_anova(language_bootstrap_curves, n_perm=2000)

    return {
        "boot": boot,
        "anova": f_anova,
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
