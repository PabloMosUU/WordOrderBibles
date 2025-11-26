import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# ============================================================
# 1. Plot monotone curve + bootstrap CI per language
# ============================================================

def plot_language_bootstrap(results, save=False):
    grid = results["grid"]
    boot = results["boot"]

    n_langs = len(boot)
    fig, axes = plt.subplots(n_langs, 1, figsize=(8, 3*n_langs))

    if n_langs == 1:
        axes = [axes]

    for ax, lang in zip(axes, boot.keys()):
        curve = boot[lang]["curve"]
        boot_curves = boot[lang]["boot"]

        lower = np.percentile(boot_curves, 2.5, axis=0)
        upper = np.percentile(boot_curves, 97.5, axis=0)

        ax.plot(grid, curve, color="blue", label="Monotone fit")
        ax.fill_between(grid, lower, upper, color="blue", alpha=0.2,
                        label="95% Bootstrap CI")

        ax.set_title(f"Language: {lang}")
        ax.set_xlabel("Order")
        ax.set_ylabel("Structure")
        ax.legend()

    plt.tight_layout()
    if save:
        plt.savefig("bootstrap_fits.png", dpi=300)
    plt.show()


# ============================================================
# 2. Overlay curves for all languages
# ============================================================

def plot_overlay(results, save=False):
    grid = results["grid"]
    boot = results["boot"]

    plt.figure(figsize=(8,6))

    for lang in boot.keys():
        plt.plot(grid, boot[lang]["curve"], lw=2, label=lang)

    plt.xlabel("Order")
    plt.ylabel("Structure")
    plt.title("All Languages: Monotone Fits Overlay")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig("overlay_fits.png", dpi=300)
    plt.show()


# ============================================================
# 3. Global Envelope Test Visualization
# ============================================================

def plot_global_envelope(results, save=False):
    grid = results["grid"]
    boot = results["boot"]
    envelope = results["envelope"]

    langs = envelope["languages"]
    curves = np.stack([boot[lang]["curve"] for lang in langs])
    L, M = curves.shape

    perm_stats = envelope["perm_stats"]  # shape (n_perm, L)

    # Envelope of permuted distances (95%)
    lower_env = np.percentile(perm_stats, 2.5, axis=0)
    upper_env = np.percentile(perm_stats, 97.5, axis=0)
    observed = envelope["observed_stats"]

    plt.figure(figsize=(8,6))

    # Plot envelope
    plt.fill_between(range(L), lower_env, upper_env,
                     color="gray", alpha=0.3, label="95% permutation envelope")

    # Plot observed curve distances
    plt.plot(range(L), observed, "o-", color="red", label="Observed curve stats")

    plt.xticks(range(L), langs, rotation=45)
    plt.ylabel("L2 distance from mean curve")
    plt.title("Global Envelope Test")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig("global_envelope.png", dpi=300)
    plt.show()


# ============================================================
# 4. Functional ANOVA Plot
# ============================================================

def plot_functional_anova(results, save=False):
    anova = results["anova"]

    plt.figure(figsize=(8,5))
    plt.hist(anova["perm_stats"], bins=40, alpha=0.7, label="Null distribution")
    plt.axvline(anova["between_stat"], color="red", lw=3, label=f"Observed stat (p={anova['p']:.4f})")

    plt.xlabel("Between-group variance statistic")
    plt.ylabel("Frequency")
    plt.title("Permutation Functional ANOVA")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig("functional_anova.png", dpi=300)
    plt.show()


# ============================================================
# 5. Hierarchical GP: Shared Curve Posterior
# ============================================================

def plot_gp_shared(results, save=False):
    grid = results["grid"]
    trace = results["gp_trace"]

    # Extract posterior samples
    f_shared = trace.posterior["f_shared"].values  # shape: (chains, draws, grid)

    # Compute posterior mean and 95% CI
    mean_curve = f_shared.mean(axis=(0,1))
    lower = np.percentile(f_shared, 2.5, axis=(0,1))
    upper = np.percentile(f_shared, 97.5, axis=(0,1))

    plt.figure(figsize=(8,6))
    plt.plot(grid, mean_curve, color="black", lw=2, label="Shared GP mean")
    plt.fill_between(grid, lower, upper, color="gray", alpha=0.3,
                     label="95% credible interval")

    plt.xlabel("Order")
    plt.ylabel("Structure")
    plt.title("Hierarchical GP: Shared Underlying Curve")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig("gp_shared.png", dpi=300)
    plt.show()


# ============================================================
# 6. Hierarchical GP: Language-specific Deviations
# ============================================================

def plot_gp_deviations(results, save=False):
    grid = results["grid"]
    trace = results["gp_trace"]
    envelope = results["envelope"]

    langs = envelope["languages"]

    # Extract deviations g_i
    g = [trace.posterior[f"g_{i}"].values for i in range(len(langs))]

    n_langs = len(langs)
    fig, axes = plt.subplots(n_langs, 1, figsize=(8, 3*n_langs))

    if n_langs == 1:
        axes = [axes]

    for ax, lang, g_i in zip(axes, langs, g):
        mean_dev = g_i.mean(axis=(0,1))
        lower = np.percentile(g_i, 2.5, axis=(0,1))
        upper = np.percentile(g_i, 97.5, axis=(0,1))

        ax.plot(grid, mean_dev, color="purple", lw=2, label=f"g_i deviation ({lang})")
        ax.fill_between(grid, lower, upper, color="purple", alpha=0.2,
                        label="95% credible interval")

        ax.axhline(0, color="black", lw=1)
        ax.set_title(f"GP Language Deviation: {lang}")
        ax.set_xlabel("Order")
        ax.set_ylabel("Deviation")
        ax.legend()

    plt.tight_layout()

    if save:
        plt.savefig("gp_deviations.png", dpi=300)
    plt.show()


# ============================================================
# 7. Convenience wrapper to plot everything
# ============================================================

def plot_all(results):
    plot_language_bootstrap(results)
    plot_overlay(results)
    plot_global_envelope(results)
    plot_functional_anova(results)
    plot_gp_shared(results)
    plot_gp_deviations(results)
