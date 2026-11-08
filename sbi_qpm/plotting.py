import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import corner as corner_lib


def plot_mollweide(hmap, title="", unit="", **kwargs):
    hp.mollview(hmap, title=title, unit=unit, **kwargs)
    plt.show()


def plot_power_spectrum(ells, cl, err=None, label="", ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    if err is not None:
        ax.errorbar(ells, cl, yerr=err, fmt="o-", label=label, **kwargs)
    else:
        ax.plot(ells, cl, "o-", label=label, **kwargs)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell$")
    if label:
        ax.legend()
    return ax


def plot_corner(samples, labels=None, truths=None, **kwargs):
    fig = corner_lib.corner(samples, labels=labels, truths=truths, **kwargs)
    plt.show()
    return fig


def plot_snr_per_ell(ells, snr, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    ax.bar(ells, snr, **kwargs)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("SNR")
    return ax


def plot_posterior_samples(samples, param_names=None, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    for i in range(samples.shape[1]):
        label = param_names[i] if param_names else f"param {i}"
        ax.hist(samples[:, i], bins=50, density=True, alpha=0.5, label=label, **kwargs)
    ax.legend()
    return ax


def plot_sky_distribution(ra, dec, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(ra, dec, s=0.5, **kwargs)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    return ax


def plot_comparisons(obs_map, sim_map, xlabel="", ax=None):
    if ax is None:
        _, ax = plt.subplots()
    lo = np.percentile(obs_map, 1)
    hi = np.percentile(obs_map, 99)
    bins = np.linspace(lo, hi, 80)
    ax.hist(obs_map, bins=bins, density=True, alpha=0.6, color="steelblue", label="observed")
    ax.hist(sim_map, bins=bins, density=True, alpha=0.6, color="tomato", label="simulated")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend()
    return ax


def plot_tarp_ecp(alphas, ecp_list, labels=None, colors=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for i, (alphas_k, ecp_k) in enumerate(zip(alphas, ecp_list)):
        label = labels[i] if labels else None
        color = colors[i] if colors else None
        ax.plot(alphas_k, ecp_k, color=color, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Ideal")
    ax.set_xlabel("Credibility", fontsize=13)
    ax.set_ylabel("Expected Coverage Probability", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if labels:
        ax.legend()
    return ax
