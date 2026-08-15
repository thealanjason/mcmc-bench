import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import arviz as az
import numpy as np

# ---------------------------------------------------------------------------
# Global matplotlib defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.6,
    "figure.dpi": 300,
})

N_BINS = 36
MAX_AUTOCORR_LAG = 100
RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400
SAVE_DPI = 200

TRACE_COLOR = '#8cc5e3'
LABEL_BBOX = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none')

# Per-subplot sizes (width, height) in inches.
# Trace rows are wide and short (time series needs horizontal space).
# Autocorr is near-square (symmetric lag axis).
# Posteriors are wider than tall (histogram).
TRACE_SUBPLOT_SIZE = (5.0, 2.5)
AUTOCORR_SUBPLOT_SIZE = (3.5, 3.0)
POSTERIOR_SUBPLOT_SIZE = (5.0, 4.0)





def run_quantitative_diagnostics(idata, param_labels, sampler_name, output_dir=None, n_model_calls=None, sampling_time_seconds=None, kish_ess=None):
    """
    Write R-hat and ESS convergence diagnostics to a CSV file.

    Parameters
    ----------
    idata : arviz.InferenceData
    param_labels : dict
        Maps parameter names to display labels used as the CSV row index.
    output_dir : str or Path, optional
        Directory for convergence_diagnostics.csv. Defaults to current directory.
    """
    param_names = list(param_labels)
    summary = az.summary(idata, var_names=param_names, round_to="none")
    display = summary[["mean", "sd", "ess_bulk", "ess_tail", "r_hat"]].copy()
    # We use either number of calls or sampling duration
    # For weighted-sample outputs (nested sampling) the ArviZ ESS is computed on
    # the equal-weight resample and overstates the information content. Use the
    # Kish ESS as the numerator when the sampler provides one.
    display["ess_kish"] = kish_ess if kish_ess is not None else np.nan
    ess_for_efficiency = kish_ess if kish_ess is not None else display["ess_bulk"]

    if n_model_calls:
        display["ess_per_call"] = ess_for_efficiency / n_model_calls
    else:
        display["ess_per_call"] = np.nan
    if sampling_time_seconds:
        display["ess_per_second"] = ess_for_efficiency / sampling_time_seconds
    else:
        display["ess_per_second"] = np.nan
        
    n_chains = idata.posterior.sizes.get("chain", 1)
    if sampler_name in {"emcee", "dynesty", "pymc_smc"} or n_chains < 2:
        display["converged"] = "N/A"
    else:
        convergence_mask = (
            display["r_hat"].notna()
            & (display["r_hat"] < RHAT_THRESHOLD)
            & (display["ess_bulk"] > ESS_THRESHOLD)
            & (display["ess_tail"] > ESS_THRESHOLD)
        )
        display["converged"] = np.where(
            convergence_mask,
            "PASS",
            "FAIL",
        )

    # Use human-readable display labels as the row index
    display.index = [param_labels[p] for p in display.index]
    display.index.name = "parameter"

    out = Path(output_dir) if output_dir is not None else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "convergence_diagnostics.csv"
    display.to_csv(csv_path, float_format="%.6g")


def _make_subplot_grid(n_params, n_cols, subplot_size):
    """Create a subplot grid scaled by per-subplot size. Returns flattened axes."""
    n_cols = min(n_cols, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    figsize = (n_cols * subplot_size[0], n_rows * subplot_size[1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.array(axes).flatten()
    for ax in axes_flat[n_params:]:
        ax.axis('off')
    return fig, axes_flat, n_rows, n_cols


def _save_figure(fig, output_dir, filename):
    plt.tight_layout()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches='tight')


def _add_panel_label(ax, idx):
    ax.text(0.03, 0.95, f'({chr(97 + idx)})',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=LABEL_BBOX)


def _plot_trace(idata, param_labels, output_dir):
    param_names = list(param_labels)
    n_params = len(param_names)

    w, h = TRACE_SUBPLOT_SIZE
    fig, axes = plt.subplots(n_params, 2, figsize=(2 * w, n_params * h))
    if n_params == 1:
        axes = axes.reshape(1, -1)

    az.plot_trace(
        idata,
        var_names=param_names,
        axes=axes,
        trace_kwargs={"color": TRACE_COLOR, "alpha": 1.0, "linewidth": 0.8},
        hist_kwargs={"color": TRACE_COLOR, "edgecolor": "none", "bins": N_BINS, "alpha": 1.0,},
        compact=False,
    )

    for i, (param, label) in enumerate(param_labels.items()):
        axes[i, 0].set_title('')
        axes[i, 1].set_title('')
        axes[i, 0].set_ylabel(label)
        # _add_panel_label(axes[i, 0], i)

    axes[-1, 0].set_xlabel('Density')
    axes[-1, 1].set_xlabel('Draw')
    _save_figure(fig, output_dir, 'trace.png')


def _plot_autocorr(idata, param_labels, output_dir):
    param_names = list(param_labels)
    n_params = len(param_names)
    n_samples = (
        idata.posterior.sizes["chain"]
        * idata.posterior.sizes["draw"]
    )
    max_lag = min(MAX_AUTOCORR_LAG, n_samples)

    fig, axes_flat, _, n_cols = _make_subplot_grid(n_params, n_cols=3, subplot_size=AUTOCORR_SUBPLOT_SIZE)
    
    az.plot_autocorr(
        idata,
        var_names=param_names,
        max_lag=max_lag,
        combined=True,
        ax=axes_flat[:n_params],
    )

    for idx, (param, label) in enumerate(param_labels.items()):
        ax = axes_flat[idx]
        ax.set_title('')
        ax.axhline(0, color='dimgray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_ylim(-0.1, 1.05)
        ax.text(0.5, 1.02, label, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='center')
        # _add_panel_label(ax, idx)
        if idx % n_cols == 0:
            ax.set_ylabel('Autocorrelation')
        if idx // n_cols == (n_params - 1) // n_cols:
            ax.set_xlabel('Lag')

    _save_figure(fig, output_dir, 'autocorr.png')


def _plot_posteriors(idata, param_labels, output_dir):
    param_names = list(param_labels)
    n_params = len(param_names)
    fig, axes_flat, _, _ = _make_subplot_grid(n_params, n_cols=2,
                                              subplot_size=POSTERIOR_SUBPLOT_SIZE)

    for idx, (param, label) in enumerate(param_labels.items()):
        ax = axes_flat[idx]
        ax.hist(idata.posterior[param].values.flatten(),
                bins=N_BINS, color=TRACE_COLOR, edgecolor='none')
        ax.set_xlabel(label)
        ax.set_ylabel('Frequency')
        # _add_panel_label(ax, idx)

    _save_figure(fig, output_dir, 'posteriors.png')


def run_diagnostics(idata, sampler_name, nburn=0, param_labels=None, output_dir=None, n_model_calls=None, sampling_time_seconds=None, kish_ess=None):
    """
    Create MCMC diagnostics. Qualitative plots and Quantitavie metrics 

    Parameters
    ----------
    idata : arviz.InferenceData
    param_labels : dict, optional
        Maps parameter names to display labels.
        Example: mu and xi
    output_dir : str or Path, optional
        Directory to save figures as PNG. Filenames are fixed:
        trace.png, autocorr.png, posteriors.png.
    """
    if nburn < 0:
        raise ValueError("nburn cannot be negative")

    if sampler_name in {"emcee", "rwmcmc"} and nburn > 0:
        n_draws = idata.posterior.sizes["draw"]

        if nburn >= n_draws:
            raise ValueError(
                f"nburn ({nburn}) must be smaller than the number "
                f"of draws ({n_draws})"
            )

        idata = idata.isel(draw=slice(nburn, None))

    if param_labels is None:
        param_labels = {p: p for p in idata.posterior.data_vars}

    out = Path(output_dir) if output_dir is not None else None


    
    _plot_trace(idata, param_labels, out)
    _plot_autocorr(idata, param_labels, out)
    _plot_posteriors(idata, param_labels, out)
    run_quantitative_diagnostics(idata, param_labels, sampler_name, out,
                                 n_model_calls=n_model_calls,
                                 sampling_time_seconds=sampling_time_seconds, kish_ess=kish_ess)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCMC diagnostics on a NetCDF inference data file.")
    parser.add_argument("--idata-path", type=Path, help="Path to the .nc InferenceData file.")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory to write outputs (default: current directory).")
    parser.add_argument("--sampler", required=True, choices=["emcee", "rwmcmc", "dynesty", "pymc_slice", "pymc_smc"], help="Sampler used to generate the inference data.",)
    parser.add_argument("--nburn", type=int, default=0, help="Number of initial draws to exclude when applicable.",)
    parser.add_argument("--ncalls-path", type=Path, default=None, help="Path to n_calls.txt written by the model server.")
    parser.add_argument("--npz-path", type=Path, default=None, help="Path to mcmc_output.npz (for sampling_time_seconds).")
    args = parser.parse_args()

    # We check whether the files exist before passing the metrics on.
    n_model_calls = None
    if args.ncalls_path is not None and args.ncalls_path.exists():
        n_model_calls = int(args.ncalls_path.read_text().strip())

    sampling_time_seconds = None
    kish_ess = None
    if args.npz_path is not None and args.npz_path.exists():
        with np.load(args.npz_path) as npz:
            if "sampling_time_seconds" in npz.files:
                sampling_time_seconds = float(npz["sampling_time_seconds"])
            if "kish_ess" in npz.files:
                kish_ess = float(npz["kish_ess"])
    idata = az.from_netcdf(args.idata_path)
    run_diagnostics(idata, sampler_name=args.sampler, nburn=args.nburn,
                    output_dir=args.output_dir,
                    n_model_calls=n_model_calls,
                    sampling_time_seconds=sampling_time_seconds, kish_ess=kish_ess)