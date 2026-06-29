"""
References
----------
- dynesty docs:
    https://dynesty.readthedocs.io/
- dynesty source:
    https://github.com/joshspeagle/dynesty
"""
import argparse
import yaml
import numpy as np
import pandas as pd
from scipy.stats import uniform, norm, truncnorm

import umbridge
import corner
import arviz as az

import dynesty
from dynesty.utils import resample_equal


class Prior:
    @staticmethod
    def _distributions_from_config(prior_config: dict):
        name = prior_config["name"]
        distribution = prior_config["distribution"]

        if distribution["type"] not in ["uniform", "normal", "truncated_normal"]:
            print(f"Distribution type {distribution['type']} not yet supported")
            print("Aborting MCMC Calibration")
            exit(1)

        if distribution["type"] == "uniform":
            if not all(attribute in distribution["attribute"] for attribute in ("upper_bound", "lower_bound")):
                print(f"Incorrect definition of prior for parameter '{name}'",
                    f"\nA uniform prior requires `lower_bound` and `upper_bound` attributes")
                print("Aborting MCMC Calibration")
                exit(1)
            else:
                loc = distribution["attribute"]["lower_bound"]
                scale = distribution["attribute"]["upper_bound"] - distribution["attribute"]["lower_bound"]
                return uniform(loc, scale)
        elif distribution["type"] == "normal":
            if not all(attribute in distribution["attribute"] for attribute in ("loc", "scale")):
                print(f"Incorrect definition of prior for parameter '{name}'",
                    f"\nA {distribution['type']} prior requires `loc` and `scale` attributes")
                print("Aborting MCMC Calibration")
                exit(1)
            else:
                loc = distribution["attribute"]["loc"]
                scale = distribution["attribute"]["scale"]
                return norm(loc, scale)
        elif distribution["type"] == "truncated_normal":
            if not (all(attribute in distribution["attribute"] for attribute in ("loc", "scale"))
                    and any(attribute in distribution["attribute"] for attribute in ("lower_bound", "upper_bound"))):
                print(f"Incorrect definition of prior for parameter '{name}'",
                    f"\nA {distribution["type"]} prior requires `loc`, `scale`, `lower_bound` and/or `upper_bound` attributes")
                print("Aborting MCMC Calibration")
                exit(1)
            else:
                lower_bound = distribution["attribute"].get("lower_bound", -np.inf)
                upper_bound = distribution["attribute"].get("upper_bound", np.inf)
                loc = distribution["attribute"]["loc"]
                scale = distribution["attribute"]["scale"]
                a, b = (lower_bound - loc) / scale, (upper_bound - loc) / scale
                return truncnorm(a, b, loc, scale)

    def __init__(self, config: dict, parameters: list, noise_parameters: list, calibrate_noise: bool = False):
        self.config = config
        self.parameters = parameters
        self.noise_parameters = noise_parameters
        self.calibrate_noise = calibrate_noise

        all_parameters = list(parameters)
        if calibrate_noise:
            all_parameters += noise_parameters
        
        prior_names = [p["name"] for p in config]
        self.distributions = [
            Prior._distributions_from_config(config[prior_names.index(name)])
            for name in all_parameters
        ]

        self.all_parameters=all_parameters


class LogLikelihood:
    def __init__(self, model, data: np.ndarray, n_noise_parameters: int = 1, calibrate_noise: bool = True, noise_sigma=None, distribution_type: str = "normal"):
        if calibrate_noise and n_noise_parameters > 1:
            print("Only 1 noise parameter supported for log_likelihood")
            print("Aborting MCMC Calibration")
            exit(1)
        if not calibrate_noise and noise_sigma is None:
            print("log_likelihood requires `noise_sigma` to be provided")
            print("Aborting MCMC Calibration")
            exit(1)

        self.model = model
        self.data = data
        self.n_noise_parameters = n_noise_parameters
        self.calibrate_noise = calibrate_noise
        self.noise_sigma = noise_sigma
        self.distribution_type = distribution_type
        self.distribution_func = self._resolve_distribution_func()

    def eval(self, parameters) -> float:
        if self.calibrate_noise:
            model_parameters = parameters[:-self.n_noise_parameters]
            noise_sigma = np.asarray(parameters[-self.n_noise_parameters:])
            if any(sigma <= 0.0 for sigma in noise_sigma):
                return -np.inf
        else:
            noise_sigma = self.noise_sigma
            model_parameters = parameters
        try:
            prediction_mean = np.asarray(self.model([[*model_parameters]]))
        except Exception:
            return -np.inf
        if prediction_mean.shape != self.data.shape:
            raise ValueError("shape of model predictions does not match observations")
        log_likelihood  = self.distribution_func(prediction_mean, noise_sigma)
        return log_likelihood

    def _log_normal(self, prediction_mean, noise_sigma):
        variance = noise_sigma * noise_sigma
        return -0.5 * (np.log(2.0 * np.pi * variance) + ((self.data - prediction_mean) ** 2) / variance).sum()

    def _resolve_distribution_func(self):
        if self.distribution_type == "normal":
            return self._log_normal
        else:
            print(f"Distribution type {self.distribution_type} not yet supported for log_likelihood")
            print("Aborting MCMC Calibration")
            exit(1)


def make_prior_transform(prior: Prior):

    def prior_transform(u):
        theta = np.empty_like(u)
        for i, dist in enumerate(prior.distributions):
            theta[i] = dist.ppf(u[i])
        return theta
    return prior_transform

def perform_nested_sampling(log_likelihood_eval, prior_transform, ndim, nlive, dlogz=0.01):
    """Run dynesty's static Nested Sampler.
    # dynesty docs (NestedSampler API): https://dynesty.readthedocs.io/en/v3.0.0/

    Parameters
    ----------
    log_likelihood_eval : callable
        Maps a parameter vector -> log-likelihood.
    prior_transform : callable
        Maps the unit cube [0,1]^ndim -> parameter space.
    ndim : int
        Number of free parameters being sampled (Xi and Mu, so 2 for our case.)
    nlive : int
        Number of live points. More = more accurate evidence, slower.
    dlogz : float
        Stopping criterion: stop when the estimated remaining evidence
        contribution to ln(Z) drops below this threshold.
        We do not have this in emcee or rwmcmc, good thing to have.

    Returns
    -------
    results : dynesty.results.Results
        Contains weighted samples, logwt (log-weights), and logz (evidence).
    """
    sampler = dynesty.NestedSampler(
        log_likelihood_eval,
        prior_transform,
        ndim,
        nlive=nlive,
    )
    sampler.run_nested(dlogz=dlogz)
    return sampler.results

def parse_arguments():
    parser = argparse.ArgumentParser(description="MCMC Calibration with dynesty")
    parser.add_argument("--config", type=str, help="YAML configuration file")
    parser.add_argument("--data", type=str, help="Path to data CSV")
    parser.add_argument("--port", type=int, default=49152, help="Model server port")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    config_file = args.config
    data_file = args.data
    port = args.port

    with open(config_file) as f:
        config = yaml.safe_load(f)


    # Load Data 
    required_columns = config["calibration"]["data"]
    df = pd.read_csv(
        data_file,
        usecols = required_columns
        )
    data = df[required_columns].to_numpy()

    # Connect Model
    try:
        model_name = config["model"]["name"]
        model = umbridge.HTTPModel(f"http://localhost:{args.port}", model_name)
        print(f"Connected to model: {model_name}")
    except Exception as e:
        print(f"Error connecting to model: {e}")
        exit(1)


    # Build prior + likelihood (identical to emcee/rwmcmc; sampler-agnostic)
    # Define Prior Distributions
    prior = Prior(config["calibration"]["priors"],
                    parameters=config["calibration"]["parameters"],
                    noise_parameters=config["calibration"]["noise_parameters"],
                    calibrate_noise=config["calibration"]["calibrate_noise"])

    log_likelihood = LogLikelihood(model, data,
                                    calibrate_noise=config["calibration"]["calibrate_noise"],
                                    n_noise_parameters=len(config["calibration"]["noise_parameters"]),
                                    noise_sigma=config["calibration"].get("noise_sigma", None))

    # dynesty specific, prior transform for mapping from unit cube to parameter space
    prior_transform = make_prior_transform(prior)
    ndim = len(prior.all_parameters)

    sampler_params = config["calibration"]["sampler_params"]["dynesty"]
    nlive = sampler_params.get("nlive", 500)
    dlogz = sampler_params.get("dlogz", 0.01)

    print(f"Running dynesty: ndim={ndim}, nlive={nlive}, dlogz={dlogz}")

    results = perform_nested_sampling(
        log_likelihood_eval=log_likelihood.eval,
        prior_transform=prior_transform,
        ndim=ndim,
        nlive=nlive,
        dlogz=dlogz,
    )

    print(f"Nested sampling completed. logz = {results.logz[-1]:.3f} +/- {results.logzerr[-1]:.3f}")


    ndim = len(prior.all_parameters)

    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])  # each point has a weight, here we normalize them to sum to 1
    weights /= weights.sum()    # just to be safe...
    trace = resample_equal(samples, weights)            # (n_equal, ndim), we are just mimicking MCMC output, so we can use the same plotting functions (as if in the rwmh the more weighted samples are more likely to be drawn, we can resample them to get an unweighted trace)

    # 1. mcmc_output.npz
    samples_3d = trace.reshape(1, -1, ndim)             # mimic (nchains=1, nsteps, ndim), for reporting. 
    lnprob = np.zeros((1, trace.shape[0]))
    np.savez("mcmc_output.npz", trace=trace, samples=samples_3d, lnprob=lnprob,
             logz=results.logz[-1], logzerr=results.logzerr[-1])    #  logz is the log evidence, logzerr is the uncertainty in the log evidence (error in the log of the evidence)
    print("Results saved to mcmc_output.npz")

    # 2. corner_plot.png
    corner_plot = corner.corner(trace, labels=prior.all_parameters, show_titles=True)
    corner_plot.savefig("corner_plot")
    print("Corner plot saved to corner_plot.png")

    # 3. trace.npy
    np.save("trace.npy", trace)
    print("Samples saved to trace.npy")

    # 4. mcmc_idata.nc, single "chain" (nested sampling has no MCMC chains)
    posterior = {name: trace[:, i].reshape(1, -1) for i, name in enumerate(prior.all_parameters)}
    idata = az.from_dict(posterior=posterior)
    idata.to_netcdf("mcmc_idata.nc")
    print("Calibration inference data saved to mcmc_idata.nc")