import yaml
import numpy as np
import pandas as pd
from scipy.stats import uniform, norm, truncnorm
# from concurrent.futures import ThreadPoolExecutor
# from multiprocessing import Pool
# import contextlib
import umbridge
import argparse
# import os
import corner
import arviz as az
from rwmcmc.samplers import random_walk_metropolis_hastings



class Prior:
    @staticmethod
    def _distributions_from_config(prior_config: dict):
        name = prior_config["name"]
        distribution = prior_config["distribution"]

        if distribution["type"] not in ["uniform", "normal", "truncated_normal"]:
            print(f"Distribution type {distribution["type"]} not yet supported")
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
                    f"\nA {distribution["type"]} prior requires `loc` and `scale` attributes")
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


class LogPrior:
    def __init__(self, prior: Prior):
        self.prior = prior

    def eval(self, parameters) -> float:
        log_p = 0.0
        for i, theta in enumerate(parameters):
            support = self.prior.distributions[i].support()
            if not (support[0] <= theta <= support[1]):
                return -np.inf
            log_p += self.prior.distributions[i].logpdf(theta)
        return log_p


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
            print(prediction_mean)
        except Exception:
            print(f"MODEL_ERROR at parameters {[*model_parameters]}")
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


class LogPosterior:
    def __init__(self, log_prior: LogPrior, log_likelihood: LogLikelihood):
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood

    def eval(self, parameters) -> float:
        log_prior = self.log_prior.eval(parameters)
        if not np.isfinite(log_prior):
            print("Prior_INF")
            return log_prior
        return log_prior + self.log_likelihood.eval(parameters)


""" 
def initialize_walkers(nwalkers: int, prior: Prior) -> np.ndarray:
    nparameters = len(prior.distributions)
    initial_positions = np.zeros((nwalkers, nparameters))

    for i in range(nparameters):
        initial_positions[:, i] = prior.distributions[i].rvs(size=nwalkers)

    return initial_positions

POOL_TYPES = ("serial", "thread", "process")
 """

# Here I renamed the walkers function with chains.

def initialize_chains(nchains: int, prior: Prior) -> np.ndarray:
    nparameters = len(prior.distributions)
    initial_positions = np.zeros((nchains, nparameters))

    for i in range(nparameters):
        initial_positions[:, i] = prior.distributions[i].rvs(size=nchains)

    return initial_positions

""" 
@contextlib.contextmanager
def _build_pool(pool_type: str, n_workers: int):
    if pool_type == "thread":
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            yield pool
    elif pool_type == "process":
        with Pool(processes=n_workers) as pool:
            yield pool
    else:
        yield None
 """

""" def perform_mcmc(prior, log_posterior, nchains=50, nburn=2500, nsteps=5000, n_workers=1, pool_type="serial"):
    if pool_type not in POOL_TYPES:
        print(f"pool_type '{pool_type}' not supported. Choose from: {POOL_TYPES}")
        exit(1)

    # Initialize Chains
    initial_positions = initialize_chains(nchains, prior)

    # Setup Sampler
    ndim = len(prior.distributions)
    with _build_pool(pool_type, n_workers) as pool:
        sampler = emcee.EnsembleSampler(nchains, ndim, log_posterior, pool=pool)

        # Run Calibration
        sampler.run_mcmc(initial_positions, nsteps, progress=True)

    # Extract Results (post burn-in)
    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim)
    lnprob = sampler.lnprobability[:, nburn:]
    samples = sampler.chain[:, nburn:, :]

    return trace, sampler, lnprob, samples """

def perform_mcmc(prior, log_posterior_eval, nchains=100, nsteps=2000, step_size=0.1, rng=None)-> tuple:
    """
    Here, we want to mimic the emcee with number of chains. For each chain, we run the random walk metropolis hastings sampler independently inside a for loop. No parallelization, so it might be slower than the classic rwmh.
    """
    
    if rng is None: rng = np.random.default_rng()
    step_size = np.asarray(step_size)

    initial_positions = initialize_chains(nchains, prior)
    ndim = len(prior.distributions)

    # We will run a loop so we need to pre-allocate the samples_all, accepted_all.

    samples_all = np.zeros((nchains, nsteps, ndim))
    accepted_all = np.zeros((nchains, nsteps), dtype=bool)

    # Run MCMC for each chain
    for chain_i in range(nchains):
        chain_rng = np.random.default_rng(rng.integers(0, 1e9))  # Create a separate RNG for each chain to make them statistically independent (we're just reducing the risks of correlations between chains, not guaranteeing it)

        samples, accepted = random_walk_metropolis_hastings(
            target_log_pdf=log_posterior_eval,
            x0=initial_positions[chain_i],
            n_samples=nsteps,
            step_size=step_size,
            rng=chain_rng
        )
        samples_all[chain_i] = samples
        accepted_all[chain_i] = accepted
        print(f"Chain {chain_i + 1}/{nchains} done. Acceptance rate: {accepted.mean():.2%}")

    return samples_all, accepted_all


def parse_arguments():
    parser = argparse.ArgumentParser(description='MCMC Calibration with `rwmh`')
    parser.add_argument('--config', type=str,
                        help='YAML file for Configuration Parameters')
    parser.add_argument('--data', type=str,
                        help='Path to Data File')
    parser.add_argument('--port', type=int, default=49152,
                       help='Server port')
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

    # Define Prior Distributions
    prior = Prior(config["calibration"]["priors"],
                    parameters=config["calibration"]["parameters"],
                    noise_parameters=config["calibration"]["noise_parameters"],
                    calibrate_noise=config["calibration"]["calibrate_noise"])

    log_prior = LogPrior(prior)
    log_likelihood = LogLikelihood(model, data,
                                    calibrate_noise=config["calibration"]["calibrate_noise"],
                                    n_noise_parameters=len(config["calibration"]["noise_parameters"]),
                                    noise_sigma=config["calibration"].get("noise_sigma", None))
    log_posterior = LogPosterior(log_prior, log_likelihood)

    """     
        # Perform MCMC Calibration
        trace, sampler, lnprob, samples = perform_mcmc(prior, log_posterior.eval,
                    nwalkers=config["calibration"]["nwalkers"],
                    nburn=config["calibration"]["nburn"],
                    nsteps=config["calibration"]["nsteps"],
                    n_workers=config["calibration"].get("n_workers", 1),
                    pool_type=config["calibration"].get("pool_type", "serial"))
    """

    # Perform MCMC Calibration
    nchains = config["calibration"]["nwalkers"] # this is basically nchains, not nwalkers but I want to keep the same naming within the config in params.yml.
    nsteps = config["calibration"]["nsteps"]
    nburn = config["calibration"]["nburn"]
    step_size_cfg = config["calibration"].get("step_size", 0.1)

    print(f"Running {nchains} chains x {nsteps} steps ({nburn} burn-in)")
    print(f"step_size: {step_size_cfg}")

    samples_all, accepted_all = perform_mcmc(
        prior=prior,
        log_posterior_eval=log_posterior.eval,
        nchains=nchains,
        nsteps=nsteps,
        step_size=step_size_cfg,
    )

    print(f"MCMC completed. Samples shape: {samples_all.shape}")
    print(f"Overall acceptance rate: {accepted_all.mean():.2%}")
    
    
    # SAVE RESULTS
    # data_basename, ext = os.path.splitext(args.data) # I deleted the .os import, I do not think that I will need that.
    ndim = len(prior.all_parameters)

    # Here, we need to represent our samples in the same format as emcee, nchains*nsteps, ndim.
    trace = samples_all[:, nburn:, :].reshape(-1, ndim)

    samples_post_burn = samples_all[:, nburn:, :]

    # Just a placeholder, otherwise we will get an error.
    lnprob = np.zeros((nchains, nsteps-nburn))

    np.savez(f"mcmc_output.npz", 
                trace=trace, samples=samples_post_burn, lnprob=lnprob)
    print(f"Results saved to mcmc_output.npz")

    corner_plot = corner.corner(trace, labels=prior.all_parameters, show_titles=True)
    corner_plot.savefig(f"corner_plot")
    print(f"Corner Plot saved to corner_plot.png")

    # Save samples as .npy file
    np.save(f"trace.npy", trace)
    print(f"Samples saved to trace.npy")
    """ 
        # Idata for Diagnostics with Arviz
        idata = az.from_emcee(sampler, var_names = prior.all_parameters)
        idata.to_netcdf(f"mcmc_idata.nc")
        print(f"Calibration inference data saved to mcmc_idata.nc")
    """

    # Idata for Diagnostics with Arviz - for rwmcmc we do not have the same sampler, we need to create a custom InferenceData object
    posterior = dict()
    for i, name in enumerate(prior.all_parameters):
        posterior[name] = samples_all[:, :, i]
    
    idata = az.from_dict(posterior=posterior)
    idata.to_netcdf("mcmc_idata.nc")
    print(f"Calibration inference data saved to mcmc_idata.nc")

