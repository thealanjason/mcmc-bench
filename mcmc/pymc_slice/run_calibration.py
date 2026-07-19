import yaml
import numpy as np
import pandas as pd




import umbridge
import argparse

import corner

import pymc as pm
import pytensor.tensor as pt

from pytensor.graph.op import Op


class Prior:
    @staticmethod
    def _distributions_from_config(prior_config: dict):
        name = prior_config["name"]
        distribution = prior_config["distribution"]
        attributes = distribution["attribute"]
        distribution_type = distribution["type"]

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

            return pm.Uniform(
                name,
                lower=attributes["lower_bound"],
                upper=attributes["upper_bound"],
            )

        elif distribution["type"] == "normal":
            if not all(attribute in distribution["attribute"] for attribute in ("loc", "scale")):
                print(f"Incorrect definition of prior for parameter '{name}'",
                    f"\nA {distribution["type"]} prior requires `loc` and `scale` attributes")
                print("Aborting MCMC Calibration")
                exit(1)
            return pm.Normal(name, mu=attributes["loc"], sigma=attributes["scale"])
        
        elif distribution["type"] == "truncated_normal":
            if not (all(attribute in distribution["attribute"] for attribute in ("loc", "scale"))
                    and any(attribute in distribution["attribute"] for attribute in ("lower_bound", "upper_bound"))):
                print(f"Incorrect definition of prior for parameter '{name}'",
                    f"\nA {distribution["type"]} prior requires `loc`, `scale`, `lower_bound` and/or `upper_bound` attributes")
                print("Aborting MCMC Calibration")
                exit(1)
            return pm.TruncatedNormal(
                name,
                mu=attributes["loc"],
                sigma=attributes["scale"],
                lower=attributes.get("lower_bound",-np.inf),
                upper=attributes.get("upper_bound",np.inf),
            )

    def __init__(self, config: dict, parameters: list, noise_parameters: list, calibrate_noise: bool = False):
        self.config = config
        self.parameters = parameters
        self.noise_parameters = noise_parameters
        self.calibrate_noise = calibrate_noise

        self.config_by_name = {prior["name"]: prior for prior in config}
        self.all_parameters = list(parameters)
        if calibrate_noise:
            self.all_parameters += noise_parameters
        
    
    def create_variables(self):
        """Create the random variables inside a PyMC model."""
        return[self._distributions_from_config(self.config_by_name[name]) for name in self.all_parameters]


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

class UMBridgeLogLikelihood(Op):
    """Expose the numerical UM-Bridge likelihood as a scalar PyTensor Op: https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html"""

    itypes = [pt.dvector]
    otypes = [pt.dscalar]

    def __init__(self, log_likelihood: LogLikelihood):
        self.log_likelihood = log_likelihood

    def perform(self, node, inputs, outputs):
        (parameters,) = inputs
        loglike_eval = self.log_likelihood.eval(parameters)
        outputs[0][0] = np.asarray(loglike_eval, dtype=np.float64)


def perform_mcmc(
    prior: Prior,
    log_likelihood: LogLikelihood,
    draws: int,
    tune: int,
    chains: int = 4,
    cores: int = 1,
    random_seed: int | None = None,
):
    """Run gradient-free Slice sampling and return repository-compatible arrays."""

    with pm.Model():
        variables = prior.create_variables()
        parameters = pt.stack(variables).astype("float64")

        loglike_op = UMBridgeLogLikelihood(log_likelihood)
        pm.Potential("external_loglikelihood", loglike_op(parameters))

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            step=pm.Slice(),
            random_seed=random_seed,
            return_inferencedata=True,
        )

    samples = np.stack(
        [np.asarray(idata.posterior[name]) for name in prior.all_parameters],
        axis=-1,
    )
    trace = samples.reshape(-1, len(prior.all_parameters))

    if "lp" in idata.sample_stats:
        lnprob = np.asarray(idata.sample_stats["lp"])
    else:
        lnprob = np.full(samples.shape[:2], np.nan)

    return trace, idata, lnprob, samples



def parse_arguments():
    parser = argparse.ArgumentParser(description='MCMC Calibration with PyMC Slice')
    parser.add_argument('--config', type=str,
                        help='YAML file for Configuration Parameters')
    parser.add_argument('--data', type=str,
                        help='Path to Data File')
    parser.add_argument('--port', type=int, default=49152,
                       help='Server port')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)


    calibration = config["calibration"]

    # Load Data 
    required_columns = calibration["data"]
    df = pd.read_csv(
        args.data,
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
    prior = Prior(
        config["calibration"]["priors"],
                    parameters=config["calibration"]["parameters"],
                    noise_parameters=config["calibration"]["noise_parameters"],
                    calibrate_noise=config["calibration"]["calibrate_noise"])


    log_likelihood = LogLikelihood(model, data,
                                    calibrate_noise=config["calibration"]["calibrate_noise"],
                                    n_noise_parameters=len(config["calibration"]["noise_parameters"]),
                                    noise_sigma=config["calibration"].get("noise_sigma", None))


    sampler_config = calibration["sampler_params"].get("pymc_slice", {})
    tune = sampler_config.get("tune", calibration["nburn"])

    # emcee treats nsteps as including burn-in. Preserve its retained-sample
    # count unless an explicit PyMC `draws` value is configured.
    default_draws = calibration["nsteps"] - calibration["nburn"]
    draws = sampler_config.get("draws", default_draws)
    if draws <= 0:
        raise ValueError("PyMC `draws` must be greater than zero")

    trace, idata, lnprob, samples = perform_mcmc(
        prior,
        log_likelihood,
        draws=draws,
        tune=tune,
        chains=sampler_config.get("chains", 4),
        cores=sampler_config.get("cores", 1),
        random_seed=sampler_config.get("random_seed"),
    )



    print(f"MCMC completed. Trace shape: {trace.shape}")

    # Save results
    np.savez("mcmc_output.npz", trace=trace, samples=samples, lnprob=lnprob)
    print("Results saved to mcmc_output.npz")

    corner_plot = corner.corner(
        trace, labels=prior.all_parameters, show_titles=True
    )
    corner_plot.savefig("corner_plot.png")
    print("Corner plot saved to corner_plot.png")

    np.save("trace.npy", trace)
    print("Samples saved to trace.npy")

    idata.to_netcdf("mcmc_idata.nc")
    print("Calibration inference data saved to mcmc_idata.nc")



