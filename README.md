# Benchmarking Sampling Methods for Probabilistic Calibration in Gravity-Driven Mass Flow Modeling

## Motivation 
Bayesian parameter calibration of computational models can be performed using a variety of sampling methods. Each method differs in algorithm design, implementation details, accuracy, speed, and resource consumption. Some implementations are domain-specific, leverage modern hardware such as GPUs, or are tied to a particular programming language. These differences make it difficult to systematically compare methods and select the most suitable one for a given application.

## Goal
The goal of this project is to compare and gain insights into the behavior of different sampling methods in the context of Bayesian parameter calibration of mass flow simulators.

## Tasks
- \>\> Implement a selection of sampling methods and apply them to a representative mass flow simulation model.

- \>\> Compare the methods across key criteria such as accuracy, convergence, speed, and resource consumption.

- \>\> Document findings and derive insights into the strengths and limitations of each method in the context of Bayesian parameter calibration.

## Getting started

We use `nextflow` to run this usecase which is specified through the main.nf file. 

1. Create and activate the conda environment
```
conda env create -f environment.yml
conda activate mcmc-bench
```

2. Run the workflow

```
nextflow run main.nf -params-file params.yml --config_file params.yml
```

### Selecting a sampling method

The sampling method is chosen via the `calibration.sampler` field in
`params.yml`. Five methods are currently available:
```yaml
calibration:
  sampler: rwmcmc   # options: rwmcmc | emcee | pymc_slice | pymc_smc | dynesty
```
Each sampler implementation reads its own parameters from `calibration.sampler_params`,
so you can configure them independently (e.g. `nwalkers` for emcee,
`step_size` for rwmcmc).

### Running all sampling methods

The five sampler implementations can be executed sequentially with:

```bash
NXF_VER=25.10.4 ./run_all_samplers.sh
```

The script restores `params.yml` after execution and writes the individual
sampler logs to `benchmark_logs/`.

### Outputs

Each workflow execution creates a session-specific output bundle under
`outputs/`. The bundle contains the run configuration, posterior samples,
diagnostic results, plots, model-call count, and an automatically generated
PDF report.

### Parallelization (dynesty)

dynesty supports parallel likelihood evaluations via the
`sampler_params.dynesty.nprocs` field in `params.yml`. Setting
`nprocs > 1` distributes likelihood calls across worker processes
using Python's `multiprocessing` pool; `nprocs = 1` runs serially (default).

## Report

The evaluation report and preserved benchmark results are available in the [mcmc-bench-report repository](https://github.com/korkmazgoz-caglar/mcmc-bench-report).

## References
[1] Chi-Feng, H.: MCMC Playground – Interactive Visualization of Markov Chain Monte Carlo Algorithms, https://chi-feng.github.io/mcmc-demo/app.html?algorithm=HamiltonianMC&target=banana (last access: 23 April 2026).

[2] van Ravenzwaaij, D., Cassey, P., and Brown, S. D.: A simple introduction to Markov Chain Monte Carlo sampling, Psychon. Bull. Rev., 25, 143–154, https://doi.org/10.3758/s13423-016-1015-8, 2018.

[3] The DarkMachines High Dimensional Sampling Group, Albert, J., Balázs, C., Fowlie, A., Handley, W., Hunt-Smith, N., Ruiz de Austri, R., and White, M.: A comparison of Bayesian sampling algorithms for high-dimensional particle physics and cosmology applications, Comput. Phys. Commun., https://doi.org/10.1016/j.cpc.2025.109756, 2025.

[4] Lim, Y.: A review of the Bayesian approach with the MCMC and the HMC as a competitor of classical likelihood statistics for pharmacometricians, Transl. Clin. Pharmacol., 31, 69–79, https://doi.org/10.12793/tcp.2023.31.e9, 2023.


