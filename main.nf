#!/usr/bin/env nextflow
import groovy.yaml.YamlBuilder


workflow bpc_massflowIA {
    def model = params.model.name
    def umbridge_port = Math.abs( new Random().nextInt() % (31767 - 16384) ) + 16384 // Safe range [16384, 31767], excludes kernel ephemeral [32768, 60999]


    COLLECT_INPUTS()

    SETUP_UM_IPC () 
    
    SERVE_MODEL(
        script = file("$moduleDir/model/surrogate_model_server.py"),
        config = params,
        name = model,
        surrogate_model = COLLECT_INPUTS.out.surrogate_model,
        umbridge_port = umbridge_port,
        um_highway = SETUP_UM_IPC.out
    )
    

    MCMC_CALIBRATION_EMCEE(
        script = file("$moduleDir/mcmc/emcee/run_calibration.py"),
        config = params,
        data = COLLECT_INPUTS.out.ground_truth,
        um_highway = SETUP_UM_IPC.out
    )

    RUN_DIAGNOSTICS(
      script = file("$moduleDir/diagnostics/run_diagnostics.py"),
      mcmc_idata = MCMC_CALIBRATION_EMCEE.out.mcmc_idata,
      outdir = "diagnostics",
    )


    BUNDLE_OUTPUTS(
        experiment_params = params,
        mcmc_output = MCMC_CALIBRATION_EMCEE.out.mcmc_output,
        mcmc_corner_plot = MCMC_CALIBRATION_EMCEE.out.mcmc_corner_plot,
        mcmc_trace = MCMC_CALIBRATION_EMCEE.out.mcmc_trace,
        mcmc_idata = MCMC_CALIBRATION_EMCEE.out.mcmc_idata,
        mcmc_diagnostics = RUN_DIAGNOSTICS.out
    )

    GENERATE_REPORT(
        script = file("$moduleDir/report/generate_report.py"),
        bundle_dir = BUNDLE_OUTPUTS.out.bundle_dir
    )

    emit:
    bundle = BUNDLE_OUTPUTS.out.bundle_dir
}

process COLLECT_INPUTS {
    script:
    """
    cp $moduleDir/model/surrogate_voellmy.pkl .
    cp $moduleDir/mcmc/ground_truth.csv .
    """
    output:
    path "ground_truth.csv", emit: ground_truth
    path "surrogate_voellmy.pkl", emit: surrogate_model
}

process SETUP_UM_IPC {

    script:
    """
    mkdir comm
    mkdir comm/model_info
    mkdir comm/uq_info
    mkdir comm/umbridge_port
    """

    output:
    path "comm"
}

process SERVE_MODEL {
    conda "$moduleDir/model/environment.yml"
    cache 'lenient'
    errorStrategy 'retry'
    maxRetries 3

    input:
    path script
    val config
    val name
    path surrogate_model
    val umbridge_port
    path um_highway

    script:
    def parameters = new groovy.yaml.YamlBuilder()
    parameters(config)
    """
    #!/bin/bash
    UMBRIDGE_PORT=${umbridge_port+100*(task.attempt-1)}

    echo "${parameters.toString()}" > _server_config.yml

    # Abort if port already used (success means something is listening)
    if bash -c "echo > /dev/tcp/localhost/\$UMBRIDGE_PORT" 2>/dev/null; then
        echo "Port \$UMBRIDGE_PORT already in use"
        exit 1
    fi

    # Start model server
    python ${script} --config _server_config.yml --port \$UMBRIDGE_PORT &
    SERVER_PID=\$!
    trap 'kill \$SERVER_PID 2>/dev/null || true' EXIT INT TERM
    echo "Model Server PID: \$SERVER_PID (trying port \$UMBRIDGE_PORT)"

    # Wait for model server to start 
    while ! bash -c "echo > /dev/tcp/localhost/\$UMBRIDGE_PORT" 2>/dev/null; do
        sleep 1
    done

    touch $um_highway/model_info/READY 
    echo "Model ${name} ready" > $um_highway/model_info/READY

    touch $um_highway/umbridge_port/\$UMBRIDGE_PORT
    echo "Model server is running on port \$UMBRIDGE_PORT"

    # Monitor the status 
    echo "Waiting for UQ to complete..."
    until [ -e $um_highway/uq_info/DONE ]; do
        sleep 1
    done

    # Stop the model server when the signal is received
    kill \$SERVER_PID 2>/dev/null || true
    echo "Model server on port \$UMBRIDGE_PORT stopped."

    """
}

process MCMC_CALIBRATION_EMCEE {
    conda "$moduleDir/mcmc/emcee/environment.yml"
    cache 'lenient'
    
    input:
    path script
    val config
    path data
    path um_highway


    script:
    def parameters = new YamlBuilder()
    parameters(config)
    """
    echo "${parameters.toString()}" > _params.yml

    echo "Waiting for model server to start..."
    until [ -e $um_highway/model_info/READY ]; do
        sleep 1
    done
    cat $um_highway/model_info/READY
    
    MODEL_PORT=\$(ls $um_highway/umbridge_port/ | head -n 1)
    echo "Model server is running on port \${MODEL_PORT}"
    
    python ${script}  --config _params.yml --data ${data} --port \${MODEL_PORT}
    
    touch $um_highway/uq_info/DONE # signal to stop the model server
    """

    output:
    path "mcmc_output.npz", emit: mcmc_output
    path "corner_plot.png", emit: mcmc_corner_plot
    path "trace.npy", emit: mcmc_trace
    path "mcmc_idata.nc", emit: mcmc_idata

}

process RUN_DIAGNOSTICS {
    conda "$moduleDir/diagnostics/environment.yml"

    input:
    path script
    path mcmc_idata
    val outdir

    script:
    """
    #!/bin/bash
    python3 ${script} --idata-path ${mcmc_idata} --output-dir "${outdir}" 
    """

    output:
    path "${outdir}"


}

process BUNDLE_OUTPUTS {
    publishDir "$moduleDir/outputs", mode: 'copy'

    input:
    val experiment_params
    path mcmc_output
    path mcmc_corner_plot
    path mcmc_trace
    path mcmc_idata
    path mcmc_diagnostics

    script:
    def parameters = new YamlBuilder()
    parameters(experiment_params)
    """
    #!/bin/bash
    echo "${parameters.toString()}" > _params.yml

    mkdir "${params.model.name}_${workflow.sessionId}"

    cp _params.yml "${params.model.name}_${workflow.sessionId}"
    cp ${mcmc_output} "${params.model.name}_${workflow.sessionId}/"
    cp ${mcmc_corner_plot} "${params.model.name}_${workflow.sessionId}/"
    cp ${mcmc_trace} "${params.model.name}_${workflow.sessionId}/"
    cp ${mcmc_idata} "${params.model.name}_${workflow.sessionId}/"
    cp -r ${mcmc_diagnostics} "${params.model.name}_${workflow.sessionId}/"
    """

    output:
    path "${params.model.name}_${workflow.sessionId}", emit: bundle_dir
}

process GENERATE_REPORT {
    conda "$moduleDir/report/environment.yml"
    publishDir "$moduleDir/outputs/${bundle_dir.name}", mode: 'copy'

    input:
    path script
    path bundle_dir

    output:
    path "report.pdf"

    script:
    """
    #!/bin/bash
    python3 ${script} \\
        --bundle-dir ${bundle_dir} \\
        --output report.pdf
    """
}

workflow {
    bpc_massflowIA()
}