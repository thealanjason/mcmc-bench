#!/usr/bin/env nextflow

def toYaml(data) {
    def yb = new groovy.yaml.YamlBuilder()
    yb.call(data)
    return yb.toString()
}

def _workflowMeta(f) {
    def cfg = new groovy.yaml.YamlSlurper().parseText(f.text) as Map
    def bundle_name = "${cfg.model.name}_${workflow.sessionId}".toString()
    return [cfg, bundle_name, toYaml(cfg)]
}


workflow bpc_massflowIA {

    take:
    config_file

    main:
    def umbridge_port = Math.abs( new Random().nextInt() % (31767 - 16384) ) + 16384 // Safe range [16384, 31767], excludes kernel ephemeral [32768, 60999]

    // Single source of truth: all per-experiment metadata derived once from the config
    workflow_meta = config_file.map { f -> _workflowMeta(f) }
    // workflow_meta: [cfg, bundle_name, params_yaml]

    model       = workflow_meta.map { row -> row[0].model.name }
    bundle_name = workflow_meta.map { row -> row[1] }
    params_yaml = workflow_meta.map { row -> row[2] }

    COLLECT_INPUTS()

    SETUP_UM_IPC()

    SERVE_MODEL(
        file("$moduleDir/model/surrogate_model_server.py"),
        params_yaml,
        model,
        COLLECT_INPUTS.out.surrogate_model,
        umbridge_port,
        SETUP_UM_IPC.out
    )

    MCMC_CALIBRATION_EMCEE(
        file("$moduleDir/mcmc/emcee/run_calibration.py"),
        params_yaml,
        COLLECT_INPUTS.out.ground_truth,
        SETUP_UM_IPC.out
    )

    RUN_DIAGNOSTICS(
        file("$moduleDir/diagnostics/run_diagnostics.py"),
        MCMC_CALIBRATION_EMCEE.out.mcmc_idata,
        "diagnostics"
    )

    BUNDLE_OUTPUTS(
        "$moduleDir/outputs",
        bundle_name,
        params_yaml,
        MCMC_CALIBRATION_EMCEE.out.mcmc_output,
        MCMC_CALIBRATION_EMCEE.out.mcmc_corner_plot,
        MCMC_CALIBRATION_EMCEE.out.mcmc_trace,
        MCMC_CALIBRATION_EMCEE.out.mcmc_idata,
        RUN_DIAGNOSTICS.out
    )

    GENERATE_REPORT(
        file("$moduleDir/report/generate_report.py"),
        "$moduleDir/outputs",
        BUNDLE_OUTPUTS.out.bundle_dir
    )

    emit:
    bundle = BUNDLE_OUTPUTS.out.bundle_dir
}

process COLLECT_INPUTS {
    output:
    path "ground_truth.csv", emit: ground_truth
    path "surrogate_voellmy.pkl", emit: surrogate_model

    script:
    """
    cp $moduleDir/model/surrogate_voellmy.pkl .
    cp $moduleDir/mcmc/ground_truth.csv .
    """
}

process SETUP_UM_IPC {
    output:
    path "comm"

    script:
    """
    mkdir comm
    mkdir comm/model_info
    mkdir comm/uq_info
    mkdir comm/umbridge_port
    """
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
    """
    #!/bin/bash
    UMBRIDGE_PORT=${umbridge_port+100*(task.attempt-1)}

    echo "${config}" > _server_config.yml

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

    output:
    path "mcmc_output.npz", emit: mcmc_output
    path "corner_plot.png", emit: mcmc_corner_plot
    path "trace.npy",       emit: mcmc_trace
    path "mcmc_idata.nc",   emit: mcmc_idata

    script:
    """
    echo "${config}" > _params.yml

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
}

process RUN_DIAGNOSTICS {
    conda "$moduleDir/diagnostics/environment.yml"

    input:
    path script
    path mcmc_idata
    val outdir

    output:
    path "${outdir}"

    script:
    """
    #!/bin/bash
    python3 ${script} --idata-path ${mcmc_idata} --output-dir "${outdir}"
    """
}

process BUNDLE_OUTPUTS {

    input:
    val output_base_dir
    val bundle_name
    val params_yaml
    path mcmc_output
    path mcmc_corner_plot
    path mcmc_trace
    path mcmc_idata
    path mcmc_diagnostics

    output:
    path "${bundle_name}", emit: bundle_dir

    script:
    """
    #!/bin/bash
    echo "${params_yaml}" > _params.yml

    mkdir "${bundle_name}"

    cp _params.yml "${bundle_name}"
    cp ${mcmc_output} "${bundle_name}/"
    cp ${mcmc_corner_plot} "${bundle_name}/"
    cp ${mcmc_trace} "${bundle_name}/"
    cp ${mcmc_idata} "${bundle_name}/"
    cp -r ${mcmc_diagnostics} "${bundle_name}/"

    mkdir -p "${output_base_dir}"
    cp -r "${bundle_name}" "${output_base_dir}/"
    """
}

process GENERATE_REPORT {
    conda "$moduleDir/report/environment.yml"

    input:
    path script
    val output_base_dir
    path bundle_dir

    output:
    path "report.pdf"

    script:
    """
    #!/bin/bash
    python3 ${script} \\
        --bundle-dir ${bundle_dir} \\
        --output report.pdf
    cp report.pdf "${output_base_dir}/${bundle_dir.name}/"
    """
}

workflow {
    bpc_massflowIA(
        Channel.value(file(params.config_file))
    )
}
