import umbridge
import numpy as np
import os
import argparse
import psimpy
import pickle
import yaml

# Workaround for rpy2 >= 3.6.x deprecating numpy2ri.activate().
# psimpy calls activate() at import time; patch it to use the new API before
# psimpy is imported (which happens implicitly when the pickle is loaded).
# Remove once https://git.rwth-aachen.de/mbd/psimpy/-/work_items/4 is fixed upstream.
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter


def _rpy2_converter():
    """Return the rpy2 converter context manager."""
    return localconverter(default_converter + numpy2ri.converter)


numpy2ri.activate = lambda: _rpy2_converter().__enter__()

class SurrogateModel(umbridge.Model):
    
    
    def __init__(self, name, surrogate_model, input_dim, output_dim, debug=False):
        self.name = name
        self.model = surrogate_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.evaluate_func = self.debug_evaluate if debug else self.evaluate

    def get_input_sizes(self, config):
        return self.input_dim

    def get_output_sizes(self, config):
        return self.output_dim

    def __call__(self, parameters, config):
        try:
            return self.evaluate_func(parameters)
        except Exception as e:
            print(f"Error in model: {e}")
            raise e

    def supports_evaluate(self):
        return True
    
    def debug_evaluate(self, parameters):
        # Input parameter validation
        if len(parameters[0]) != self.input_dim[0]:
            raise ValueError(f"Model expects {self.input_dim[0]} parameters, got {len(parameters[0])}")
        x = np.asarray(parameters)
        with _rpy2_converter(): # Remove once https://git.rwth-aachen.de/mbd/psimpy/-/work_items/4 is fixed upstream.
            output = self.model.predict(x)
        # Output shape validation: ScalarGaSP returns (ntest, 4) with [mean, lower, upper, std]
        expected_shape = (len(parameters), 4)
        if output.shape != expected_shape:
            raise ValueError(f"Expected output shape {expected_shape}, got {output.shape}")
        # Check mean column is finite
        if not np.all(np.isfinite(output[:, 0])):
            raise ValueError(f"Model output means are not all finite")
        return [output[:, 0].tolist()]

    def evaluate(self, parameters):
        x = np.asarray(parameters)
        with _rpy2_converter(): # Remove once https://git.rwth-aachen.de/mbd/psimpy/-/work_items/4 is fixed upstream.
            output = self.model.predict(x)
        return [output[:, 0].tolist()]
        

def parse_arguments():
    parser = argparse.ArgumentParser(description='Surrogate Model UMBridge Server')
    parser.add_argument('--port', type=int, default=49152,
                       help='Server port')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to config.yml file')
    return parser.parse_args()

def main():
    args = parse_arguments()
    port = args.port
    config_path = args.config

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    name = model_config['name']
    model_file = model_config['file']
    input_dim = model_config['input_dim']
    output_dim = model_config['output_dim']
    debug = model_config.get('debug', False)
    max_workers = model_config.get('max_workers', 1)

    print(f"Loading model from {model_file}")
    with open(model_file, 'rb') as f:
        surrogate_model = pickle.load(f)

    print(f"Starting UMBridge server on port {port}")
    print(f"Model: {name}")

    # Create the model instance
    model = SurrogateModel(name, surrogate_model, input_dim, output_dim, debug=debug)
    print(f"Successfully created model: {model.name}")

    # Serve the model
    umbridge.serve_models([model], port, max_workers=max_workers)

if __name__ == "__main__":
    main()