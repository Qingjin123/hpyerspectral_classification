import multiprocessing
import logging
import subprocess
import os

def setup_logger(identifier, log_file):
    """Setup logger with a specific log file."""
    logger = logging.getLogger(identifier)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    return logger

def train_specific(model_script, model_name, data, gnn, parameter, value):
    identifier = f"{model_name}_{data}_{gnn}_{parameter}_{value}"
    logger = setup_logger(identifier, f'logs/{identifier}.log')
    logger.info(f"Starting training {model_name} on {data} using {gnn} with {parameter}={value}")
    cmd = f'python {model_script} --model_name {model_name} --data_name {data} --gnn_function_name {gnn} --{parameter} {value}'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        logger.info(f"Training completed successfully for {model_name} on {data} using {gnn} with {parameter}={value}")
    else:
        logger.error(f"Training failed for {model_name} on {data} using {gnn} with {parameter}={value}\n{stderr.decode()}")
    logger.info(stdout.decode())

def train_multiple_configs(parameter, values, model_script, model_name, data, gnn):
    processes = []
    for value in values:
        p = multiprocessing.Process(target=train_specific, args=(model_script, model_name, data, gnn, parameter, value))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == '__main__':
    # Setup logs directory
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Example usage
    parameters = {
        'n_segments': [30, 40, 50, 70, 100],
        'train_nums': [1, 3, 5, 10, 15, 20, 30]
    }
    model_script = 'train_script.py'
    model_name = 'example_model'
    data = 'example_data'
    gnn = 'example_gnn'
    
    for parameter, values in parameters.items():
        train_multiple_configs(parameter, values, model_script, model_name, data, gnn)
