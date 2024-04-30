import multiprocessing
import logging
import subprocess

def setup_logger(data, gnn, log_file):
    """Function setup as many loggers as you want with a specific log file"""
    logger = logging.getLogger(f'{data}_{gnn}')
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    return logger

def train_model(model_script, model_name, data, gnn):
    logger = setup_logger(data, gnn, f'logs/{model_name}_{data}_{gnn}.log')
    logger.info(f"Starting training {model_name} on {data} using {gnn}")
    cmd = f'python {model_script} --model_name {model_name+gnn} --data_name {data} --gnn_function_name {gnn}'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        logger.info(f"Training completed successfully for {model_name} on {data} using {gnn}")
    else:
        logger.error(f"Training failed for {model_name} on {data} using {gnn}\n{stderr.decode()}")
    logger.info(stdout.decode())

def train_pair(data, gnn):
    # 启动两个进程分别训练两个模型
    processes = []
    processes.append(multiprocessing.Process(target=train_model, args=('trains.py', 'gnet', data, gnn)))
    processes.append(multiprocessing.Process(target=train_model, args=('train_tgnet.py', 'tgnet_v1', data, gnn)))
    
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()