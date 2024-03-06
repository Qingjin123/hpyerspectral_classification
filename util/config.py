import argparse
import random
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for GCN based hyperspectral image classification.')

    # Data-related arguments
    parser.add_argument('--data_name', type=str, default='Indian_pines', help='Name of the dataset.')
    parser.add_argument('--model_name', type=str, default='DMSGCN', help='Name of the model.')

    # Hyperparameter arguments
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--epoch', type=int, default=3000, help='Number of training epochs.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for regularization.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='mps', help='Device to use for training')
    
    # Output-related arguments
    # parser.add_argument('--log_dir', type=str, default='results', help='Directory to save the logs')

    args = parser.parse_args()

    return args

def setup_seed(seed: int = None):
    if seed is None:
        seed = random.randint(0, 2**32)
    print(f'Random seed: {seed}')
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁用优化，以增加可复现性
    return seed