"""
This package provides utility functions for hyperspectral image classification using Graph Convolutional Networks (GCN).

It includes the following modules:
- config: Configuration and hyperparameter settings.
- misc: Miscellaneous utility functions for device, optimizer, loss, and directory creation.
- metrics: Performance metrics calculation.
- pixel_prediction: Pixel-level prediction generation.
- show: Visualization and display functions.

For more details, please refer to the documentation of each module.
"""

__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your_email@example.com'
__all__ = ['config', 'misc', 'metrics', 'pixel_prediction', 'show']

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from .config import parse_args, setup_seed
from .misc import device, optimizer, loss, mkdir
from .metrics import performance, acc
from .pixel_prediction import pixel_level_prediction
from .show import DataVisualizer

logger.info(f"Package initialized. Version: {__version__}")