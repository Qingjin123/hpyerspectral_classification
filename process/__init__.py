"""
This package provides utility classes for data processing, dimensionality reduction, graph calculation, and neighbor computation.

It includes the following classes:
- DataProcessor: Processes and samples data.
- DimensionalityReducer: Performs dimensionality reduction using PCA and t-SNE.
- GraphCalculator: Calculates pixel block means, adjacency matrix, and support matrix.
- Neighbor: Computes neighbor relationships of pixel blocks.

For more details, please refer to the documentation of each class.
"""

__version__ = '1.0.0'
__author__ = 'Qing Jin'
__email__ = 'qingjin159@outlook.com'
__all__ = ['DataProcessor', 'DimensionalityReducer', 'GraphCalculator', 'Neighbor']

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from .data_processor import DataProcessor
from .dimensionality_reducer import DimensionalityReducer
from .graph_calculator import GraphCalculator
from .neighbor import Neighbor

logger.info(f"Package initialized. Version: {__version__}")