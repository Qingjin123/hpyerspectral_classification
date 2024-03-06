import numpy as np
from scipy.stats import mode

class GraphCalculator:
    def __init__(self, data: np.ndarray, seg_index: np.ndarray):
        """
        图计算器,用于计算超像素块的均值、邻接矩阵和支持度矩阵。

        Args:
            data (np.ndarray): 原始数据数组。
            seg_index (np.ndarray): 超像素分割索引数组。
        """
        self.data = data
        self.seg_index = seg_index
        self.block_num = np.max(seg_index) + 1

    def calcul_means(self) -> np.ndarray:
        """
        计算每个超像素块的均值。

        Returns:
            np.ndarray: 每个超像素块的均值数组。
        """
        h, w, c = self.data.shape
        data_reshaped = self.data.reshape(-1, c)
        seg_index_reshaped = self.seg_index.reshape(-1)
        regional_means = np.zeros((self.block_num, c), dtype=np.float32)

        for i in range(self.block_num):
            block_indices = np.where(seg_index_reshaped == i)
            block_data = data_reshaped[block_indices]
            regional_means[i] = np.mean(block_data, axis=0)

        return regional_means

    def calcul_adjacency_matrix(self, regional_means: np.ndarray, gamma: float = 0.2) -> np.ndarray:
        """
        利用区域均值和指数衰减计算全连接的邻接矩阵。

        Args:
            regional_means (np.ndarray): 每个超像素块的均值数组。
            gamma (float): 比例系数,等同于1 / (2 * sigma^2),默认为0.2。

        Returns:
            np.ndarray: 全连接的邻接矩阵。
        """
        diff_squared = np.sum((regional_means[:, np.newaxis] - regional_means) ** 2, axis=2)
        adjacency_matrix = np.exp(-gamma * diff_squared)
        np.fill_diagonal(adjacency_matrix, 0)

        return adjacency_matrix

    def calcul_support(self, adjacency_matrix: np.ndarray, neighbor_matrix: np.ndarray,
                       self_connection_weight: float = 0) -> np.ndarray:
        """
        利用邻接矩阵和邻居矩阵计算支持度矩阵。

        Args:
            adjacency_matrix (np.ndarray): 全连接的邻接矩阵。
            neighbor_matrix (np.ndarray): 邻居矩阵,它是一个0-1矩阵,指示图中节点间是否为直接邻居。
            self_connection_weight (float): 自环的权重,用于增强节点自身的影响力,默认为0。

        Returns:
            np.ndarray: 支持度矩阵。
        """
        assert adjacency_matrix.shape == neighbor_matrix.shape, "邻接矩阵和邻居矩阵的形状必须相同。"

        filtered_adjacency_matrix = adjacency_matrix * neighbor_matrix
        degree = np.sum(filtered_adjacency_matrix, axis=1) + self_connection_weight
        degree_inv_sqrt = np.reciprocal(np.sqrt(degree))
        degree_matrix = np.diag(degree_inv_sqrt)

        support_matrix = degree_matrix @ (filtered_adjacency_matrix + self_connection_weight * np.eye(adjacency_matrix.shape[0])) @ degree_matrix
        support_matrix = support_matrix.astype(np.float32)

        return support_matrix

    def calculate_regional_labels(self, label: np.ndarray, class_num: int) -> np.ndarray:
        """
        计算超像素块的标签,并转化为 one-hot 编码。

        Args:
            label (np.ndarray): 图像的标签数组。
            class_num (int): 类别数。

        Returns:
            np.ndarray: one-hot 编码的超像素块标签。
        """
        num_segments = self.block_num
        regional_labels = np.zeros(num_segments, dtype=int)

        for i in range(num_segments):
            pixels_in_block = label[self.seg_index == i]
            if pixels_in_block.size > 0:
                regional_labels[i] = mode(pixels_in_block)[0]

        one_hot_labels = np.eye(class_num)[regional_labels]

        return one_hot_labels