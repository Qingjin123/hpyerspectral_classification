import numpy as np

class Neighbor:
    def __init__(self, seg_index: np.ndarray, nei: int = 8):
        """
        邻居计算器,用于计算超像素块的邻居关系。

        Args:
            seg_index (np.ndarray): 超像素分割索引数组。
            nei (int): 邻居类型,可以是4或8,默认为8。
        """
        self.seg_index = seg_index
        self.block_num = np.max(seg_index) + 1
        self.directions = self._get_directions(nei)

    def _get_directions(self, nei: int) -> list[tuple[int, int]]:
        """
        根据邻居类型获取方向。

        Args:
            nei (int): 邻居类型,可以是4或8。

        Returns:
            list[tuple[int, int]]: 方向列表。
        """
        if nei == 8:
            return [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        elif nei == 4:
            return [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            raise ValueError("邻居类型必须是4或8。")

    def find_first_order_neighbors(self) -> np.ndarray:
        """
        计算一阶邻接矩阵。

        Returns:
            np.ndarray: 一阶邻接矩阵。
        """
        height, width = self.seg_index.shape
        neighbor_matrix = np.zeros((self.block_num, self.block_num), dtype=np.int8)

        for i in range(height):
            for j in range(width):
                current_block = self.seg_index[i, j]
                for di, dj in self.directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_block = self.seg_index[ni, nj]
                        if current_block != neighbor_block:
                            neighbor_matrix[current_block, neighbor_block] = 1
                            neighbor_matrix[neighbor_block, current_block] = 1

        return neighbor_matrix

    def find_nth_order_neighbors(self, adjacency_matrix: np.ndarray, n: int) -> np.ndarray:
        """
        计算第n阶邻接矩阵。

        Args:
            adjacency_matrix (np.ndarray): 一阶邻接矩阵。
            n (int): 邻居阶数。

        Returns:
            np.ndarray: 第n阶邻接矩阵。
        """
        self._assert_adjacency_matrix(adjacency_matrix)
        nth_order_matrix = np.linalg.matrix_power(adjacency_matrix, n)
        nth_order_matrix[nth_order_matrix > 0] = 1
        np.fill_diagonal(nth_order_matrix, 0)

        if n > 1:
            nth_order_matrix = nth_order_matrix | adjacency_matrix

        return nth_order_matrix

    def find_max_order_neighbors(self, adjacency_matrix: np.ndarray) -> int:
        """
        找到最大的邻居阶数。

        Args:
            adjacency_matrix (np.ndarray): 一阶邻接矩阵。

        Returns:
            int: 最大的邻居阶数。
        """
        self._assert_adjacency_matrix(adjacency_matrix)
        n = adjacency_matrix.shape[0]
        cumulative_matrix = adjacency_matrix.copy()
        power_matrix = adjacency_matrix.copy()

        for max_order in range(1, n):
            power_matrix = np.dot(power_matrix, adjacency_matrix)
            power_matrix[power_matrix > 0] = 1
            new_connections = np.bitwise_and(power_matrix, np.bitwise_not(cumulative_matrix))
            if not np.any(new_connections):
                break
            cumulative_matrix = np.bitwise_or(cumulative_matrix, power_matrix)

        return max_order

    def _assert_adjacency_matrix(self, adjacency_matrix: np.ndarray):
        """
        断言邻接矩阵的有效性。

        Args:
            adjacency_matrix (np.ndarray): 邻接矩阵。
        """
        assert adjacency_matrix.ndim == 2, "邻接矩阵必须是二维的。"
        assert np.all((adjacency_matrix == 0) | (adjacency_matrix == 1)), "邻接矩阵必须只包含0和1。"
        assert np.all(np.diag(adjacency_matrix) == 0), "邻接矩阵的对角线元素必须为0。"