import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class DimensionalityReducer:
    def __init__(self, data: np.ndarray):
        """
        降维器,用于对数据进行降维操作。

        Args:
            data (np.ndarray): 原始数据数组。
        """
        self.data = data

    def pca_reduction(self, n_components: int = 3) -> np.ndarray:
        """
        使用PCA算法对数据进行降维。

        Args:
            n_components (int): 降维后的目标维度,默认为3。

        Returns:
            np.ndarray: 降维后的数据数组。
        """
        h, w, c = self.data.shape
        data_reshaped = self.data.reshape(-1, c)

        pca = PCA(n_components=n_components)
        data_reduced = pca.fit_transform(data_reshaped)

        return data_reduced.reshape(h, w, n_components)

    def tsne_reduction(self, n_components: int = 2, perplexity: float = 30.0, early_exaggeration: float = 12.0,
                       learning_rate: float = 200.0, n_iter: int = 1000, n_iter_without_progress: int = 300,
                       min_grad_norm: float = 1e-7, init: str = 'random', verbose: int = 0,
                       random_state: int = None, method: str = 'barnes_hut', angle: float = 0.5) -> np.ndarray:
        """
        使用t-SNE算法对数据进行降维。

        Args:
            n_components (int): 降维后的目标维度,默认为2。
            perplexity (float): t-SNE的困惑度,关系到高斯分布的方差,默认为30.0。
            early_exaggeration (float): 早期夸大因子,默认为12.0。
            learning_rate (float): 学习率,默认为200.0。
            n_iter (int): 最大迭代次数,默认为1000。
            n_iter_without_progress (int): 没有进展时的最大迭代次数,默认为300。
            min_grad_norm (float): 最小梯度范数,默认为1e-7。
            init (str): 初始化方法,可以是 'random' 或者 'pca',默认为 'random'。
            verbose (int): 冗长模式,0 表示不输出日志,1 表示输出进度日志,默认为0。
            random_state (int): 随机种子,用于可重复性。默认为None。
            method (str): 梯度计算方法,可以是 'barnes_hut' 或者 'exact',默认为 'barnes_hut'。
            angle (float): 角度阈值,用于 'barnes_hut' 方法,默认为0.5。

        Returns:
            np.ndarray: 降维后的数据数组。
        """
        data_reshaped = self.data.reshape(-1, self.data.shape[-1])

        tsne = TSNE(n_components=n_components, perplexity=perplexity,
                    early_exaggeration=early_exaggeration, learning_rate=learning_rate,
                    n_iter=n_iter, n_iter_without_progress=n_iter_without_progress,
                    min_grad_norm=min_grad_norm, init=init, verbose=verbose,
                    random_state=random_state, method=method, angle=angle)

        data_reduced = tsne.fit_transform(data_reshaped)

        return data_reduced