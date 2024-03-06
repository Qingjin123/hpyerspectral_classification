import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class DataVisualizer:
    """
    数据可视化类,用于展示数据集的 ground truth、PCA、t-SNE 结果以及 SLIC 分割结果。
    """
    def __init__(self, data: np.ndarray, label: np.ndarray, data_name: str, save_path: str) -> None:
        """
        初始化 DataVisualizer 类。

        Args:
            data (np.ndarray): 输入数据。
            label (np.ndarray): 数据的标签。
            data_name (str): 数据集或图像的标识符。
            save_path (str): 图像保存的目录路径。
        """
        h, w, c = data.shape
        self.h = h
        self.w = w
        self.data = data.reshape(h*w, c)
        self.label = label
        self.data_name = data_name
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)  # 确保保存目录存在

    def show_ground_truth(self) -> None:
        """
        显示并保存数据集的 ground truth。
        """
        class_num = len(np.unique(self.label)) - 1
        plt.figure()
        plt.imshow(self.label)
        plt.title(f'Ground Truth of {self.data_name} ({class_num} classes)')
        plt.savefig(os.path.join(self.save_path, f'{self.data_name}_ground_truth.png'))
        plt.close()

    def pca_data(self, n_components: int=3) -> np.ndarray:
        """
        对数据进行 PCA 降维。

        Args:
            n_components (int): PCA 降维后的维度数。默认为 3。

        Returns:
            np.ndarray: PCA 降维后的数据。
        """
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(self.data)
        data_pca = data_pca.reshape(self.h, self.w, n_components)
        return data_pca

    def plot_pca(self) -> None:
        """
        显示并保存数据集的 PCA 结果。
        """
        data_pca = self.pca_data()
        data_pca = (data_pca - data_pca.min()) / (data_pca.max() - data_pca.min())  # 归一化到 0-1
        plt.figure()
        plt.imshow(data_pca)
        plt.title(f'PCA of {self.data_name}')
        plt.savefig(os.path.join(self.save_path, f'{self.data_name}_pca.png'))
        plt.close()

    def tsne_data(self, n_components: int=2) -> np.ndarray:
        """
        对数据进行 t-SNE 降维。

        Args:
            n_components (int): t-SNE 降维后的维度数。默认为 2。

        Returns:
            np.ndarray: t-SNE 降维后的数据。
        """
        tsne = TSNE(n_components=n_components)
        data_tsne = tsne.fit_transform(self.data)
        return data_tsne

    def plot_tsne(self) -> None:
        """
        显示并保存数据集的 t-SNE 结果。
        """
        data_tsne = self.tsne_data()
        plt.figure()
        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=self.label.flatten(), s=1)
        plt.title(f't-SNE of {self.data_name}')
        plt.colorbar()
        plt.savefig(os.path.join(self.save_path, f'{self.data_name}_tsne.png'))
        plt.close()

    def show_mask(self, mask: np.ndarray, plot_name: str) -> None:
        """
        显示并保存应用掩码后的标签图像。

        Args:
            mask (np.ndarray): 二进制掩码,指示感兴趣的区域。
            plot_name (str): 图像的特定名称。
        """
        plt.figure()
        plt.imshow(mask * self.label)
        plt.title(self.data_name)
        plt.savefig(os.path.join(self.save_path, f'{self.data_name}_{plot_name}_mask.png'))
        plt.close()

    def plot_slic(self, seg_index: np.ndarray) -> None:
        """
        显示并保存使用 SLIC 算法分割的图像。

        Args:
            seg_index (np.ndarray): 分割区域的索引。
        """
        plt.figure()
        plt.imshow(seg_index)
        plt.title(f'SLIC of {self.data_name}')
        plt.savefig(os.path.join(self.save_path, f'{self.data_name}_slic.png'))
        plt.close()