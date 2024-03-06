import numpy as np
from skimage.segmentation import slic

class DataProcessor:
    def __init__(self, data: np.ndarray, label: np.ndarray):
        """
        数据处理器,用于处理和采样数据。

        Args:
            data (np.ndarray): 原始数据数组。
            label (np.ndarray): 对应的标签数组。
        """
        self.data = data
        self.label = label

    def count_label(self) -> tuple[np.ndarray, int]:
        """
        统计标签中每个类别的数量。

        Returns:
            tuple[np.ndarray, int]: 包含每个类别数量的数组和总类别数。
        """
        unique_labels, counts = np.unique(self.label, return_counts=True)
        class_num = len(unique_labels)
        return counts, class_num - 1

    def sample_mask(self, ratio: float = 0.15, seed: int = None) -> tuple[np.ndarray, np.ndarray]:
        """
        根据比例采样出训练集和测试集的掩码。

        Args:
            ratio (float): 训练集的比例,默认为0.15。
            seed (int): 随机种子,用于可重复性。默认为None。

        Returns:
            tuple[np.ndarray, np.ndarray]: 包含训练集掩码和测试集掩码的元组。
        """
        h, w = self.label.shape
        train_mask = np.zeros((h, w), dtype=bool)
        test_mask = np.zeros((h, w), dtype=bool)

        counts, _ = self.count_label()

        if seed is not None:
            np.random.seed(seed)

        for i, cnt in enumerate(counts):
            indices = np.stack(np.where(self.label == i), axis=-1)
            np.random.shuffle(indices)
            train_n = min(int(cnt * ratio), 30 if cnt > 30 else 15)

            train_indices = indices[:train_n]
            test_indices = indices[train_n:]

            train_mask[train_indices[:, 0], train_indices[:, 1]] = 1
            test_mask[test_indices[:, 0], test_indices[:, 1]] = 1

        return train_mask, test_mask

    def sample_mask_with_validation(self, ratio: float = 0.1, min_samples_per_class: int = 15, seed: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        采样训练集、验证集和测试集的掩码。

        Args:
            ratio (float): 训练集中用作验证集的比例,默认为0.1。
            min_samples_per_class (int): 每个类别的最小样本数,默认为15。
            seed (int): 随机种子,用于可重复性。默认为None。

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: 包含训练集掩码、验证集掩码和测试集掩码的元组。
        """
        h, w = self.label.shape
        train_mask = np.zeros((h, w), dtype=bool)
        val_mask = np.zeros((h, w), dtype=bool)
        test_mask = np.zeros((h, w), dtype=bool)

        unique_labels = np.unique(self.label)

        if seed is not None:
            np.random.seed(seed)

        for cls in unique_labels:
            cls_indices = np.stack(np.where(self.label == cls), axis=-1)
            np.random.shuffle(cls_indices)
            cls_sample_count = max(min_samples_per_class, 30) if cls_indices.shape[0] >= 30 else min_samples_per_class

            train_indices = cls_indices[:cls_sample_count]
            test_indices = cls_indices[cls_sample_count:]

            val_sample_count = int(len(train_indices) * ratio)
            val_indices = train_indices[:val_sample_count]
            train_indices = train_indices[val_sample_count:]

            train_mask[train_indices[:, 0], train_indices[:, 1]] = True
            val_mask[val_indices[:, 0], val_indices[:, 1]] = True
            test_mask[test_indices[:, 0], test_indices[:, 1]] = True

        return train_mask, val_mask, test_mask

    def slic_segmentation(self, n_segments: int = 50, compactness: float = 0.2, sigma: float = 0.5) -> tuple[np.ndarray, int]:
        """
        使用SLIC算法分割数据。

        Args:
            n_segments (int): 分割的超像素块数量,默认为50。
            compactness (float): 超像素块的紧凑度,默认为0.2。
            sigma (float): 预处理的平滑程度,默认为0.5。

        Returns:
            tuple[np.ndarray, int]: 包含分割后的超像素索引和总超像素块数的元组。
        """
        seg_index = slic(self.data, n_segments=n_segments, compactness=compactness, sigma=sigma)
        block_num = np.max(seg_index) + 1
        return seg_index, block_num