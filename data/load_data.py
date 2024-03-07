import os
import scipy.io as sio
import numpy as np

def load_data(data_config: dict, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    根据 YAML 配置文件中的信息加载数据和标签。

    Args:
        data_config (dict): 包含数据路径的字典。
        dataset_name (str): 数据集的名称。

    Returns:
        tuple[np.ndarray, np.ndarray]: 加载的数据和对应的标签。

    Raises:
        ValueError: 如果在 data_config 中找不到 dataset_name 对应的信息。
        FileNotFoundError: 如果 data_config 中指定的数据文件夹不存在。
    """
    dataset_info = data_config.get(dataset_name)
    if dataset_info is None:
        raise ValueError(f"在配置文件中找不到数据集 {dataset_name} 的信息。")

    data_folder = dataset_info[0]
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"数据集 {dataset_name} 的文件夹不存在!")

    # 构建文件路径
    data_file = os.path.join(data_folder, f"{dataset_name}_corrected.mat")
    label_file = os.path.join(data_folder, f"{dataset_name}_gt.mat")

    # 加载数据和标签
    data_key = dataset_name.lower() if dataset_name == "PaviaU" else f"{dataset_name}_corrected".lower()
    if data_key == 'paviau':
        data_key = 'paviaU'
    label_key = f"{dataset_name}_gt".lower()
    if label_key == 'paviau_gt':
        label_key = 'paviaU_gt'

    try:
        data = sio.loadmat(data_file)[data_key]
        labels = sio.loadmat(label_file)[label_key]
    except KeyError as e:
        raise KeyError(f"加载 MAT 文件时发生键错误: {e}")

    # 转换数据类型
    data = data.astype(np.float32)
    labels = labels.astype(np.int64)

    return data, labels

class DataLoader:
    """
    用于加载和管理高光谱分类数据和标签的类。

    Attributes:
        data (np.ndarray): 加载的数据。
        labels (np.ndarray): 对应的标签。
    """

    def __init__(self, data_config: dict, dataset_name: str) -> None:
        """
        初始化 DataLoader 类的实例。

        Args:
            data_config (dict): 包含数据路径的字典。
            dataset_name (str): 数据集的名称。
        """
        try:
            self.data, self.labels = load_data(data_config, dataset_name)
        except Exception as e:
            print(f"加载数据时发生错误: {e}")
            self.data = None
            self.labels = None

    @property
    def shape(self) -> tuple[int, ...]:
        """
        返回数据的形状。

        Returns:
            tuple[int, ...]: 数据的形状。
        """
        return self.data.shape if self.data is not None else None

    @property
    def normalized_data(self) -> np.ndarray:
        """
        返回归一化后的数据。

        Returns:
            np.ndarray: 归一化后的数据。
        """
        return self._normalize(self.data) if self.data is not None else None

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        使用 z-score 归一化方法归一化数据。

        Args:
            data (np.ndarray): 要归一化的数据。

        Returns:
            np.ndarray: 归一化后的数据。
        """
        if data.ndim == 3:
            # 如果是三维数组,在特征维度上进行归一化
            mean = np.mean(data, axis=(0, 1), keepdims=True)
            std = np.std(data, axis=(0, 1), keepdims=True)
        else:
            mean = np.mean(data)
            std = np.std(data)
        return (data - mean) / std
