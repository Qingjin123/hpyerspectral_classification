import numpy as np
import scipy.io as sio
import os

def loadData(data_info_yaml: dict, dataset_name: str):
    data_info = data_info_yaml.get(dataset_name)
    assert data_info is not None, 'The data path is not found.'

    data_folder = data_info[0]
    data_path = os.path.join(data_folder, f"{dataset_name}_corrected.mat")
    label_path = os.path.join(data_folder, f"{dataset_name}_gt.mat")

    data_key = dataset_name.lower() if dataset_name == "PaviaU" else f"{dataset_name}_corrected".lower()
    if data_key == 'paviau':
        data_key = 'paviaU'
    label_key = f"{dataset_name}_gt".lower()
    if label_key == 'paviau_gt':
        label_key = 'paviaU_gt'

    data = sio.loadmat(data_path)[data_key]
    label = sio.loadmat(label_path)[label_key]

    data = data.astype(np.float32)
    label = label.astype(np.int64)

    return data, label


