from sklearn.preprocessing import normalize
import numpy as np

def normData(data: np.ndarray):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean)/std

def countLabel(label: np.ndarray):
    unique_labels, counts = np.unique(label, return_counts=True)
    class_num = len(unique_labels)
    # 计算了无标记的样本
    return counts[1:], class_num-1

def sampleMask(label: np.ndarray, count: np.ndarray, ratio: float = 0.15):

    h, w = label.shape
    train_mask = np.zeros((h, w), dtype=bool)
    test_mask = np.ones((h, w), dtype=bool)
    train_num = []
    test_num = []
    for i, cut in enumerate(count):
        indexs = np.argwhere(label == i)
        np.random.shuffle(indexs)
        # train_num = int(cut * ratio)
        train_n = 30 if cut > 30 else 15
        train_indexs = indexs[:train_n]
        train_mask[train_indexs[:, 0], train_indexs[:, 1]] = 1
        test_mask[train_indexs[:, 0], train_indexs[:, 1]] = 0

        train_num.append(train_n)
        test_num.append(cut - train_n)

    print('train_num:', train_num)
    print('test_num:', test_num)

    return train_mask, test_mask
