import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import random
from datetime import datetime
import os


def parser():
    parser = argparse.ArgumentParser(description='''Arguments for GCN based
                    hyperspectral image classification.''')

    parser.add_argument('--data_name',
                        type=str,
                        default='Indian_pines',
                        help='Name of the dataset.')
    parser.add_argument('--model_name',
                        type=str,
                        default='segnet_v1',
                        help='Name of the model.')
    parser.add_argument('--superpixel_name',
                        type=str,
                        default='SLIC',
                        help='Name of superpixel function')
    parser.add_argument('--gnn_function_name',
                        type=str,
                        default='fagcn',
                        help='Name of gnn function')
    parser.add_argument('--ratio',
                        type=float,
                        default=0.15,
                        help='Ratio of the training data.')
    parser.add_argument('--n_segments',
                        type=int,
                        default=40,
                        help='Number of superpixels.')
    parser.add_argument('--seeds',
                        type=int,
                        default=None,
                        help='Seeds for random.')
    parser.add_argument('--device_name',
                        type=str,
                        default=None,
                        help='Name of the device.')
    parser.add_argument('--yaml_path',
                        type=str,
                        default='dataset/data_info.yaml',
                        help='Path of the yaml file.')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-4,
                        help='Weight decay.')
    parser.add_argument('--if_ratio',
                        type=bool,
                        default=False,
                        help='If ratio.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Batch size.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0005,
                        help='Learning rate.')
    parser.add_argument('--epoch',
                        type=int,
                        default=500,
                        help='Number of training epochs.')
    parser.add_argument('--train_nums',
                        type=int,
                        default=5,
                        help='Number of training samples.')
    parser.add_argument('--scale_layer',
                        type=int,
                        default=4,
                        help='Number of scale layers.')

    return parser.parse_args()


def calculateTopkAccuracy(y_pred, y, k=5):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.reshape(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k


def mkdir(data_name: str, model_name: str):
    TIMESTAMP = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = 'save/' + model_name + '/'
    base_dir = os.path.join(path, data_name, TIMESTAMP)

    tb_dir = os.path.join(base_dir, 'log')
    model_dir = os.path.join(base_dir, 'model')
    img_dir = os.path.join(base_dir, 'train')
    png_path = os.path.join(base_dir, 'png')

    for directory in [tb_dir, model_dir, img_dir, png_path]:
        os.makedirs(directory, exist_ok=True)  # 更简洁的方式，使用exist_ok避免检查路径是否存在

    return tb_dir, model_dir, img_dir, png_path


def setupSeed(seed: int = None):
    if seed is None:
        seed = random.randint(0, 2**32)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁用优化，以增加可复现性
    return seed


def getDevice(device_name: str):
    if device_name is None:
        # 检测时cuda环境还是mps环境还是cpu环境
        if torch.backends.mps.is_available():
            device = torch.device("mps")  # 使用 Apple Silicon GPU
        elif torch.cuda.is_available():
            device = torch.device("cuda")  # 使用其他 GPU
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
        else:
            device = torch.device("cpu")  # 使用 CPU
        return device
    else:
        return torch.device(device_name)


# optimizer
def getOptimizer(optimizer_name: str, parameters: list, lr: float,
                 weight_decay: float):
    # Define the optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        # Default to SGD if optimizer_name is not recognized
        optimizer = optim.SGD(parameters,
                              lr=lr,
                              weight_decay=weight_decay,
                              momentum=0.9)

    # Define the learning rate scheduler

    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.6)

    # Uncomment the following line if you prefer to use ReduceLROnPlateau

    return optimizer, scheduler


def getLoss(loss_name):
    loss_functions = {
        'crossentropyloss': torch.nn.CrossEntropyLoss(reduction='none'),
        'nllloss': torch.nn.NLLLoss()
        # add other loss functions here if needed
    }

    # Convert loss_name to lowercase and return the corresponding loss function
    # Default to CrossEntropyLoss if loss_name is not recognized
    return loss_functions.get(loss_name.lower(), torch.nn.CrossEntropyLoss())


def calculate_confusion_matrix(pred_labels, true_labels, class_num):
    """计算混淆矩阵，忽略标签为0的数据"""
    matrix = np.zeros((class_num, class_num))  # class_num-1因为第一个类别被忽略
    for pred, true in zip(pred_labels, true_labels):
        if true > 0:  # 忽略true_labels为0的情况
            matrix[pred - 1, true - 1] += 1
    return matrix


def calculate_accuracy(matrix):
    """计算每类的准确率"""
    with np.errstate(divide='ignore', invalid='ignore'):
        accuracy = np.diag(matrix) / np.sum(matrix, axis=0)
        accuracy = np.nan_to_num(accuracy)  # 将NaN转换为0
    return np.round(accuracy, 4)


def calculate_overall_accuracy(matrix):
    """计算总体准确率(OA)"""
    total = np.sum(matrix)
    if total == 0:
        return 0
    return np.trace(matrix) / total


def calculate_average_accuracy(accuracies):
    """计算平均准确率(AA)"""
    if np.isnan(accuracies).all():  # 如果所有值都是NaN，则返回0
        return 0
    return np.nanmean(accuracies)


def calculate_kappa(matrix):
    """计算Kappa系数"""
    total = np.sum(matrix)
    if total == 0:
        return 0
    pe = np.sum(
        np.sum(matrix, axis=0) * np.sum(matrix, axis=1)) / (total * total)
    pa = np.trace(matrix) / total
    if 1 - pe == 0:
        return 1  # 如果分母为0，则假设完美一致，返回1
    return (pa - pe) / (1 - pe)


# def performance(predict_labels, gt_labels, class_num):
#     """评估模型性能
#     参数:
#     predict_labels -- 模型的预测标签 (torch tensor)
#     gt_labels -- 真实标签 (numpy array)
#     class_num -- 总类别数（包括一个不参与分类的类别）
#     返回:
#     OA -- 总体准确率
#     AA -- 平均准确率
#     kappa -- Kappa系数
#     accuracies -- 每类的准确率列表
#     """
#     pred_labels = torch.argmax(predict_labels, dim=1).numpy()
#     # 从tensor转换为numpy，并取最大值索引
#     matrix = calculate_confusion_matrix(pred_labels, gt_labels, class_num)
#     accuracies = calculate_accuracy(matrix)
#     OA = calculate_overall_accuracy(matrix)
#     AA = calculate_average_accuracy(accuracies)
#     kappa = calculate_kappa(matrix)

#     return OA, AA, kappa, accuracies


def performance(predict_labels, gt_labels, mask, class_num):
    """评估模型性能
    参数:
    predict_labels -- 模型的预测标签 (torch tensor),
        shape [1, class_num, height, width]
    gt_labels -- 真实标签 (numpy array), shape [height, width]
    mask -- 训练样本的mask (numpy array), shape [height, width]
    class_num -- 总类别数
    返回:
    OA -- 总体准确率
    AA -- 平均准确率
    kappa -- Kappa系数
    accuracies -- 每类的准确率列表
    """
    # 压缩批次维度，并取最大值索引，形成预测类别矩阵
    pred_labels = torch.argmax(predict_labels.squeeze(0), dim=0).numpy()

    # 应用mask
    valid_indices = (mask > 0)
    filtered_pred_labels = pred_labels[valid_indices]
    filtered_gt_labels = gt_labels[valid_indices]

    # 接下来是计算混淆矩阵和其他统计数据...
    matrix = calculate_confusion_matrix(filtered_pred_labels,
                                        filtered_gt_labels, class_num)
    accuracies = calculate_accuracy(matrix)
    OA = calculate_overall_accuracy(matrix)
    AA = calculate_average_accuracy(accuracies)
    kappa = calculate_kappa(matrix)

    return OA, AA, kappa, accuracies
