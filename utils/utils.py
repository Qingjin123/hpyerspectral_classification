import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler  
import random
from datetime import datetime
import os
import sklearn.metrics as sm

def parser():
    parser = argparse.ArgumentParser(description='Arguments for GCN based hyperspectral image classification.')

    parser.add_argument('--data_name', type=str, default='Indian_pines', help='Name of the dataset.')
    parser.add_argument('--model_name', type=str, default='DMSGCN', help='Name of the model.')
    parser.add_argument('--superpixel_name', type=str, default='SLIC', help='Name of superpixel function')

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--epoch', type=int, default=500, help='Number of training epochs.')

    return parser.parse_args()

def performance(predict_labels, gt_labels, class_num):
    matrix = np.zeros((class_num, class_num))
    predict_labels = torch.max(predict_labels,dim=1)[1]
    
    for j in range(len(predict_labels)):
        o = predict_labels[j]
        q = gt_labels[j]
        if q == 0:
            continue
        matrix[o-1, q-1] += 1

    OA = np.sum(np.trace(matrix)) / np.sum(matrix)

    ac_list = np.zeros((class_num))
    # alpha = 1/class_num
    alpha = 1e-10
    for k in range(len(matrix)):
        
        n_samples = sum(matrix[:, k])
        ac_k = (matrix[k, k] + alpha) / (n_samples + class_num * alpha)
        # ac_k = matrix[k, k] / sum(matrix[:, k])
        ac_list[k] = round(ac_k,4)
    
    AA = np.mean(ac_list)

    
    mm = 0
    for l in range(matrix.shape[0]):
        mm += np.sum(matrix[l]) * np.sum(matrix[:, l])
    pe = mm / (np.sum(matrix) * np.sum(matrix))
    pa = np.trace(matrix) / np.sum(matrix)
    kappa = (pa - pe) / (1 - pe)
    
    
    return OA, AA, kappa, ac_list

def calculateTopkAccuracy(y_pred, y, k = 5):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.reshape(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
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
    print(f'Random seed: {seed}')
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁用优化，以增加可复现性
    return seed

def getDevice(device_name: str):
    if device_name == 'cuda':
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    return torch.device(device_name)

# optimizer
def getOptimizer(optimizer_name: str, parameters: list, lr: float, weight_decay: float):
    # Define the optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        # Default to SGD if optimizer_name is not recognized
        optimizer = optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)

    # Define the learning rate scheduler
    # StepLR reduces the learning rate by a factor of gamma every step_size epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.6)

    # Uncomment the following line if you prefer to use ReduceLROnPlateau
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)

    return optimizer, scheduler

def getLoss(loss_name):
    loss_functions = {
        'crossentropyloss': torch.nn.CrossEntropyLoss(),
        'nllloss': torch.nn.NLLLoss()
        # add other loss functions here if needed
    }
    
    # Convert loss_name to lowercase and return the corresponding loss function
    # Default to CrossEntropyLoss if loss_name is not recognized
    return loss_functions.get(loss_name.lower(), torch.nn.CrossEntropyLoss())

def getMetrics(predict_labels, gt_labels, class_num):
    predict_labels = torch.max(predict_labels, dim=1)[1]
    # predict_labels = predict_labels.cpu().numpy()
    # gt_labels = gt_labels.cpu().numpy()
    # # confusion_matrix = sm.confusion_matrix(gt_labels, predict_labels)
    # OA = sm.accuracy_score(gt_labels, predict_labels)
    # AA = sm.f1_score(gt_labels, predict_labels, average='macro')
    # kappa = sm.cohen_kappa_score(gt_labels, predict_labels)
    # ac_list = sm.precision_recall_fscore_support(gt_labels, predict_labels, average=None)[0]
    # return OA, AA, kappa, ac_list.round(4)
    print(predict_labels)