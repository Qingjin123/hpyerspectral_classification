import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime

def device(device_name):
    return torch.device(device_name)

def optimizer(optimizer_name, parameters, lr, weight_decay):
    optimizer_map = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD
    }
    optimizer_class = optimizer_map.get(optimizer_name.lower(), optim.SGD)
    optimizer = optimizer_class(parameters, lr=lr, weight_decay=weight_decay)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.6)

    return optimizer, scheduler

def loss(loss_name):
    loss_functions = {
        'crossentropyloss': torch.nn.CrossEntropyLoss(),
        'nllloss': torch.nn.NLLLoss()
    }
    return loss_functions.get(loss_name.lower(), torch.nn.CrossEntropyLoss())

def mkdir(data_name, model_name):
    # 获取当前时间的时间戳
    current_timestamp = datetime.now().timestamp()
    # 将时间戳转换为字符串并去掉小数部分
    time_name = str(int(current_timestamp))
    
    base_dir = os.path.join('save', model_name, data_name, time_name)
    
    tb_dir = os.path.join(base_dir, 'log')
    model_dir = os.path.join(base_dir, 'model')
    img_dir = os.path.join(base_dir, 'train')
    png_path = os.path.join(base_dir, 'png')
    
    for directory in [tb_dir, model_dir, img_dir, png_path]:
        os.makedirs(directory, exist_ok=True)
    
    return tb_dir, model_dir, img_dir, png_path
