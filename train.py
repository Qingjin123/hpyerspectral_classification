from load_data import loadData
from logger import readYaml
from process_data import normData, countLabel, sampleMask
from process_data import superpixels
from utils import parser, performance, mkdir, getDevice, getOptimizer, getLoss, setupSeed, calculateTopkAccuracy
from model import SegNet
import torch.utils.tensorboard as tb
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch


def getAll():
    args = parser()

    yaml_path = 'dataset/data_info.yaml'
    data_name = args.data_name
    model_name = args.model_name
    lr = args.lr
    epoch = args.epoch
    batch_size = args.batch_size



    data, label = loadData(readYaml(yaml_path), data_name)
    print('data shape:', data.shape)

    ndata = normData(data)
    counts, class_num = countLabel(label)
    # print('counts:', counts)
    print('class_num:', class_num)

    train_mask, test_mask = sampleMask(label, counts)
    

    


def superpixel(data: np.ndarray,function_name: str):
    # superpixels
    seg_index, block_num = superpixels(data, 'SLIC')
    print('block num:', block_num)
    return seg_index, block_num
def train():
    pass
