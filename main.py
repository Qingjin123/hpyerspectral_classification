from config.yamls import read_yaml
from data.load_data import DataLoader
from process import DataProcessor, DimensionalityReducer, GraphCalculator, Neighbor
from model.mdgcnnet import MDGCNNet
from model.segnet import SegNet
from util import parse_args, setup_seed
from util import device, optimizer, loss, mkdir
from util import performance, acc
from util import pixel_level_prediction
from util import DataVisualizer
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from trainer.dmsgcn_trainer import DMSGCNTrainer

args = parse_args()

# data
dataloader = DataLoader(read_yaml(), args.data_name)
data = dataloader.normalized_data
label = dataloader.labels
[height, width, channels] = dataloader.shape()

# data processor
dataprocessor = DataProcessor(data, label)
counts, class_num = dataprocessor.count_label()
train_mask, test_mask = dataprocessor.sample_mask()
seg_index, block_num = dataprocessor.slic_segmentation()

# show
dataview = DataVisualizer(data, label, args.data_name, './save/')
dataview.show_ground_truth()
dataview.plot_pca()
dataview.plot_tsne()
dataview.show_mask(train_mask, 'train_mask')
dataview.show_mask(test_mask, 'test_mask')
dataview.plot_slic(seg_index)

if args.model_name == 'segnet':
    Model = DMSGCNTrainer

