from config.yamls import read_yaml
from data.load_data import DataLoader
from process import DataProcessor, DimensionalityReducer, GraphCalculator, Neighbor
from util import parse_args
from util import DataVisualizer
import matplotlib.pyplot as plt
import numpy as np
from trainer.dmsgcn_trainer import DMSGCNTrainer

args = parse_args()

# data
dataloader = DataLoader(read_yaml(), args.data_name)
data = dataloader.normalized_data
label = dataloader.labels
height, width, channels = data.shape

# data processor
dataprocessor = DataProcessor(data, label)
counts, class_num = dataprocessor.count_label()
train_mask, test_mask = dataprocessor.sample_mask()
seg_index, block_num = dataprocessor.slic_segmentation()
adj_mask = np.zeros((block_num, block_num), dtype=np.float32)

print('block_num:', block_num)
print('all counts:', counts)
print('class num:', class_num)

# show
# dataview = DataVisualizer(data, label, args.data_name, './save/')
# dataview.show_ground_truth()
# dataview.plot_pca()
# dataview.plot_tsne()
# dataview.show_mask(train_mask, 'train_mask')
# dataview.show_mask(test_mask, 'test_mask')
# dataview.plot_slic(seg_index)

if args.model_name == 'DMSGCN':
    Model = DMSGCNTrainer(config=args)
    
Model.create_model(in_channels=channels, block_num=block_num, class_num=class_num, adj_mask=adj_mask)
Model.create_data(data, label, seg_index, train_mask, test_mask)
Model.create_loss()
Model.create_optimizer()
Model.run()

