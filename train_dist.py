import os
import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard as tb
from load_data import loadData
from logger import readYaml
from process_data import normData, countLabel, sampleMask, superpixels
from utils import parser, performance, mkdir, getDevice, getOptimizer, getLoss, setupSeed
from model import SegNet_v1
import matplotlib.pyplot as plt

def train(model_name: str, data_name: str, superpixels_name: str, gnn_function_name: str = 'gcn',
          lr: float = 5e-4, epochs: int = 500, weight_decay: float = 1e-4, batch_size: int = 1,
          ratio: float = 0.15, seeds: int = None, n_segments: int = 40, train_nums: int = 30,
          device_name: str = None, if_ratio: bool = False, yaml_path: str = 'dataset/data_info.yaml'):
    
    # Load and preprocess data
    data, label = loadData(readYaml(yaml_path), data_name)
    ndata = normData(data)
    seed = setupSeed(seeds)
    tb_dir, model_dir, img_dir, png_path = mkdir(data_name, model_name)
    writer = tb.SummaryWriter(tb_dir)
    count, class_num = countLabel(label)
    train_mask, test_mask = sampleMask(label, count, ratio, if_ratio, train_nums)
    seg_index, block_num = superpixels(ndata, superpixels_name, n_segments)
    device = getDevice(device_name)
    ndata, label, train_mask, test_mask, seg_index = [torch.from_numpy(x).to(device) for x in [ndata, label, train_mask, test_mask, seg_index]]
    adj_mask = torch.ones((block_num, block_num), dtype=np.float32).to(device)

    # Initialize model
    model = SegNet_v1(in_channels=ndata.shape[2], block_num=block_num, class_num=class_num+1,
                      batch_size=batch_size, gnn_name=gnn_function_name, adj_mask=adj_mask, device=device)
    model.to(device)
    optimizer, scheduler = getOptimizer('adam', model.parameters(), lr, weight_decay)
    loss_function = getLoss('cross_entropy')

    # Recording metrics
    records = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        final, finalsoft = model(ndata, seg_index)
        pred_gt = final * train_mask
        pred_gt = pred_gt[pred_gt != 0]  # Assuming non-zero entries are valid
        loss = loss_function(pred_gt, label * train_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            final, _ = model(ndata, seg_index)
            pred_gt = final * test_mask
            pred_gt = pred_gt[pred_gt != 0]
            test_loss = loss_function(pred_gt, label * test_mask)
        
        OA, AA, kappa, ac_list = performance(pred_gt.cpu(), (label * test_mask).cpu(), class_num)
        records.append([epoch, loss.item(), test_loss.item(), OA, AA, kappa] + ac_list)
        
        # Tensorboard logging
        writer.add_scalars('Losses', {'train': loss.item(), 'test': test_loss.item()}, epoch)
        writer.add_scalars('Accuracy', {'OA': OA, 'AA': AA, 'Kappa': kappa}, epoch)
    
    # Save model and records
    torch.save(model.state_dict(), os.path.join(model_dir, f'{model_name}_{data_name}.pth'))
    record_df = pd.DataFrame(records, columns=['Epoch', 'Train Loss', 'Test Loss', 'OA', 'AA', 'Kappa'] + [f'Class_{i}_Acc' for i in range(class_num)])
    record_df.to_csv(os.path.join(model_dir, f'{model_name}_{data_name}_records.csv'), index=False)

def run():
    args = parser()
    train(model_name=args.model_name, data_name=args.data_name, superpixels_name=args.superpixel_name,
          gnn_function_name=args.gnn_function_name, lr=args.lr, epochs=args.epoch,
          weight_decay=args.weight_decay, batch_size=args.batch_size, ratio=args.ratio,
          if_ratio=args.if_ratio, seeds=args.seeds, n_segments=args.n_segments, train_nums=args.train_nums,
          device_name=args.device_name)

if __name__ == '__main__':
    run()
