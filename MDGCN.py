import config.yamls as yamls
from process import data_process, show_process
from data import load_data, show_data
import torch
from model.mdgcnnet import MDGCNNet
from util import utils
import tqdm
import torch.utils.tensorboard as tb
import matplotlib.pyplot as plt
import numpy as np

def train(data_name:str, model_name: str, lr:float, epochs:int):
    # seed
    seed_ = utils.setup_seed(seed=None)
    tb_dir, model_dir, img_dir, png_path = utils.mkdir(data_name, model_name)
    
    # tensorboard
    writer = tb.SummaryWriter(tb_dir)
    writer.add_text('data name:', data_name)
    writer.add_text('lr:', str(lr))
    writer.add_text('seed:', str(seed_))
    
    # data 
    loader = load_data.data_load(yamls.read_yaml(), data_name)
    label = loader.label
    ndata = loader.ndata
    height, width, channels = loader.data_shape
    print(loader.data_shape)
    
    # plot data 
    show_data.ShowData(ndata, label, data_name, if_pca=False, if_tsne=False, save_png_path=png_path)
    
    # data_process
    count, class_num = data_process.count_label(label)
    train_mask, test_mask = data_process.sample_mask(label, count, 0.15)
    show_process.show_mask(train_mask, label, data_name, 'train', png_path)
    show_process.show_mask(test_mask, label, data_name, 'test',  png_path)
    
    # slic 
    seg_index, block_num = data_process.slic_data(ndata)
    show_process.plot_slic(seg_index, data_name, png_path)
    
    # neighbor
    neighbor = data_process.Neighbor(seg_index, block_num)
    neis = []
    neis.append(neighbor.find_first_order_neighbors) 
    max_nei = neighbor.find_max_order_neighbors(neis[0])
    print('max neighbor number:', max_nei)
    assert max_nei >=2, 'neighbor number must be greater than 2'
    for i in range(1, max_nei+1):
        neis.append(neighbor.find_nth_order_neighbors(neis[0], i+1))
    
    
    # means
    regional_means = data_process.calcul_means(ndata, seg_index, block_num) 
    
    # adj
    A = data_process.calcul_A(regional_means)
    
    # mutil scale support
    supports = []
    for nei in neis:
        support = data_process.calcul_support(A, nei)
        supports.append(support)
        
    # hidden layers
    hidden_channels_list = [20 for i in range(max_nei)]
    
    # parameters
    args = utils.parser()
    hyperparameters = utils.hyperparameter(args)
    hyperparameters['lr'] = lr
    hyperparameters['epoch'] = epochs
    
    # device
    device = utils.device('mps')
    
    # model
    model = MDGCNNet(in_channels=regional_means.shape[1], hidden_channels_list=hidden_channels_list, class_num=class_num, support=supports).to(device)
    for param in model.parameters():
        assert param.requires_grad, "Parameters do not require gradients"

    # optimizer
    optimizer, scheduler = utils.optimizer('adam', model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])

    # loss
    loss_function = utils.loss('cross_entropy')
    # performance = utils.performance

    # record
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    # to tensor
    regional_labels = data_process.one_hot_regional_labels(label, seg_index, class_num)
    supports = np.array(supports)
    neis = np.array(neis)
    regional_means = torch.tensor(regional_means).to(device)
    supports = torch.tensor(supports).to(device)
    neis = torch.tensor(neis).to(device)
    ndata = torch.tensor(ndata).to(device)
    label = torch.tensor(label).to(device)
    seg_index = torch.tensor(seg_index).to(device)
    train_mask = torch.tensor(train_mask).to(device)
    test_mask = torch.tensor(test_mask).to(device)
    regional_labels = torch.tensor(regional_labels).float().to(device)
    
    for epoch in tqdm.tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        features = model(regional_means, supports, neis)
        pixels_pred = utils.pixel_level_prediction(features, seg_index).to(device)
        
        loss = loss_function(features, regional_labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss.append(loss.item())
        masked_pixels_pred = pixels_pred[train_mask].float()
        masked_labels = label[train_mask].float()
        train_acc.append(utils.acc(masked_pixels_pred, masked_labels))
        writer.add_scalar('train_loss', train_loss[-1], epoch)
        writer.add_scalar('train_acc', train_acc[-1], epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            
        if (epoch % 3 == 0) and epoch >= 1:
            model.eval()
            with torch.no_grad():
                features = model(regional_means, supports, neis)
                # pred = utils.prediction(features, seg_index)
                pixels_pred = utils.pixel_level_prediction(features, seg_index).to(device)
                
                loss = loss_function(features, regional_labels)
                val_loss.append(loss.item())
                
                masked_pixels_pred = pixels_pred[test_mask].float()
                masked_labels = label[test_mask].float()
                val_acc.append(utils.acc(masked_pixels_pred, masked_labels))
                writer.add_scalar('train_loss', val_loss[-1], epoch)
                writer.add_scalar('train_acc', val_acc[-1], epoch)
                
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(pixels_pred.reshape(height, width).cpu().numpy())
                axs[1].imshow(label.reshape(height, width).cpu().numpy())
                plt.savefig(img_dir + 'epoch_' + str(epoch) + '.png')
    

data_name = 'Indian_pines'
train(data_name, model_name='MDGCN', lr=1e-3, epochs=1000)