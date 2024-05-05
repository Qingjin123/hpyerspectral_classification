from load_data import loadData
from logger import readYaml, saveYaml
from process_data import normData, countLabel, sampleMask
from process_data import superpixels
from utils import parser, performance, mkdir, getDevice
from utils import getOptimizer, getLoss, setupSeed
from show import show_data, show_mask, plot_slic
from model import SegNet_v2, SegNet_v1, TGNet_v1
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from tqdm import tqdm

data_seed = {
    'Indian_pines': 2073516639,
    'PaviaU': 1568036650,
    'Salinas': 3503767999,
}


def train(model_name: str,
          data_name: str,
          superpixels_name: str,
          gnn_function_name: str = 'fagcn',
          lr: float = 5e-4,
          epochs: int = 500,
          weight_decay: float = 1e-4,
          batch_size: int = 1,
          ratio: float = 0.15,
          seeds: int = None,
          n_segments: int = 40,
          train_nums: int = 5,
          scale_layer: int = 1,
          device_name: str = None,
          if_ratio: bool = False,
          yaml_path: str = 'dataset/data_info.yaml'):
    # print
    print('\n')
    print('model_name:', model_name)
    print('data_name:', data_name)
    print('gnn_function_name:', gnn_function_name)

    # data
    data, label = loadData(readYaml(yaml_path), data_name)
    ndata = normData(data)

    seed = setupSeed(data_seed[data_name])

    # # seed
    # seed = setupSeed(seeds)

    # mkdir
    _, model_dir, img_dir, png_path = mkdir(data_name, model_name)

    # show data
    show_data(ndata,
              label,
              data_name,
              if_pca=False,
              if_tsne=False,
              save_png_path=png_path)
    count, class_num = countLabel(label)
    train_mask, test_mask = sampleMask(label, count, ratio, if_ratio,
                                       train_nums)
    show_mask(train_mask, label, data_name, 'train', png_path)
    show_mask(test_mask, label, data_name, 'test', png_path)
    seg_index, block_num = superpixels(ndata, superpixels_name, n_segments)
    plot_slic(seg_index, data_name, png_path)
    adj_mask = np.ones((block_num, block_num), dtype=np.float32)
    print('seed:', seed)
    print('block_num:', block_num)
    print('class_num:', class_num)

    device = getDevice(device_name)

    ndata = torch.from_numpy(ndata).to(device)
    label = torch.from_numpy(label).to(device)
    train_mask = torch.from_numpy(train_mask).to(device)
    test_mask = torch.from_numpy(test_mask).to(device)
    seg_index = torch.from_numpy(seg_index).to(device)
    adj_mask = torch.from_numpy(adj_mask).to(device)

    if model_name == 'segnet_v1':
        Model = SegNet_v1
    elif model_name == 'segnet_v2':
        Model = SegNet_v2
    elif model_name == 'tgnet_v1':
        Model = TGNet_v1
    else:
        print('model_name error')

    model = Model(in_channels=ndata.shape[2],
                  block_num=block_num,
                  class_num=class_num + 1,
                  batch_size=batch_size,
                  gnn_name=gnn_function_name,
                  adj_mask=adj_mask,
                  device=device,
                  scale_layer=scale_layer)

    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print("Total trainable parameters:", total_params)

    model.to(device)
    optimizer, scheduler = getOptimizer('adam', model.parameters(), lr,
                                        weight_decay)
    loss_function = getLoss('cross_entropy')

    # record
    train_loss = []
    test_acc = []
    record = []
    # [oa, aa, kappa]
    best_value = [0, 0, 0, 0, []]

    # def prediction(classes: torch.Tensor, gt: torch.Tensor,
    #                mask: torch.tensor):
    #     sum = mask.sum()

    #     train_gt = gt * mask
    #     train_gt = label * train_mask
    #     pre_gt = torch.cat((train_gt.unsqueeze(0).to(device), classes[0]),
    #                        dim=0)
    #     pre_gt = pre_gt.view(class_num + 2, -1).permute(1, 0)
    #     pre_gt_ = pre_gt[torch.argsort(pre_gt[:, 0], descending=True)]
    #     pre_gt_ = pre_gt_[:int(sum)]
    #     return pre_gt_

    parameters = {
        "model name": model_name,
        "data name": data_name,
        "gnn name": gnn_function_name,
        "superpixel name": superpixels_name,
        "class number": class_num,
        "seed": seed,
        "lr": lr,
        "block number": block_num,
        "Epoch": 0,
        "Best Epoch": 0,
        "Best OA": 0,
        "Best AA": 0,
        "Best Kappa": 0,
        "Time Spent": 0,
        "Ac List": None
    }

    label = label.unsqueeze(0)  # label 现在的形状是 [1, height, width]
    train_mask = train_mask.unsqueeze(0)
    start_time = time.time()
    print('Training...')
    for epoch in tqdm(range(epochs)):
        model.train()
        final, finalsoft = model(ndata, seg_index)

        # pred_gt = prediction(final, label, train_mask)

        # loss1 = loss_function(pred_gt[:, 1:], pred_gt[:, 0].long())
        loss1 = loss_function(finalsoft, label)
        # 将 train_mask 应用到计算得到的逐像素损失上
        loss = loss1 * train_mask

        # 计算最终的损失值，这里通过平均所有被 mask 包含的损失
        final_loss = loss.sum() / train_mask.sum()

        # 将计算得到的损失值添加到损失列表中
        train_loss.append(float(final_loss))

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            final, finalsoft = model(ndata, seg_index)

            OA, AA, kappa, ac_list = performance(finalsoft.cpu(),
                                                 label.squeeze(0).cpu(),
                                                 test_mask.cpu(), class_num)
            test_acc.append(ac_list)
            record.append([epoch, ac_list, OA, AA, kappa])

            if best_value[3] < kappa:
                best_value = [epoch, OA, AA, kappa, ac_list]
                torch.save(model.state_dict(),
                           model_dir + '/' + 'lr_' + str(lr) + '_model.pth')

                plt.figure()
                plt.imshow(
                    torch.max(torch.softmax(finalsoft[0].cpu(), dim=0),
                              dim=0)[1].cpu() *
                    (label.squeeze(0).cpu() > 0).float())
                plt.savefig(img_dir + '/' + 'DMSGer' + '_epoch_' + str(epoch) +
                            '_OA_' + str(round(OA, 2)) + '_AA_' +
                            str(round(AA, 2)) + '_KAPPA_' +
                            str(round(kappa, 2)) + '.png')
                plt.close()

        end_time = time.time()

        parameters['Epoch'] = int(epoch + 1)
        parameters['Best Epoch'] = int(best_value[0])
        parameters['Best OA'] = float(round(best_value[1], 5))
        parameters['Best AA'] = float(round(best_value[2], 5))
        parameters['Best Kappa'] = float(round(best_value[3], 5))
        parameters['Time Spent'] = float(round(end_time - start_time, 5))
        parameters['Ac List'] = best_value[4]

    for key, value in parameters.items():
        print(key, ':', value)

    parameters_yaml = 'parameters.yaml'
    saveYaml(parameters_yaml, parameters)
    return best_value[1:]


def run():
    args = parser()
    train(model_name=args.model_name,
          data_name=args.data_name,
          superpixels_name=args.superpixel_name,
          gnn_function_name=args.gnn_function_name,
          lr=args.lr,
          epochs=args.epoch,
          weight_decay=args.weight_decay,
          ratio=args.ratio,
          if_ratio=args.if_ratio,
          seeds=args.seeds,
          n_segments=args.n_segments,
          device_name=args.device_name,
          train_nums=args.train_nums,
          scale_layer=args.scale_layer)


if __name__ == '__main__':
    run()
