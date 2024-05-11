from load_data import loadData, saveYaml
from logger import readYaml
from process_data import normData, countLabel, sampleMask
from process_data import superpixels
from utils import parser, performance, mkdir, getDevice
from utils import getOptimizer, getLoss, setupSeed
from show import show_data, show_mask, plot_slic
from model import SegNet_v2, SegNet_v1, TGNet_v1, TGNet_v2
import torch.utils.tensorboard as tb
import numpy as np
import matplotlib.pyplot as plt
import torch
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.console import Console
from rich.live import Live
import time


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

    # data
    data, label = loadData(readYaml(yaml_path), data_name)
    ndata = normData(data)

    # seed
    seed = setupSeed(seeds)

    # mkdir
    tb_dir, model_dir, img_dir, png_path = mkdir(data_name, model_name)

    # tensorboard
    writer = tb.SummaryWriter(tb_dir)
    writer.add_text('data name:', data_name)
    writer.add_text('lr:', str(lr))
    writer.add_text('seed:', str(seed))

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
    elif model_name == 'tgnet_v2':
        Model = TGNet_v2
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

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print("Total trainable parameters:", total_params)

    optimizer, scheduler = getOptimizer('adam', model.parameters(), lr,
                                        weight_decay)

    loss_function = getLoss('cross_entropy')

    # record
    train_loss = []
    test_loss = []
    test_acc = []
    record = []
    best_value = [0, 0, 0, 0, []]  # [oa, aa, kappa]

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="dim")
    table.add_column("Value")

    console = Console()
    parameters = {
        "model name": model_name,
        "data name": data_name,
        "gnn name": gnn_function_name,
        "superpixel name": superpixels_name,
        "class number": str(class_num),
        "seed": str(seed),
        "lr": str(lr),
        "block number": str(block_num),
        "Epoch": "0",
        "Train loss": "N/A",
        "Test loss": "N/A",
        "Test OA": "N/A",
        "Test AA": "N/A",
        "Test Kappa": "N/A",
        "Best Epoch": "N/A",
        "Best OA": "N/A",
        "Best AA": "N/A",
        "Best Kappa": "N/A",
        "Time Spent": "N/A",
    }
    # def update_display(progress, parameters):
    #     table = Table(show_header=True, header_style="bold magenta")
    #     table.add_column("Parameter", style="dim")
    #     table.add_column("Value")

    #     # 为每个参数填充数据
    #     for name, value in parameters.items():
    #         table.add_row(name, str(value))

    #     # 使用 Live 来更新输出
    #     with Live(progress, console=console, refresh_per_second=10) as live:
    #         # progress.refresh()  # 刷新进度条
    #         console.print(table)  # 打印表格

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True  # 进度条完成后自动隐藏
    )

    task = progress.add_task("Epoch:", total=epochs)
    with Live(console=console, refresh_per_second=10):
        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            # print('training, epochs: ', epoch)
            final, finalsoft = model(ndata, seg_index)

            loss1 = loss_function(finalsoft, label)
            loss = loss1 * train_mask
            final_loss = loss.sum() / train_mask.sum()

            train_loss.append(float(loss1))

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            scheduler.step()

            writer.add_scalar('train_loss', train_loss[-1], epoch)

            # 记录梯度
            for name, param in model.named_parameters():
                writer.add_histogram(name,
                                     param.clone().cpu().data.numpy(), epoch)

            with torch.no_grad():
                # print('\ntesting...')
                final, _ = model(ndata, seg_index)

                OA, AA, kappa, ac_list = performance(finalsoft.cpu(),
                                                     label.squeeze(0).cpu(),
                                                     test_mask.cpu(),
                                                     class_num)

                test_acc.append(ac_list)
                record.append([
                    epoch,
                    loss1.item(),
                ])
                # writer.add_scalar('test_acc', ac_list, epoch)
                writer.add_scalar('OA', OA, epoch)
                writer.add_scalar('AA', AA, epoch)
                writer.add_scalar('kappa', kappa, epoch)

                if best_value[3] < kappa:
                    best_value = [epoch, OA, AA, kappa, ac_list]
                    torch.save(
                        model.state_dict(),
                        model_dir + '/' + 'lr_' + str(lr) + '_model.pth')

                    plt.figure()
                    plt.imshow(
                        torch.max(torch.softmax(final[0].cpu(), dim=0),
                                  dim=0)[1].cpu() * (label.cpu() > 0).float())
                    plt.savefig(img_dir + '/' + 'DMSGer' + '_epoch_' +
                                str(epoch) + '_OA_' + str(round(OA, 2)) +
                                '_AA_' + str(round(AA, 2)) + '_KAPPA_' +
                                str(round(kappa, 2)) + '.png')
                    plt.close()

            end_time = time.time()
            parameters['Epoch'] = str(epoch + 1)
            parameters['Train loss'] = str(round(train_loss[-1], 4))
            parameters['Test loss'] = str(round(test_loss[-1], 4))
            parameters['Test OA'] = str(round(OA, 4))
            parameters['Test AA'] = str(round(AA, 4))
            parameters['Test Kappa'] = str(round(kappa, 4))
            parameters['Best Epoch'] = str(best_value[0])
            parameters['Best OA'] = str(round(best_value[1], 4))
            parameters['Best AA'] = str(round(best_value[2], 4))
            parameters['Best Kappa'] = str(round(best_value[3], 4))
            parameters['Time Spent'] = str(round(end_time - start_time, 4))

            # 重新构建表格以更新数据
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Parameter", style="dim")
            table.add_column("Value")
            for name, value in parameters.items():
                table.add_row(name, value)

            # 更新进度条和表格
            progress.update(task, advance=1)
            console.clear()  # 清空之前的输出
            console.print(table)  # 打印最新的表格
            console.print(progress)  # 打印进度条

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
