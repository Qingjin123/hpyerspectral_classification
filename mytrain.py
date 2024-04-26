from load_data import loadData
from logger import readYaml
from process_data import normData, countLabel, sampleMask
from process_data import superpixels
from utils import parser, performance, mkdir, getDevice, getOptimizer, getLoss, setupSeed, getMetrics
from show import show_data, show_mask, plot_slic
from model import mynet
import torch.utils.tensorboard as tb
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

def train(args: dict = parser(), yaml_path: str = 'dataset/data_info.yaml'):
    superpixels_name = args.superpixel_name
    model_name = args.model_name
    data_name = args.data_name

    print('data name:', data_name)
    print('superpixels name:', superpixels_name)
    data, label = loadData(readYaml(yaml_path), data_name)
    print('data shape:', data.shape)
    ndata = normData(data)

    seed = setupSeed(None)
    tb_dir, model_dir, img_dir, png_path = mkdir(data_name, model_name)

    writer = tb.SummaryWriter(tb_dir)
    writer.add_text('data name:', data_name)
    writer.add_text('lr:', str(args.lr))
    writer.add_text('seed:', str(seed))

    show_data(ndata, label, data_name, if_pca=False, if_tsne=False, save_png_path=png_path)

    count, class_num = countLabel(label)
    train_mask, test_mask = sampleMask(label, count, 0.15)
    show_mask(train_mask, label, data_name, 'train', png_path)
    show_mask(test_mask, label, data_name, 'test', png_path)

    seg_index, block_num = superpixels(ndata, superpixels_name)
    plot_slic(seg_index, data_name, png_path)
    adj_mask = np.ones((block_num, block_num), dtype=np.float32)
    print('block_num:', block_num)

    device = getDevice('mps')

    ndata = torch.from_numpy(ndata).to(device)
    label =  torch.from_numpy(label).to(device)
    train_mask = torch.from_numpy(train_mask).to(device)
    test_mask = torch.from_numpy(test_mask).to(device)
    seg_index = torch.from_numpy(seg_index).to(device)
    adj_mask = torch.from_numpy(adj_mask).to(device)

    model = mynet.SegNet(
        in_channels=ndata.shape[2],
        block_num=block_num,
        class_num=class_num+1,
        batch_size=args.batch_size,
        gnn_name='gat',
        adj_mask=adj_mask,
        device=device
        )
    
    model.to(device)

    optimizer, scheduler = getOptimizer('adam', model.parameters(), args.lr, 0.0001)

    loss_function = getLoss('cross_entropy')
    
    # record
    train_loss = []
    train_los = []
    test_loss = []
    test_acc = []
    record = []
    best_value = [0 ,0, 0, 0, []] #[oa, aa, kappa]
    
    def prediction(classes:torch.Tensor, gt:torch.Tensor, mask:torch.tensor):
        sum = mask.sum()
        
        train_gt = gt * mask
        train_gt = label * train_mask
        pre_gt = torch.cat((train_gt.unsqueeze(0).to(device), classes[0]),dim=0)
        pre_gt = pre_gt.view(class_num+2,-1).permute(1,0)
        pre_gt_ = pre_gt[torch.argsort(pre_gt[:,0],descending=True)]
        pre_gt_ = pre_gt_[:int(sum)]
        return pre_gt_
        
    model.to(device)
    for epoch in tqdm.tqdm(range(args.epoch)):
        model.train()
        # print('training, epochs: ', epoch)
        final, finalsoft = model(ndata, seg_index)

        pred_gt = prediction(final, label, train_mask)
        
        loss1 = loss_function(pred_gt[:,1:],pred_gt[:,0].long())

        train_loss.append(float(loss1))
        
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar('train_loss', train_loss[-1], epoch)
        # print('train_loss: ', train_loss[-1])
        
        # 记录梯度
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        if (epoch % 3 == 0) and epoch >= 1:
            # model.eval()
            with torch.no_grad():
            # print('\ntesting...')
                final, _ = model(ndata, seg_index)
                pred_gt = prediction(final, label, test_mask)

                loss2 = loss_function(pred_gt[:,1:], pred_gt [:,0].long())
                train_los.append(float(loss1))
                test_loss.append(float(loss2))
                writer.add_scalar('test_loss', test_loss[-1], epoch)
                # writer.add_image('test image', torch.max(torch.softmax(final[0].cpu(), dim =0),dim = 0)[1].cpu()*(label.cpu()>0).float(), epoch)
                OA, AA, kappa, ac_list = performance(pred_gt[:,1:], pred_gt[:,0].long(), class_num+1)
                # OA, AA, kappa, ac_list = getMetrics(pred_gt[:,1:], pred_gt[:,0].long())

                print('epoch: {}, OA: {:.4f}, AA: {:.4f}, Kappa: {:.4f}'.format(epoch, OA, AA, kappa))
                print('ac_list:', ac_list)

                test_acc.append(ac_list)
                record.append([epoch, loss1.item(), loss2.item(), ac_list, OA, AA, kappa])
                # writer.add_scalar('test_acc', ac_list, epoch)
                writer.add_scalar('OA', OA, epoch)
                writer.add_scalar('AA', AA, epoch)
                writer.add_scalar('kappa', kappa, epoch)
                
                if best_value[3] < kappa:
                    best_value = [epoch, OA, AA, kappa, ac_list] 
                    torch.save(model.state_dict(), model_dir  + '/' + 'lr_'+ str(args.lr) + '_model.pth')
                    
                    plt.figure()
                    plt.imshow(torch.max(torch.softmax(final[0].cpu(), dim =0),dim = 0)[1].cpu()*(label.cpu()>0).float())
                    plt.savefig(img_dir + '/' +'DMSGer' + '_epoch_'+str(epoch)+'_OA_'+str(round(OA, 2))+'_AA_'+str(round(AA, 2))+'_KAPPA_'+str(round(kappa, 2))+'.png')
                    plt.close()
    
    
    plt.figure()
    plt.title('train loss and test loss')
    plt.plot(train_los, label='train_loss',c='blue')
    plt.plot(test_loss, label='test_loss', c='red')
    plt.savefig(img_dir + '/'+'DMSGer_' + 'train_and_test_loss' + 'lr' + str(args.lr) +'.png')
    plt.close()
    
    writer.close()
    print('best_oa:',best_value[1], 'best_aa:',best_value[2], 'best_kappa:',best_value[3])
    print('best_accuracy_list:',best_value[4], 'epoch:', best_value[0])


if __name__ == '__main__':
    train()