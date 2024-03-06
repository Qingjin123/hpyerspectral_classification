import config.yamls as yamls
from process import data_process, show_process
from data import load_data, show_data
import torch
from model.segnet import segnet
from util import utils
import tqdm
import torch.utils.tensorboard as tb
import matplotlib.pyplot as plt

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
    print(loader.data_shape)

    # plot data 
    show_data.ShowData(ndata, label, data_name, if_pca=False, if_tsne=False, save_png_path=png_path)

    # data_process
    count, class_num = data_process.count_label(label)
    train_mask, test_mask = data_process.sample_mask(label, count, 0.15)
    show_process.show_mask(train_mask, label, data_name, 'train', png_path)
    show_process.show_mask(test_mask, label, data_name, 'test', png_path)
    
    # slic 
    seg_index, block_num = data_process.slic_data(ndata)
    show_process.plot_slic(seg_index, data_name, png_path)
    adj_matrix = data_process.adj_matrix(block_num)

    # parameters
    args = utils.parser()
    hyperparameters = utils.hyperparameter(args)
    hyperparameters['lr'] = lr
    hyperparameters['epoch'] = epochs

    # device
    device = utils.device('mps')

    # to tensor and device
    ndata = torch.from_numpy(ndata).to(device)
    label =  torch.from_numpy(label).to(device)
    train_mask = torch.from_numpy(train_mask).to(device)
    test_mask = torch.from_numpy(test_mask).to(device)
    seg_index = torch.from_numpy(seg_index).to(device)
    adj_matrix = torch.from_numpy(adj_matrix).to(device)
    
    # model
    model = segnet(
        in_channels=loader.data_shape[2],
        block_num=block_num,
        class_num=class_num,
        batch_size=hyperparameters['batch_size'],
        adj_mask=adj_matrix,
        device=device
    )
    
    # optimizer
    optimizer, scheduler = utils.optimizer('adam', model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])

    # loss
    loss_function = utils.loss('cross_entropy')
    performance = utils.performance

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
        pre_gt = torch.cat((train_gt.unsqueeze(0).to(device), classes[0]),dim=0)
        pre_gt = pre_gt.view(class_num+1,-1).permute(1,0)
        pre_gt_ = pre_gt[torch.argsort(pre_gt[:,0],descending=True)]
        pre_gt_ = pre_gt_[:int(sum)]
        return pre_gt_
        
    model.to(device)
    for epoch in tqdm.tqdm(range(hyperparameters['epoch'])):
        model.train()
        # print('training, epochs: ', epoch)
        final, features = model(ndata, seg_index)
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
            model.eval()
            with torch.no_grad():
            # print('\ntesting...')
                final, _ = model(ndata, seg_index)
                
                pred_gt = prediction(final, label, test_mask)

                loss2 = loss_function(pred_gt[:,1:], pred_gt [:,0].long())
                train_los.append(float(loss1))
                test_loss.append(float(loss2))
                writer.add_scalar('test_loss', test_loss[-1], epoch)

                OA, AA, kappa, ac_list = performance(pred_gt [:,1:], pred_gt[:,0].long(),class_num)
                test_acc.append(ac_list)
                record.append([epoch, loss1.item(), loss2.item(), ac_list, OA, AA, kappa])
                # writer.add_scalar('test_acc', ac_list, epoch)
                writer.add_scalar('OA', OA, epoch)
                writer.add_scalar('AA', AA, epoch)
                writer.add_scalar('kappa', kappa, epoch)

                if best_value[3] < kappa:
                    best_value = [epoch, OA, AA, kappa, ac_list] 
                    torch.save(model.state_dict(), model_dir + 'lr_'+ str(lr) + '_model.pth')
                    
                    # if epoch >= 200:
                    plt.figure()
                    plt.imshow(torch.max(torch.softmax(final[0].cpu(), dim =0),dim = 0)[1].cpu()*(label.cpu()>0).float())
                    plt.savefig(img_dir +'DMSGer' + '_epoch_'+str(epoch)+'_OA_'+str(round(OA, 2))+'_AA_'+str(round(AA, 2))+'_KAPPA_'+str(round(kappa, 2))+'.png')
                    plt.close()
                    # print('\nepoch',epoch,':','OA:',round(OA,4),'AA:',round(AA,4),'KAPPA:',round(kappa,4))
                    # print('epoch',epoch,':','Accuracy_list:',ac_list)
                    # print('\n')
                    # plt.close()
            
        # # early stop
        # if test_loss[-1]>test_loss[-2]>test_loss[-3]>test_loss[-4] and epoch>100:
        #     print('early stop, and epoch:',epoch)
        #     break

        # if epoch >= 3000:
        #     optimizer, scheduler = utils.optimizer('SGD', model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
    
    
    plt.figure()
    plt.title('train loss and test loss')
    plt.plot(train_los, label='train_loss',c='blue')
    plt.plot(test_loss, label='test_loss', c='red')
    plt.savefig(img_dir +'DMSGer_' + 'train_and_test_loss' + 'lr' + str(lr) +'.png')
    plt.close()
    
    writer.close()
    return best_value, model_dir

info = {
    'lr':1,
    'data_name':'Indian_pines',
    'model_name':'DMSGer',
    'best_oa':0,
    'best_aa':0,
    'best_kappa':0,
    'best_accuracy_list':[],
}

# train_list = [[0.0005, 500],[0.0005, 700],[0.001, 700],[0.001, 2000], [0.0005, 2500], [0.0001, 3000]]

# for learn_rate, epoch in train_list:
for i in range(10):
    learn_rate = 0.0005
    epoch = 3000
    print('-----------------------------------')
    data_name = 'Indian_pines'
    print('learn_rate:', learn_rate)
    best_value, model_dir = train(data_name=data_name, model_name='DMSGCN', lr = learn_rate, epochs=epoch)
    print('best_oa:',best_value[1], 'best_aa:',best_value[2], 'best_kappa:',best_value[3])
    print('best_accuracy_list:',best_value[4], 'epoch:', best_value[0])
    info['lr'] = learn_rate
    info['best_oa'] = best_value[1]
    info['best_aa'] = best_value[2]
    info['best_kappa'] = best_value[3]
    info['best_accuracy_list'] = best_value[4]

    # yamls.save_yaml(info, model_dir + data_name +'_' +str(info['lr']) +'.yaml')
    

