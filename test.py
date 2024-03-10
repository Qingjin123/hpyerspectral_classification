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

yaml_payh = 'dataset/data_info.yaml'

data_info_yaml = readYaml(yaml_payh)

data_name = 'Indian_pines'
print('data name:', data_name)
data, label = loadData(data_info_yaml, data_name)
print('data shape:', data.shape)

ndata = normData(data)

counts, class_num = countLabel(label)
# print('counts:', counts)
print('class_num:', class_num)

train_mask, test_mask = sampleMask(label, counts)

# superpixels
seg_index, block_num = superpixels(ndata, 'SLIC')
print('block num:', block_num)

adj_mask = np.ones((block_num, block_num), dtype=np.int32)

lr = 0.0005
epoch = 500
batch_size = 1

seed_ = setupSeed()

tb_dir, model_dir, img_dir, png_path = mkdir(data_name, 'DMSGCN')

# tensorboard
writer = tb.SummaryWriter(tb_dir)
writer.add_text('data name:', data_name)
writer.add_text('lr:', str(lr))
writer.add_text('seed:', str(seed_))

device = getDevice('mps')

# to tensor and device
ndata = torch.from_numpy(ndata).to(device)
label =  torch.from_numpy(label).to(device)
train_mask = torch.from_numpy(train_mask).to(device)
test_mask = torch.from_numpy(test_mask).to(device)
seg_index = torch.from_numpy(seg_index).to(device)
adj_mask = torch.from_numpy(adj_mask).to(device)

model = SegNet(
        in_channels=ndata.shape[2],
        block_num=block_num,
        class_num=class_num,
        batch_size=batch_size,
        adj_mask=adj_mask,
        device=device
    )
model.to(device)
optimizer, scheduler = getOptimizer('adam', model.parameters(), lr, 0.00001)
loss = getLoss('crossentropyloss')

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

for epoch in tqdm.tqdm(range(epoch)):
    model.train()
    # print('training, epochs: ', epoch)
    final, features = model(ndata, seg_index)
    pred_gt = prediction(final, label, train_mask)
    
    loss1 = loss(pred_gt[:,1:],pred_gt[:,0].long())

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

            loss2 = loss(pred_gt[:,1:], pred_gt [:,0].long())
            train_los.append(float(loss1))
            test_loss.append(float(loss2))
            writer.add_scalar('test_loss', test_loss[-1], epoch)

            OA, AA, kappa, ac_list = performance(pred_gt [:,1:], pred_gt[:,0].long(),class_num-1)
            test_acc.append(ac_list)
            record.append([epoch, loss1.item(), loss2.item(), ac_list, OA, AA, kappa])
            # writer.add_scalar('test_acc', ac_list, epoch)
            writer.add_scalar('OA', OA, epoch)
            writer.add_scalar('AA', AA, epoch)
            writer.add_scalar('kappa', kappa, epoch)

            if best_value[3] < kappa:
                best_value = [epoch, OA, AA, kappa, ac_list] 
                torch.save(model.state_dict(), model_dir + '/' + '_lr_'+ str(lr) + '_model.pth')
                
                # if epoch >= 200:
                plt.figure()
                plt.imshow(torch.max(torch.softmax(final[0].cpu(), dim =0),dim = 0)[1].cpu()*(label.cpu()>0).float())
                plt.savefig(img_dir+ '/' +'DMSGer' + '_epoch_'+str(epoch)+'_OA_'+str(round(OA, 2))+'_AA_'+str(round(AA, 2))+'_KAPPA_'+str(round(kappa, 2))+'.png')
                plt.close()
              

plt.figure()
plt.title('train loss and test loss')
plt.plot(train_los, label='train_loss',c='blue')
plt.plot(test_loss, label='test_loss', c='red')
plt.savefig(img_dir + '/'+'DMSGer_' + 'train_and_test_loss' + 'lr' + str(lr) +'.png')
plt.close()

writer.close()