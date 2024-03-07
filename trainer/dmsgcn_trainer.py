from trainer.base_trainer import BaseTrainer
from model.segnet import SegNet
import torch
from util.metrics import performance
from tqdm import tqdm


BATCH_SIZE = 1

class DMSGCNTrainer(BaseTrainer):
    def __init__(self, config: dict):
        super().__init__(config)
        # self.model = self.create_model()
        # self.data = self.create_data()
        self.kappa = 0
        
    def create_model(self, in_channels, block_num, class_num, adj_mask):
        self.model = SegNet(
            in_channels=in_channels,
            block_num=block_num,
            class_num=class_num,
            batch_size = BATCH_SIZE,
            adj_mask=adj_mask,
            device=self.device
        )
        self.class_num = class_num
        self.block_num = block_num
        self.model.to(self.device)
        
    def create_data(self, data, label, seg_index, train_mask, test_mask):
        super().create_data(data, label, seg_index, train_mask, test_mask)
        
    def _prediction(self, classes:torch.Tensor, gt:torch.Tensor, mask:torch.tensor):
        sum = mask.sum()
        train_gt = gt * mask
        pre_gt = torch.cat((train_gt.unsqueeze(0).to(self.device), classes[0]),dim=0)
        pre_gt = pre_gt.view(self.class_num+1,-1).permute(1,0)
        pre_gt_ = pre_gt[torch.argsort(pre_gt[:,0],descending=True)]
        pre_gt_ = pre_gt_[:int(sum)]
        return pre_gt_
    
    def _train_step(self, epoch):
        self.model.train()
        final, _ = self.model(self.data, self.seg_index)
        pred_gt = self._prediction(final, self.label, self.train_mask)
        
        loss = self.loss(pred_gt[:,1:],pred_gt[:,0].long())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        self.train_loss.append(loss.item())
        self.writer.add_scalar('train_loss', loss.item(), epoch)
        
        # 记录梯度
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            
        return pred_gt

    
    def _test_step(self, epoch):
        self.model.eval()
        with torch.no_grad():
        # print('\ntesting...')
            final, _ = self.model(self.data, self.seg_index)
            pred_gt = self._prediction(final, self.label, self.test_mask)
            loss = self.loss(pred_gt[:,1:],pred_gt[:,0].long())

            self.test_loss.append(loss.item())
            loss = self.loss(pred_gt[:,1:], pred_gt [:,0].long())
            self.writer.add_scalar('test_loss', loss.item(), epoch)
        return pred_gt
            
    def run(self):
        # train
        try:
            for epoch in tqdm(range(self.config.epoch)):
                _ = self._train_step(epoch)
            
                if epoch >=100 and epoch % 10 == 0:
                    pred_gt = self._test_step(epoch)
                    self._performance(epoch, pred_gt)

        except KeyboardInterrupt:
            print("Training interrupted by user.")
            self.save_checkpoint(self.best_epoch)
            print(f"Saved checkpoint from epoch {self.best_epoch} with validation loss {self.best_valid_loss:.4f}")

    def _performance(self, epoch, pred_gt):
        OA, AA, kappa, ac_list = performance(pred_gt[:,1:], pred_gt[:,0].long(), self.class_num)
        self.writer.add_scalar('OA', OA, epoch)
        self.writer.add_scalar('AA', AA, epoch)
        self.writer.add_scalar('kappa', kappa, epoch)
        record = {
            'data': self.config.data_name,
            'model': self.config.model_name,
            'seed': self.seed,
            'lr': self.config.lr,
            'epoch': epoch,
            'OA': OA,
            'AA': AA,
            'kappa': kappa,
            'ac_list': ac_list,
        }
        self.logger(record)
        if kappa > self.kappa:
            self.kappa = kappa
            k = round(kappa, 4)
            torch.save(self.model.state_dict(), f'{self.model_dir}/{self.config.model_name}_kappa:{k}' +'_model.pth')
        