from abc import abstractmethod
from util.config import argparse, setup_seed
from util.utils import device, mkdir, optimizer, loss
import torch
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer:
    def __init__(self, config: dict):
        if config == None:
            config = argparse()
        self.config = config
        self.seed = setup_seed(self.config.seed)
        self.device = device(self.config.device)
        self.tb_dir, self.model_dir, self.img_dir, self.png_dir = mkdir(self.config.data_name, self.config.model_name)
        self.writer = SummaryWriter(log_dir=self.tb_dir)
        self.best_valid_loss = float('inf')
        self.best_epoch = 0
        
        self.train_loss = []
        self.test_loss = []
        self.test_acc = []
        
    @abstractmethod
    def create_model(self):
        self.model = None
        
    def create_optimizer(self, optimizer_name='adam'):
        self.optimizer, self.scheduler = optimizer(optimizer_name, self.model.parameters(), self.config.lr, self.config.weight_decay)
    
    def create_loss(self, loss_name='crossentropyloss'):
        self.loss = loss(loss_name)
    
    def create_data(self, data, label, seg_index, train_mask, test_mask):
        self.data = torch.tensor(data).to(self.device)
        self.label = torch.tensor(label).to(self.device)
        self.seg_index = torch.tensor(seg_index).to(self.device)
        self.train_mask = torch.tensor(train_mask).to(self.device)
        self.test_mask = torch.tensor(test_mask).to(self.device)
    
    @abstractmethod
    def _train_step(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def _test_step(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def run(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def logger(self, param: dict):
        with open(f'{self.model_dir}/{self.model_name}_log.txt', 'a') as f:
            f.write(f'{param}\n')
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_valid_loss': self.best_valid_loss,
        }
        torch.save(checkpoint, f'{self.model_dir}/{self.model_name}_best.pth')
        
    def load_best_checkpoint(self):
        checkpoint = torch.load(f'{self.model_dir}/{self.model_name}_best.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_valid_loss = checkpoint['best_valid_loss']
        self.best_epoch = checkpoint['epoch']
        print(f"Loaded best checkpoint from epoch {self.best_epoch} with validation loss {self.best_valid_loss:.4f}")
