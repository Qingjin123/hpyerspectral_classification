import os
from utils import train_model, train_pair
if not os.path.exists('logs'):
    os.makedirs('logs')

data_name = ['Indian_pines', 'Salinas', 'PaviaU']
gnn_fun = ['gcn', 'gat', 'gin', 'gcnii', 'fagcn']
for _ in range(5):
    for data in data_name:
        for gnn in gnn_fun:
            train_pair(data, gnn)

