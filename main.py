from trains import train
import os
import torch
import subprocess

data_name = ['Indian_pines', 'Salinas', 'PaviaU']
gnn_fun = ['gcn', 'gat', 'gin', 'gcnii', 'fagcn']
model_name = 'segnet'
spn = 'SLIC'
for _ in range(5):
    for data in data_name:
        for gnn in gnn_fun:
                cmd = f'python trains.py --model_name {model_name} --data_name {data}  --gnn_function_name {gnn} --ratio 0.15 --if_ratio False'
                subprocess.call(cmd, shell=True)

n_segments = [30, 40, 50, 70, 100]
ratio = [0.01, 0.05, 0.1, 0.15, 0.2]
dn = 'Indian_pines'
gnnn = 'fagcn'
for n_s in n_segments:
    cmd = f'python train.py --model_name {model_name} --data_name {dn}  --gnn_function_name {gnnn} --ratio 0.15 --if_ratio False --n_segments {n_s}'
    subprocess.call(cmd, shell=True)

for r in ratio:
    cmd = f'python train.py --model_name {model_name} --data_name {dn}  --gnn_function_name {gnnn} --ratio {r} --if_ratio True'
    subprocess.call(cmd, shell=True)
