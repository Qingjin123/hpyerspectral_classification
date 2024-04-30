import os
import torch
import subprocess

# 设置CUDA设备，假设使用GPU 0和GPU 1
gpu_count = torch.cuda.device_count()

# 将所有可用的GPU编号设置为CUDA_VISIBLE_DEVICES环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(gpu_count))

data_name = ['Indian_pines', 'Salinas', 'PaviaU']
gnn_fun = ['gcn', 'gat', 'gin', 'gcnii', 'fagcn']
model_name = 'gcn_base'
spn = 'SLIC'
gpu_ids = list(range(gpu_count))

count = 0
processes = []
for _ in range(5):
    for data in data_name:
        for gnn in gnn_fun:
            gpu_id = gpu_ids[count % len(gpu_ids)]
            cmd = f'python train_dist.py --model_name {model_name} --data_name {data} --gnn_function_name {gnn} --ratio 0.15 --if_ratio False --gpu {gpu_id}'
            # 使用Popen开始一个新的进程
            process = subprocess.Popen(cmd, shell=True)
            processes.append(process)
            count += 1

# 等待所有启动的进程完成
for process in processes:
    process.wait()

n_segments = [30, 40, 50, 70, 100]
ratio = [0.01, 0.05, 0.1, 0.15, 0.2]
dn = 'Indian_pines'
gnnn = 'fagcn'


count = 0
processes = []

# 处理 n_segments 部分
for n_s in n_segments:
    gpu_id = gpu_ids[count % len(gpu_ids)]
    cmd = f'python train_dist.py --model_name {model_name} --data_name {dn} --gnn_function_name {gnnn} --ratio 0.15 --if_ratio False --n_segments {n_s} --gpu {gpu_id}'
    process = subprocess.Popen(cmd, shell=True)
    processes.append(process)
    count += 1

# 处理 ratio 部分
for r in ratio:
    gpu_id = gpu_ids[count % len(gpu_ids)]
    cmd = f'python train_dist.py --model_name {model_name} --data_name {dn} --gnn_function_name {gnnn} --ratio {r} --if_ratio True --gpu {gpu_id}'
    process = subprocess.Popen(cmd, shell=True)
    processes.append(process)
    count += 1

# 等待所有启动的进程完成
for process in processes:
    process.wait()

