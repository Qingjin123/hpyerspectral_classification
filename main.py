import subprocess

data_name = ['Indian_pines', 'Salinas', 'PaviaU']
gnn_fun = ['gcn', 'gat', 'gin', 'gcnii', 'fagcn']
model_name = 'gnet '
spn = 'SLIC'
for _ in range(5):
    for data in data_name:
        for gnn in gnn_fun:
                cmd = f'python trains.py --model_name {model_name+gnn} --data_name {data}  --gnn_function_name {gnn}'
                subprocess.call(cmd, shell=True)

model_name = 'tgnet_v1'
for _ in range(5):
    for data in data_name:
        for gnn in gnn_fun:
                cmd = f'python train_tgnet.py --model_name {model_name+gnn} --data_name {data}  --gnn_function_name {gnn} '
                subprocess.call(cmd, shell=True)

n_segments = [30, 40, 50, 70, 100]
train_nums = [1, 3, 5, 10, 15, 20, 30]
dn = 'Indian_pines'
gnnn = 'fagcn'
for n_s in n_segments:
    cmd = f'python trains.py --model_name fagcn --data_name {dn}  --gnn_function_name {gnnn}  --n_segments {n_s}'
    subprocess.call(cmd, shell=True)
    cmd = f'python train_tgnet.py --model_name tgnet_v2 --data_name {dn}  --gnn_function_name {gnnn}  --n_segments {n_s}'
    subprocess.call(cmd, shell=True)

for r in train_nums:
    cmd = f'python trains.py --model_name fagcn --data_name {dn}  --gnn_function_name {gnnn} --train_nums {r}'
    subprocess.call(cmd, shell=True)
    cmd = f'python train_tgnet.py --model_name tgnet_v2 --data_name {dn}  --gnn_function_name {gnnn} --train_nums {r}'
    subprocess.call(cmd, shell=True)
