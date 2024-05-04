import subprocess

data = {'Indian_pines': 300, 'Salinas': 300, 'PaviaU': 300}
# gnn_fun = ['gcn', 'gat', 'gin', 'gcnii', 'fagcn']
gnn_fun = ['gcn', 'fagcn']
model_name = ['segnet_v1', 'segnet_v2']
spn = 'SLIC'
# for gnn in gnn_fun:
#     for model_n in model_name:
#         cmd = f'python train.py --model_name {model_n} --data_name Indian_pines --gnn_function_name {gnn} --epoch 500 --train_nums 5 --scale_layer 2'
#         subprocess.call(cmd, shell=True)

for gnn in gnn_fun:
    for model_n in model_name:
        for data_n in data:
            cmd = f'python train.py --model_name {model_n} --data_name {data_n} --gnn_function_name {gnn} --epoch 500 --train_nums 5 --scale_layer 4'
            subprocess.call(cmd, shell=True)

for i in range(1, 5):
    cmd = f'python train.py --model_name segnet_v1 --data_name Indian_pines --gnn_function_name fagcn --epoch 500 --train_nums 5 --scale_layer {i}'

for i in [1, 3, 5, 10, 15, 30]:
    cmd = f'python train.py --model_name segnet_v1 --data_name Indian_pines --gnn_function_name fagcn --epoch 500 --train_nums {i} --scale_layer 4'