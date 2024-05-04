import subprocess

data = {'Indian_pines': 300, 'Salinas': 300, 'PaviaU': 300}
# gnn_fun = ['gcn', 'gat', 'gin', 'gcnii', 'fagcn']
gnn_fun = ['gcn', 'fagcn']
model_name = ['tgnet_v1', 'segnet_v1']  # , 'segnet_v2']
spn = 'SLIC'
for gnn in gnn_fun:
    for model_n in model_name:
        cmd = f'''python train.py --model_name {model_n}
            --data_name Indian_pines
            --gnn_function_name {gnn} --epoch 500 --n_segments 5
            --scale_layer 1'''
        subprocess.call(cmd, shell=True)
