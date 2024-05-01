import subprocess


# model_name = 'gcn_base'
# spn = 'SLIC'
# for _ in range(5):
#     for data in data_name:
#         for gnn in gnn_fun:
#                 cmd = f'python trains.py --model_name {model_name} --data_name {data}  --gnn_function_name {gnn} --ratio 0.15 --if_ratio False'
#                 subprocess.call(cmd, shell=True)

# n_segments = [30, 40, 50, 70, 100]
# ratio = [0.01, 0.05, 0.1, 0.15, 0.2]
# dn = 'Indian_pines'
# gnnn = 'fagcn'
# for n_s in n_segments:
#     cmd = f'python train.py --model_name {model_name} --data_name {dn}  --gnn_function_name {gnnn} --ratio 0.15 --if_ratio False --n_segments {n_s}'
#     subprocess.call(cmd, shell=True)

# for r in ratio:
#     cmd = f'python train.py --model_name {model_name} --data_name {dn}  --gnn_function_name {gnnn} --ratio {r} --if_ratio True'
#     subprocess.call(cmd, shell=True)

data = {'Indian_pines':300, 'Salinas':300, 'PaviaU':300}
gnn_fun = ['gcn', 'gat', 'gin', 'gcnii', 'fagcn']
model_name = ['tgnet_v1', 'segnet_v1', 'segnet_v2']
spn = 'SLIC'
for gnn in gnn_fun:
    for model_n in model_name:
        cmd = f'python train.py --model_name {model_n} --data_name Indian_pines  --gnn_function_name {gnn} --epoch 300 --n_segments 5'
        subprocess.call(cmd, shell=True)



# for data, epochs in data.items():
#     for gnn in gnn_fun:
#         for model_name in model_name:
#             cmd = f'python train.py --model_name {model_name} --data_name {data}  --gnn_function_name {gnn} --epoch {epochs}'
