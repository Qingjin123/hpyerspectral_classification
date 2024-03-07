import os

# import torch

# torch.cuda.empty_cache()
lr_list = [0.001, 0.001, 0.0005, 0.0005, 0.0001, 0.0001]
data_name = ["Indian_pines", "PaviaU", "Salinas"]

for lr in lr_list:
    os.system(f"python main.py --epoch 1000 --lr {lr}")

    # 清空 GPU 缓存
    # torch.cuda.empty_cache()

for lr in lr_list:
    os.system(f"python main.py --epoch 1000 --lr {lr} --data_name PaviaU")
    # torch.cuda.empty_cache()

for lr in lr_list:
    os.system(f"python main.py --epoch 1000 --lr {lr} --data_name Salinas")
    # torch.cuda.empty_cache()

