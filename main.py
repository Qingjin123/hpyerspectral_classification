from train import train
import os
import torch

superpixels_names = ['SLIC', 'SLICS', 'MSLIC', 'SLICO', 'Felzenszwalb', 'LSC']
data_trains = {
    'Indian_pines':500, 
    'PaviaU':200, 
    'Salinas':300
    }

for superpixels_name in superpixels_names:
    for key, value in data_trains.items():
        os.system(f'python train.py --data_name {key} --superpixel_name {superpixels_name} --epoch {value}')