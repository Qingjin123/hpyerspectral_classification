from train import train
import numpy as np
import torch

data_name = ['Indian_pines', 'Salinas', 'PaviaU']
# gnn_names= ['gcn', 'gcnii', 'gin', 'fagcn']
gnn_names = ['fagcn']
sls = [1, 2, 3, 4]

for data_n in data_name:
    for sl in sls:
        oa_list = []
        kappa_list = []
        aa_list = []
        ac_lists = []

        for i in range(3):
            print(f'the {i}th')
            OA_, AA_, kappa_, _ = train(model_name='segnet_v1',
                                        data_name=data_n,
                                        superpixels_name='SLIC',
                                        gnn_function_name='fagcn',
                                        epochs=500,
                                        train_nums=5,
                                        n_segments=40,
                                        scale_layer=sl)
            oa_list.append(OA_)
            kappa_list.append(kappa_)
            aa_list.append(AA_)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        oa_mean = np.mean(oa_list)
        oa_std = np.std(oa_list)
        kappa_mean = np.mean(kappa_list)
        kappa_std = np.std(kappa_list)
        aa_mean = np.mean(aa_list)
        aa_std = np.std(aa_list)

        print('---------------------------')
        print(f'OA_mean:{oa_mean}')
        print(f'OA_std:{oa_std}')
        print(f'AA_mean:{aa_mean}')
        print(f'AA_std:{aa_std}')
        print(f'Kappa_mean:{kappa_mean}')
        print(f'Kappa_std:{kappa_std}')
        print('---------------------------')
