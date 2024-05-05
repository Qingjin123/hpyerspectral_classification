from train import train
import numpy as np
import torch

data_name = ['Indian_pines', 'Salinas', 'PaviaU']
# gnn_names= ['gcn', 'gcnii', 'gin', 'fagcn']
gnn_names = ['fagcn', 'gcn']

for data_n in data_name:
    for gnn_name in gnn_names:
        oa_list = []
        kappa_list = []
        aa_list = []
        ac_lists = []

        for i in range(3):
            print(f'the {i}th')
            OA_, AA_, kappa_, ac_list_ = train(model_name='segnet_v1',
                                               data_name=data_n,
                                               superpixels_name='SLIC',
                                               gnn_function_name=gnn_name,
                                               epochs=500,
                                               train_nums=5,
                                               n_segments=40,
                                               scale_layer=4)
            oa_list.append(OA_)
            kappa_list.append(kappa_)
            aa_list.append(AA_)
            ac_lists.append(ac_list_)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        oa_mean = np.mean(oa_list)
        oa_std = np.std(oa_list)
        kappa_mean = np.mean(kappa_list)
        kappa_std = np.std(kappa_list)
        aa_mean = np.mean(aa_list)
        aa_std = np.std(aa_list)
        ac_list_mean = np.mean(ac_lists, axis=0)
        ac_list_std = np.std(ac_lists, axis=0)

        print('---------------------------')
        print(f'OA_mean:{oa_mean}')
        print(f'OA_std:{oa_std}')
        print(f'AA_mean:{aa_mean}')
        print(f'AA_std:{aa_std}')
        print(f'Kappa_mean:{kappa_mean}')
        print(f'Kappa_std:{kappa_std}')
        print(f'ac_list_mean:{ac_list_mean}')
        print(f'ac_list_std:{ac_list_std}')
        print('---------------------------')
