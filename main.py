from train import train

superpixels_names = ['SLIC', 'SLICS', 'MSLIC', 'SLICO', 'Felzenszwalb','LSC']
dataset_names = ['Indian_pines', 'PaviaU', 'Salinas']

for superpixels_name in superpixels_names:
    for dataset_name in dataset_names:
        best_value = train(superpixels_name=superpixels_name, data_name=dataset_name)
        print('superpixels name:', superpixels_name)
        print('dataset name:', dataset_name)
        print('best_oa:',best_value[1], 'best_aa:',best_value[2], 'best_kappa:',best_value[3])
        print('best_accuracy_list:',best_value[4], 'epoch:', best_value[0])