from train import train
import numpy as np
import torch

data_name = ['Indian_pines', 'Salinas', 'PaviaU']
train_numss = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
spn = 'SLIC'

with open('experiment_results3.txt', 'w') as file:
    file.write("Experiment Results\n")
    results = {}

    for data_n in data_name:
        for train_nums in train_numss:
            oa_list = []
            kappa_list = []
            aa_list = []
            ac_lists = []

            for _ in range(5):
                ac_list_, OA_, AA_, kappa_ = train(model_name='segnet_v1',
                                                   data_name=data_n,
                                                   superpixels_name=spn,
                                                   gnn_function_name='fagcn',
                                                   epochs=500,
                                                   train_nums=train_nums,
                                                   scale_layer=1)
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
            ac_lists_mean = np.mean(ac_lists, axis=0)
            ac_lists_std = np.std(ac_lists, axis=0)

            results[data_n][str(train_nums)] = {
                "OA": {
                    "mean": oa_mean,
                    "std": oa_std
                },
                "Kappa": {
                    "mean": kappa_mean,
                    "std": kappa_std
                },
                "AA": {
                    "mean": aa_mean,
                    "std": aa_std
                },
                "AC": {
                    "mean": list(ac_lists_mean),
                    "std": list(ac_lists_std)
                }
            }
            # 将统计结果写入文件
            stats_str = f"Stats for Data: {data_n}, GNN: fagcn , OA Mean: {oa_mean:.4f}, OA Std: {oa_std:.4f}, Kappa Mean: {kappa_mean:.4f}, Kappa Std: {kappa_std:.4f}, AA Mean: {aa_mean:.4f}, AA Std: {aa_std:.4f}\n"
            file.write(stats_str)
            print(stats_str)

for data, res in results.items():
    print(f"Results for {data}:")
    for gnn, stats in res.items():
        print(f"\tGNNS: {gnn}")
        for metric, value in stats.items():
            print(
                f"\t\t{metric}: Mean = {value['mean']:.4f}, Std = {value['std']:.4f}"
            )
