from load_data import loadData
from logger import readYaml
from cmodels.ml import HyperspectralPixelClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import numpy as np
import warnings


warnings.filterwarnings('ignore')


def train_lp(data_name: str, yaml_path: str = 'dataset/data_info.yaml'):
    # 加载数据
    data, label = loadData(readYaml(yaml_path), data_name)
    h, w, c = data.shape
    data = data.reshape(h * w, c)
    label = label.reshape(h * w)

    # 训练模型
    model = HyperspectralPixelClassifier()
    model.run(data, label)

    # 计算预测结果
    y_true = model.y_true
    y_pred = model.y_pred

    # 剔除第0类
    non_zero_indices = y_true != 0
    y_true = y_true[non_zero_indices]
    y_pred = y_pred[non_zero_indices]

    # 计算OA、Kappa
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 计算AA
    denominator_aa = conf_matrix.sum(axis=1)
    denominator_aa[denominator_aa == 0] = 1  # 避免除以零
    aa = np.nan_to_num(conf_matrix.diagonal() / denominator_aa, nan=0)
    aa_mean = aa.mean()

    # 计算AC_list
    denominator_ac = conf_matrix.sum(axis=0)
    denominator_ac[denominator_ac == 0] = 1  # 避免除以零
    ac_list = np.nan_to_num(conf_matrix.diagonal() / denominator_ac, nan=0)

    # # 输出结果
    # print(f'{data_name} OA: {oa:.4f}')
    # print(f'Kappa: {kappa:.4f}')
    # print(f'AA: {aa_mean:.4f}')
    # print('AC_list:', ac_list[1:])  # 排除第0类
    return oa, kappa, aa_mean, ac_list


if __name__ == '__main__':
    yaml_path = 'dataset/data_info.yaml'
    data_names = ['Indian_pines', 'PaviaU', 'Salinas']  # 替换成你的数据名称列表

    oa_list = []
    kappa_list = []
    aa_list = []
    ac_lists = []

    for data_name in data_names:
        print(f'data_name:{data_name}')
        for _ in range(5):  # 训练五次
            oa, kappa, aa_mean, ac_list = train_lp(data_name, yaml_path)

            # 添加到列表中
            oa_list.append(oa)
            kappa_list.append(kappa)
            aa_list.append(aa_mean)
            ac_lists.append(ac_list)

        # 计算平均值
        avg_oa = np.mean(oa_list)
        avg_kappa = np.mean(kappa_list)
        avg_aa = np.mean(aa_list)
        avg_ac_list = np.mean(ac_lists, axis=0)

        # 计算标准差
        std_oa = np.std(oa_list)
        std_kappa = np.std(kappa_list)
        std_aa = np.std(aa_list)
        std_ac_list = np.std(ac_lists, axis=0)

        # 输出平均值和标准差
        print(f'Average OA: {avg_oa:.4f} ± {std_oa:.4f}')
        print(f'Average Kappa: {avg_kappa:.4f} ± {std_kappa:.4f}')
        print(f'Average AA: {avg_aa:.4f} ± {std_aa:.4f}')
        print('Average AC_list:', avg_ac_list)
        print('Std AC_list:', std_ac_list)
