import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class show_data:
    '''
        参数：
            数据集 data: np.ndarray
            标签集 label: np.ndarray
            数据集名称 data_name: str
            保存图片的路径 save_png_path: str
        方法：
            显示数据集的 ground truth
            显示数据集的 pca ---- 前三
            显示数据集的 t-SNE
    '''

    def __init__(self,
                 data: np.ndarray,
                 label: np.ndarray,
                 data_name: str,
                 save_png_path: str,
                 if_pca: bool = True,
                 if_tsne: bool = True) -> None:
        h, w, c = data.shape
        self.h = h
        self.w = w
        self.data = data.reshape(h * w, c)
        self.label = label
        self.data_name = data_name
        self.save_png_path = save_png_path

        self.show_gt()
        if if_pca:
            self.plot_pca()
        if if_tsne:
            self.plot_tsne()

    def show_gt(self):
        class_num = len(np.unique(self.label)) - 1
        plt.figure()
        plt.imshow(self.label)
        plt.title('ground truth of {}'.format(self.data_name) + '(' +
                  str(class_num) + ' classes)')
        plt.savefig(self.save_png_path + '/' + self.data_name +
                    ' ground_truth.png')
        plt.close()

    @property
    def pca_data(self, n_components: int = 3):
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(self.data)
        data_pca = data_pca.reshape(self.h, self.w, n_components)
        return data_pca

    def plot_pca(self):
        data_pca = self.pca_data

        # data_pca 的值缩放到 0-1
        data_pca = (data_pca - data_pca.min()) / (
            data_pca.max() - data_pca.min())  # * 255
        plt.figure()
        plt.imshow(data_pca)
        plt.title('PCA of {}'.format(self.data_name))
        plt.savefig(self.save_png_path + '/' + self.data_name +
                    ' pca_data.png')
        plt.close()

    @property
    def tsne_data(self, n_components: int = 2):
        tsne = TSNE(n_components=n_components)
        data_tsne = tsne.fit_transform(self.data)
        return data_tsne

    def plot_tsne(self):
        data_tsne = self.tsne_data
        plt.figure()
        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=self.label, s=1)
        plt.title('t-SNE of {}'.format(self.data_name))
        plt.savefig(self.save_png_path + '/' + self.data_name +
                    ' tsne_data.png')
        plt.close()


def show_mask(mask: np.ndarray, label: np.ndarray, data_name: str,
              plot_name: str, png_path: str):
    plt.figure()
    plt.imshow(mask * label)
    plt.title(data_name)
    plt.savefig(png_path + '/' + data_name + '_' + plot_name + ' mask.png')
    plt.close()


def plot_slic(seg_index: np.ndarray, data_name: str, save_png_path: str):
    plt.figure()
    plt.imshow(seg_index)
    plt.title('slic of {}'.format(data_name))
    plt.savefig(save_png_path + '/' + data_name + ' slic.png')
    plt.close()


def plot_epoch_features(epoch: int, data: np.ndarray, data_name: str,
                        save_png_path: str):
    tsne = TSNE(n_components=2)
    data_tsne = tsne.fit_transform(data)
    plt.figure()
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=epoch, s=1)
    plt.title(f't-SNE of {data_name}_{epoch}')
    plt.savefig(save_png_path + '/' + data_name + f'_{epoch}' +
                ' tsne_data.png')
    plt.close()
