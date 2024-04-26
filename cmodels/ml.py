import sklearn.semi_supervised as skssl
import matplotlib.pyplot as plt
import numpy as np

"""
    使用sklearn标签传播算法进行学习
"""

from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

class HyperspectralPixelClassifier:
    def __init__(self):
        self.model = LabelPropagation()
        self.scaler = StandardScaler()

    def getData(self,X, y_true):
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.85, random_state=42)
        return X_train, X_test, y_train, y_test

    def preprocess(self, X):
        """
        标准化特征
        """
        return self.scaler.fit_transform(X)

    def fit(self, X, y):
        """
        训练分类器
        """
        X = self.preprocess(X)
        self.model.fit(X, y)

    def predict(self, X):
        """
        预测
        """
        X = self.preprocess(X)
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        """
        评估模型精度
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")

    def visualize_classification(self, X, y_true, save_path):
        """
        可视化分类结果
        """
        y_pred = self.predict(X)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', edgecolor='k', s=20, label='True Label')
        plt.title('True Labels')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=20, label='Predicted Label')
        plt.title('Predicted Labels')
        plt.legend()
        plt.savefig(save_path + 'classification_results.png')

    def run(self, X, y_true, save_path):
        X_train, X_test, y_train, y_test = self.split_data(X, y_true)
        self.fit(X_train, y_train)
        self.evaluate(X_test, y_test)
        self.visualize_classification(X_test, y_test, save_path)
