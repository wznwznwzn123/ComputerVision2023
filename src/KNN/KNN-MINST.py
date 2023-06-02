import operator
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

relative_path = os.getcwd()

batch_size = 100
train_dataset = dsets.MNIST(root=relative_path + '\pymnist',  # 选择数据的根目录
                            train=True,  # 选择训练集
                            transform=None,  # 不使用任何数据预处理
                            download=True)  # 从网络上下载图片

test_dataset = dsets.MNIST(root=relative_path + '\pymnist',  # 选择数据的根目录
                           train=False,  # 选择测试集
                           transform=None,  # 不适用任何数据预处理
                           download=True)  # 从网络上下载图片

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)


def KNN_classify(k, dis, train_data, train_label, test_data):
    assert dis == 'E' or dis == 'M', 'dis must be E or M, E代表欧拉距离，M代表曼哈顿距离'
    num_test = test_data.shape[0]  # 测试样本的数量
    label_list = []
    if dis == 'E':
        # 欧拉距离的实现
        for i in range(num_test):
            distances = np.sqrt(np.sum(
                ((train_data - np.tile(test_data[i], (train_data.shape[0], 1))) ** 2), axis=1))
            nearest_k = np.argsort(distances)
            top_k = nearest_k[:k]  # 选取前k个距离
            class_count = {}
            for j in top_k:
                class_count[train_label[j]] = class_count.get(
                    train_label[j], 0) + 1
            sorted_class_count = sorted(
                class_count.items(), key=operator.itemgetter(1), reverse=True)
            label_list.append(sorted_class_count[0][0])
    else:
        # 曼哈顿距离
        for i in range(num_test):
            distances = np.sum(
                np.abs(train_data - np.tile(test_data[i], (train_data.shape[0], 1))), axis=1)
            nearest_k = np.argsort(distances)
            top_k = nearest_k[:k]
            class_count = {}
            for j in top_k:
                class_count[train_label[j]] = class_count.get(
                    train_label[j], 0) + 1
            sorted_class_count = sorted(
                class_count.items(), key=operator.itemgetter(1), reverse=True)
            label_list.append(sorted_class_count[0][0])
    return np.array(label_list)

def getXmean(data):
    data = np.reshape(data, (data.shape[0], -1))
    mean_image = np.mean(data, axis=0)
    return mean_image


def centralized(data, mean_image):
    data = data.reshape((data.shape[0], -1))
    data = data.astype(np.float64)
    data -= mean_image  # 减去图像均值，实现领均值化
    return data

if __name__ == '__main__':
    # 训练数据
    train_data = train_loader.dataset.data.numpy()
    mean_image = getXmean(train_data)  # 计算所有图像均值
    train_data = centralized(train_data, mean_image)  # 对训练集图像进行均值化处理
    print(train_data.shape)
    train_label = train_loader.dataset.targets.numpy()
    print(train_label.shape)
    # 测试数据
    test_data = test_loader.dataset.data[:1000].numpy()
    test_data = centralized(test_data, mean_image)  # 对测试集数据进行均值化处理
    print(test_data.shape)
    test_label = test_loader.dataset.targets[:1000].numpy()
    print(test_label.shape)

    # 训练
    test_label_pred = KNN_classify(5, 'M', train_data, train_label, test_data)

    # 计算准确率  
    accuracy = accuracy_score(test_label, test_label_pred)  
    print("Accuracy:", accuracy)

    # 计算召回率  
    recall = recall_score(test_label, test_label_pred,average='macro') 
    print("Recall:", recall)

    # 计算精确度  
    precision = precision_score(test_label, test_label_pred,average='macro')
    print("Precision:", precision)

    # 计算 F1 值  
    f1 = f1_score(test_label, test_label_pred,average='macro')
    print("F1-score:", f1)  


