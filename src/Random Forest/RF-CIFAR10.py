import torch
import torchvision
import torchvision.transforms as transforms
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import random

transform = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
    ])

train_dataset = torchvision.datasets.CIFAR10(root='D:/桌面/计算机视觉/image identification/cifar10', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='D:/桌面/计算机视觉/image identification/cifar10', train=False, download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

x_train = train_dataset.data.reshape(50000, 32*32*3)
y_train = train_dataset.targets
x_test = test_dataset.data.reshape(10000, 32*32*3)
y_test = test_dataset.targets

x_train = x_train[:20000]
y_train = y_train[:20000]
x_test = x_test[:5000]
y_test = y_test[:5000]

class_TP = list(0. for i in range(10))
class_FP = list(0. for i in range(10))
class_FN = list(0. for i in range(10))

APs = [0] * 10
rec = [0] * 10
prec = [0] * 10
f1_scores = [0] * 10
correct = 0
total = 0
# 创建10个SVM分类器
for i in range(10):
    clf_rf = RandomForestClassifier(n_jobs=-1)
    
    # 处理数据
    y_train_i = [1 if y == i else 0 for y in y_train]

    y_test_i = [1 if y == i else 0 for y in y_test]

    # 训练模型
    clf_rf.fit(x_train, y_train_i)

    # 预测测试集
    y_score_i = clf_rf.predict_proba(x_test)
    y_score_i = y_score_i[:, 1]
    # 指标  其中，acc1、acc、rec、prec、TP、FP、FN不受阈值及score影响

    ## 准确率
    y_pred_i = clf_rf.predict(x_test)
    acc1 = accuracy_score(y_test_i, y_pred_i)
    print('类别', classes[i], '的准确率为:', acc1)
    
    ## 计算TP，FP，FN
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for j in range(len(y_test_i)):
        if y_test_i[j] == 1 and y_pred_i[j] == 1:
            TP += 1
            correct += 1
            total += 1
        elif y_test_i[j] == 0 and y_pred_i[j] == 1:
            FP += 1
            total += 1
        elif y_test_i[j] == 1 and y_pred_i[j] == 0:
            FN += 1
            correct += 1
            total += 1
        elif y_test_i[j] == 0 and y_pred_i[j] == 0:
            TN += 1
            total += 1
    print('类别', classes[i], '的TP为:', TP)
    print('类别', classes[i], '的FP为:', FP)
    print('类别', classes[i], '的FN为:', FN)
    print('类别', classes[i], '的TN为:', TN)

    class_TP[i] = TP
    class_FP[i] = FP
    class_FN[i] = FN

    ## 召回率、精确率
    rec[i] = class_TP[i] / (class_TP[i] + class_FN[i])
    prec[i] = class_TP[i] / (class_TP[i] + class_FP[i])
    print('类别', classes[i], '的召回率为:', rec[i])
    print('类别', classes[i], '的精确率为:', prec[i])
    ## F1
    f1_scores[i] = 2 * rec[i] * prec[i] / (rec[i] + prec[i])

    ## AP
    precision, recall, thresholds = precision_recall_curve(y_test_i, y_score_i)
    AP = average_precision_score(y_test_i, y_score_i)
    APs[i] = AP
    print('类别', classes[i], '的F1为:', f1_scores[i])
    print('类别', classes[i], '的AP为:', AP)

    ## PR曲线   
    plt.clf()
    plt.plot(precision, recall, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve of RF for ' + classes[i])
    plt.legend(loc="lower left")
    if not os.path.exists('PLOTforRF-CIFAR10'):
        os.makedirs('PLOTforRF-CIFAR10')
    plt.savefig('PLOTforRF-CIFAR10/PRclass' + classes[i] + '.png')

    ## ROC
    
    fpr, tpr, thresholds = roc_curve(y_test_i, y_score_i)
    roc_auc = auc(fpr, tpr)
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of RF for ' + classes[i])
    plt.legend(loc="lower right")
    plt.savefig('PLOTforRF-CIFAR10/ROCclass' + classes[i] + '.png')

    
# 所有类别的mAP
print('RF的测试准确率: %.3f' % (correct / total))
print('所有类别的平均召回率: %.3f' % np.mean(rec))
print('所有类别的平均精度: %.3f' % np.mean(prec))
print('所有类别的平均F1值: %.3f' % np.mean(f1_scores))
print('所有类别的平均AP值: %.3f' % np.mean(APs))

# 最后可能需要把所有类别的各种指标求平均，这里未求

