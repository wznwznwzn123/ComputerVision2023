import os
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

relative_path = os.getcwd()
batch_size = 64
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.1307, ), (0.3081, ))
])
train_dataset = datasets.MNIST(root=relative_path + '\pymnist',train=True,download=True,transform=transform)
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
 
test_dataset = datasets.MNIST(root=relative_path + '\pymnist',train=False,download=True,transform=transform)
test_loader = DataLoader(test_dataset,shuffle=False,batch_size=batch_size)

#  Residual Network（ResNet）网络模型
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels,
        kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels,
        kernel_size=3, padding=1)
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y) #两次卷积后的输出y，加上两次卷积前的输入x
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.rblock1 = ResidualBlock(16) #ResNet网络不改变输入输出维度
        self.rblock2 = ResidualBlock(32)
        self.fc = nn.Linear(512, 10)
    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
     
model = Net()
#构建损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#训练
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

#测试
def test():
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    class_TP = list(0. for i in range(10))
    class_FP = list(0. for i in range(10))
    class_FN = list(0. for i in range(10))


    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]

                class_correct[label] += c[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_TP[label] += 1
                else:
                    class_FP[predicted[i]] += 1
                    class_FN[label] += 1

    acc = [0] * 10 # 存储每个类别的准确率
    prec = [0] * 10 # 存储每个类别的精度
    rec = [0] * 10 # 存储每个类别的召回率
    
    for i in range(10):
        acc[i] = class_correct[i] / class_total[i]
        rec[i] = class_TP[i] / (class_TP[i] + class_FN[i])
        prec[i] = class_TP[i] / (class_TP[i] + class_FP[i])
        print('类别 %d 的准确率: %2d %%' % (i + 1, 100 * class_correct[i] / class_total[i]))
        print('类别 %d 的召回率: %2d %%' % (i + 1, 100 * class_TP[i] / (class_TP[i] + class_FN[i])))
        print('类别 %d 的精度: %2d %%' % (i + 1, 100 * class_TP[i] / (class_TP[i] + class_FP[i])))
        print('类别 %d 的 TP: %d' % (i + 1, class_TP[i]))
        print('类别 %d 的 FP: %d' % (i + 1, class_FP[i]))
        print('类别 %d 的 FN: %d' % (i + 1, class_FN[i]))
        print()

    precisions = [0] * 10 # 存储每个类别的精度
    recalls = [0] * 10 # 存储每个类别的召回率
    APs = [0] * 10
    f1_scores = [0] * 10 # 存储每个类别的F1值
    for i in range(10):
        labelstr = []
        predictedsc = []
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, dim=1)
                for j in range(len(labels)):
                    if labels[j] == i:
                        labelstr.append(1)
                    else:
                        labelstr.append(0)
                    if predicted[j] == i:
                        predictedsc.append(1)
                    else:
                        predictedsc.append(0)
        precision, recall, _ = precision_recall_curve(labelstr, predictedsc, pos_label=1)

        precisions[i]=precision[1]
        recalls[i]=recall[1]

        f1_scores[i] = 2 * recalls[i] * precisions[i] / (recalls[i] + precisions[i])

        AP = average_precision_score(labelstr, predictedsc)     # 慎用AP,其中存在误差(threshold=0)

        APs.append(AP)
        print('类别 %d 的F1值: %.3f' % (i + 1, f1_scores[i]))
        print('类别 %d 的AP值: %.3f' % (i + 1, AP))

    print('所有类别的平均F1值: %.3f' % np.mean(f1_scores))
    print('所有类别的平均AP值: %.3f' % np.mean(APs))

if __name__ == '__main__':
    for epoch in range(1):
        train(epoch)
    test()