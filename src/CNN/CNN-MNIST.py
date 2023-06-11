import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

relative_path = os.getcwd()
batch_size = 64
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.1307, ), (0.3081, ))
])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_dataset = datasets.MNIST(root=relative_path + '\pymnist',train=True,download=True,transform=transform)
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
 
test_dataset = datasets.MNIST(root=relative_path + '\pymnist',train=False,download=True,transform=transform)
test_loader = DataLoader(test_dataset,shuffle=False,batch_size=batch_size)

#卷积神经网络模型，卷积、池化、激活，卷积、池化、激活，reshape、全连接输出
class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()   
            self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)#输入图像通道数，卷积输出的通道数，卷积核尺寸，默认stride = 1
            self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)#填充层默认 padding=0
            self.pooling = torch.nn.MaxPool2d(2)
            # batch*20*10*10
            self.fc = torch.nn.Linear(320, 10)
        def forward(self, x):
            # Flatten data from (n, 1, 28, 28) to (n, 784)
            batch_size = x.size(0) #x是4维的，size(0)表示batch_size
            #输入 batch*1*28*28
            x = F.relu(self.pooling(self.conv1(x)))# 输出 batch*10*12*12
            x = F.relu(self.pooling(self.conv2(x)))# batch*20*4*4
            x = x.view(batch_size, -1) # flatten, -1表示自动计算列数，行数固定为batch_size
            # x batch*320            #(x.size(0),-1)将tensor的结构转换为了(batchsize, channels*w*h)，即将（channels，w，h）拉直
            x = self.fc(x)     #最后输出不做激活，在交叉熵损失函数里做了
            return x

model = Net()
model = model.to(device)
model = model.train()

#构建损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#训练
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
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

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            c = (predicted == labels).squeeze().cpu()
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
        """
        print('类别 %d 的准确率: %2d %%' % (i + 1, 100 * class_correct[i] / class_total[i]))
        """
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
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1).cpu()
                for j in range(len(labels)):
                    if labels[j] == i:
                        labelstr.append(1)
                    else:
                        labelstr.append(0)
                    predictedsc.append(probs[j][i])
                  # 取预测为i的概率值
        precision, recall, _ = precision_recall_curve(labelstr, predictedsc, pos_label=1)


         # PR曲线   
        plt.clf()
        plt.plot(precision, recall, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve of CNN for digit ' + str(i))
        plt.legend(loc="lower left")
        if not os.path.exists('PLOTforCNN-MNIST'):
            os.makedirs('PLOTforCNN-MNIST')
        plt.savefig('PLOTforCNN-MNIST/PRclass' + str(i + 1) + '.png')

        # ROC
        fpr, tpr, thresholds = roc_curve(labelstr, predictedsc)
        roc_auc = auc(fpr, tpr)
        plt.clf()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve of CNN for digit ' + str(i))
        plt.legend(loc="lower right")
        plt.savefig('PLOTforCNN-MNIST/ROCclass' + str(i + 1) + '.png')


        precisions[i]=precision[1]
        recalls[i]=recall[1]

        f1_scores[i] = 2 * rec[i] * prec[i] / (rec[i] + prec[i])

        AP = average_precision_score(labelstr, predictedsc)     # 慎用AP,其中存在误差(threshold=0)

        APs[i] = AP
        print('类别 %d 的F1值: %.3f' % (i + 1, f1_scores[i]))
        print('类别 %d 的AP值: %.3f' % (i + 1, AP))
    print('CNN的测试准确率: %.3f' % (correct / total))
    print('所有类别的平均召回率: %.3f' % np.mean(rec))
    print('所有类别的平均精度: %.3f' % np.mean(prec))
    print('所有类别的平均F1值: %.3f' % np.mean(f1_scores))
    print('所有类别的平均AP值: %.3f' % np.mean(APs))


if __name__ == '__main__':
    for epoch in range(2):
        train(epoch)
    test()
    torch.save(model, 'CNNMNISTmodel')