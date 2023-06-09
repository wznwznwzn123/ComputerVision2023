import torch
import torch.nn as nn
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
                input_size = 32,
                hidden_size = 128,
                num_layers = 1,
                batch_first = True,
        )
        self.Out2Class = nn.Linear(128*32,10)
    def forward(self, input):
        output,hn = self.rnn(input,None)
        hidden2one_res = []
        for i in range(32):
            hidden2one_res.append(output[:,i,:])
        hidden2one_res = torch.cat(hidden2one_res,dim=1)
        # print(hidden2one_res.shape)
        res = self.Out2Class(hidden2one_res)
        return res


 
 
model = RNN()
model = model.to(device)
print(model)
 
model = model.train()
 
img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.5, 0.5, 0.5],std = [0.5, 0.5, 0.5])])

dataset_train = datasets.CIFAR10(root = 'D:/桌面/计算机视觉/image identification/cifar10',transform = img_transform,train = True,download = True)
dataset_test = datasets.CIFAR10(root = 'D:/桌面/计算机视觉/image identification/cifar10',transform = img_transform,train = False,download = True)
 
train_loader = torch.utils.data.DataLoader(dataset = dataset_train,batch_size=64,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = dataset_test,batch_size=64,shuffle = False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def Get_ACC():
    correct = 0
    total_num = len(dataset_test)
    for item in test_loader:
        batch_imgs,batch_labels = item
        batch_imgs = batch_imgs.squeeze(1).view(batch_imgs.shape[0], 32*3, 32)
        batch_imgs = Variable(batch_imgs)
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        out = model(batch_imgs)
        _,pred = torch.max(out.data,1)
        correct += torch.sum(pred==batch_labels)
        # print(pred)
        # print(batch_labels)
    correct = correct.data.item()
    acc = correct/total_num
    print('correct={},Test ACC:{:.5}'.format(correct,acc))
 
 
 
optimizer = torch.optim.Adam(model.parameters())
loss_f = nn.CrossEntropyLoss()
 
Get_ACC()
# 开始训练
for epoch in range(1):
    print('epoch:{}'.format(epoch))
    cnt = 0
    for item in train_loader:
        batch_imgs ,batch_labels = item
        batch_imgs = batch_imgs.squeeze(1).view(batch_imgs.shape[0], 32*3, 32)
        # print(batch_imgs.shape)
        batch_imgs,batch_labels = Variable(batch_imgs),Variable(batch_labels)
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        out = model(batch_imgs)
        loss = loss_f(out,batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(cnt%100==0):
            print_loss = loss.data.item()
            print('epoch:{},cnt:{},loss:{}'.format(epoch,cnt,print_loss))
        cnt+=1
    Get_ACC()
 
torch.save(model,'model')

# 测试集
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
class_TP = list(0. for i in range(10))
class_FP = list(0. for i in range(10))
class_FN = list(0. for i in range(10))

with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.squeeze(1).view(images.shape[0], 32*3, 32)
            images = Variable(images).to(device)
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
f1_scores = [0] * 10 # 存储每个类别的F1值
APs = [0] * 10

for i in range(10):
    acc[i] = class_correct[i] / class_total[i]
    rec[i] = class_TP[i] / (class_TP[i] + class_FN[i])
    prec[i] = class_TP[i] / (class_TP[i] + class_FP[i])
    f1_scores[i] = 2 * rec[i] * prec[i] / (rec[i] + prec[i])
    print('类别 ', classes[i], '的准确率: ' ,acc[i])
    print('类别 ', classes[i], '的召回率: ' ,rec[i])
    print('类别 ', classes[i], '的精确率: ' ,prec[i])
    print('类别 ', classes[i], '的F1: ' , f1_scores[i])
    print('类别 ', classes[i], '的TP: ' , class_TP[i])
    print('类别 ', classes[i], '的FP: ' , class_FP[i])
    print('类别 ', classes[i], '的FN: ' , class_FN[i])
    
    print()

for i in range(10):
    labelstr = []
    predictedsc = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.squeeze(1).view(images.shape[0], 32*3, 32)
            images = Variable(images).to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1) 
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
    plt.title('Precision-Recall curve of RNN for ' + classes[i])
    plt.legend(loc="lower left")
    if not os.path.exists('PLOTforRNN-CIFAR10'):
        os.makedirs('PLOTforRNN-CIFAR10')
    plt.savefig('PLOTforRNN-CIFAR10/PRclass' + classes[i] + '.png')

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
    plt.title('ROC curve of RNN for ' + classes[i])
    plt.legend(loc="lower right")
    plt.savefig('PLOTforRNN-CIFAR10/ROCclass' + classes[i] + '.png')


    AP = average_precision_score(labelstr, predictedsc)     
    APs[i] = AP
    print('类别 ', classes[i],' 的AP值: %.3f' % AP)

print('所有类别的平均F1值: %.3f' % np.mean(f1_scores))
print('所有类别的平均AP值: %.3f' % np.mean(APs))