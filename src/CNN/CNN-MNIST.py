import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.utils.data import DataLoader


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 分批次训练，一批 100 个训练数据
BATCH_SIZE = 100
# 所有训练数据训练 3 次
EPOCHS = 3
# 学习率设置为 0.0001
LEARN_RATE = 1e-4
relative_path = os.getcwd()

# 加载数据集
train_data = torchvision.datasets.MNIST(
    root=relative_path + '\pymnist',
    train=True,
    transform=torchvision.transforms.ToTensor() ,# 将下载的文件转换成pytorch认识的tensor类型，且将图片的数值大小从（0-255）归一化到（0-1）
    download=True
)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.MNIST(
    root=relative_path + '\pymnist',
    train=False,
    transform=torchvision.transforms.ToTensor()  # 将下载的文件转换成pytorch认识的tensor类型，且将图片的数值大小从（0-255）归一化到（0-1）
)
 
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1：输入通道为 1，输出通道为 16，卷积核大小 为 5
        # 使用 Relu 激活函数
        # 使用最大值池化
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 卷积层2：输入通道为 16，输出通道为 32，卷积核大小 为 5
        # 使用 Relu 激活函数
        # 使用最大值池化
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # 输出层，全连接层，输入大小 32 * 7 * 7， 输出大小 10
        self.layer_out = nn.Linear(32 * 7 * 7, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        self.out = self.layer_out(x)
        return self.out
    
# 实例化CNN，并将模型放在 GPU 上训练
model = CNN().to(device)
# 使用交叉熵损失，同样，将损失函数放在 GPU 上
loss_fn = nn.CrossEntropyLoss().to(device)
# 使用 Adam 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

for epoch in range(EPOCHS):
    # 加载训练数据
    for step, data in enumerate(train_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        # 调用模型预测
        output = model(x).to(device)
        # 计算损失值
        loss = loss_fn(output, y.long())
        # 输出看一下损失变化
        if step % 1000 == 0:
            print(f'EPOCH({epoch})  step({step})  loss = {loss.item()}')
        # 每一次循环之前，将梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 梯度下降
        optimizer.step()

sum = 0
# test：
for i, data in enumerate(test_loader):
    x, y = data
    x, y = x.to(device), y.to(device)
    # 得到模型预测输出，10个输出，即该图片为每个数字的概率
    res = model(x)
    # 最大概率的就为预测值
    r = torch.argmax(res)
    l = y.item()
    sum += 1 if r == l else 0
    print(f'test({i})     CNN:{r} -- label:{l}')

print('accuracy：', sum / 10000)

 
