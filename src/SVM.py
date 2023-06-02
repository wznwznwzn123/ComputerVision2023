 # 导入必要的库
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# 加载MNIST数据集
digits = datasets.load_digits()

# 获取数据集
X = digits.data
y = digits.target

# 缩小数据集
num_samples = 500
idx = np.random.choice(X.shape[0], num_samples, replace=False)
X = X[idx]
y = y[idx]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建SVM分类器
svm = SVC(kernel='linear', C=1, gamma='auto')

# 训练模型
svm.fit(X_train, y_train)

# 预测测试集
y_pred = svm.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

# 输出准确率
print("Accuracy:", accuracy)
