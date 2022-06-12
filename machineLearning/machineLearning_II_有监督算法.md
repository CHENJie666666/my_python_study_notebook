有监督算法篇



# 感知机

perception

## 1. 原理

### 简介

- 二分类线性分类模型、判别模型

- 神经网络和支持向量机的基础
- 几何解释：用一个超平面将特征空间分为两个部分

### 模型

- 将输入转化为二分类输出

$$
f(x)=sign(wx+b)
$$

- 损失函数

空间中任意一点到超平面 $S$ 的距离为：
$$
\frac{1}{||w||}|wx_0+b|
$$
感知机的损失函数定义为：（分类错误的点到超平面的距离之和）
$$
L(w, b)=\sum_{x_i\in M}y_i(x_i+b)
$$
$M$ 表示误分类的集合

- 优化方法

随机梯度下降法（stochastic gradient descent）

当训练数据线性可分时，感知机是可收敛的



## 2. Numpy算法

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

class Perceptron:
    def __init__(self):
        pass
    
    def sign(self, x, w, b):
        return np.dot(x, w) + b
    
    def train(self, X_train, y_train, learning_rate):
        # 参数初始化
        w, b = self.initilize_with_zeros(X_train.shape[1])
        # 初始化误分类
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for i in range(len(X_train)):
                X = X_train[i]
                y = y_train[i]
                # 如果存在误分类点
                # 更新参数
                # 直到没有误分类点
                if y * self.sign(X, w, b) <= 0:
                    w = w + learning_rate*np.dot(y, X)
                    b = b + learning_rate*y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True
                print('There is no missclassification!')

            # 保存更新后的参数
            params = {'w': w, 'b': b}
        return params
```





# 线性回归

## 多元线性回归模型

### 1. 原理

#### 假设函数

$h_\theta(x)=\theta_0 + \theta_1x$

#### 损失函数

普通最小二乘法

$L(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2$

#### 优化方法

- 梯度下降法

$\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}$

- 正规方程

$\theta=(X^TX)^{-1}X^Ty$

遇到正规方程不可逆情况时可删除一些特征，或采用正则化

- 两种方法比较

梯度下降：需要选择α；需要迭代；适用于特征多情形（$n>10^6$）

正规方法：不需要选择α；不需要迭代；需要计算逆矩阵；适用于特征少情形

### 2. sklearn-API

#### 参数说明

```
sklearn.linear_model.LinearRegression(*, fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, positive=False)
```

| 参数          | 说明                                                         |
| ------------- | ------------------------------------------------------------ |
| fit_intercept | bool型，选择是否需要计算截距，中心化的数据可以选择false      |
| normalize     | bool型，选择是否需要标准化，减去均值再除以L2范式（将被删除） |
| copy_X        | bool型，选择是否复制原数据，如果为false则原数据会因标准化而被覆盖 |
| n_job         | int型，使用进程数                                            |
| positive      | bool型，是否强制系数为正值                                   |

#### 属性

| 属性             | 说明                                    |
| ---------------- | --------------------------------------- |
| coef_            | array，系数                             |
| rank_            | int，矩阵X的秩（仅当X为密集矩阵时可用） |
| singular_        | array，矩阵X的奇异值                    |
| intercept_       | float（0.0）或array，偏置               |
| n_features_in_   | int，输入特征数                         |
| feature_names_in | array，输入特征名称                     |

#### 方法

| 方法                         | 说明                |
| ---------------------------- | ------------------- |
| fit(X, y[, sample_weight])   | 拟合模型            |
| get_params([deep])           | 获取estimator的参数 |
| predict(X)                   | 预测                |
| score(X, y[, sample_weight]) | 返回预测结果的分数  |
| set_params(**params)         | 设置estimator的参数 |

#### 实例

```python
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

diabetes = load_diabetes()
data = diabetes.data
target = diabetes.target 
X, y = shuffle(data, target, random_state=13)
X = X.astype(np.float32)
y = y.reshape((-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
```

### 3. 评估指标

#### MSE

- mean squared error，均方误差

- 计算：$MSE = \frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2$

- sklearn-API

```python
from sklearn.metrics import mean_squared_error as MSE
MSE(yhat,Ytest)
from sklearn.model_selection import cross_val_score
cross_val_score(reg,X,y,cv=10,scoring="neg_mean_squared_error")
```

sklearn中的参数scoring下，均方误差作为评判标准时，是计算”负均方误差“（neg_mean_squared_error），真正的均方误差MSE的数值，其实就是neg_mean_squared_error去掉负号的数字

#### $R^2$

- 相关系数
- 计算：$R^2=1-\frac{\sum_{i=0}^m(y_i-\hat{y_i})^2}{\sum_{i=0}^m(y_i-\overline{y})^2}=1-\frac{RSS}{\sum_{i=0}^m(y_i-\overline{y})^2}$

- sklearn-API

```python
from sklearn.metrics import r2_score
r2_score(Ytest,yhat)
r2_score(y_true = Ytest,y_pred = yhat)
```

#### 可解释性方差分数

- explained_variance_score，EVS

- 计算：$EVS=1-\frac{Var(y_i-\hat{y})}{Var(y_i)}$

- 越接近1越好

### 4. 交叉验证

#### K-fold交叉验证

（待补充）

#### sklearn-API

```python
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

### 交叉验证
def cross_validate(model, x, y, folds=5, repeats=5):
    
    ypred = np.zeros((len(y),repeats))
    score = np.zeros(repeats)
    for r in range(repeats):
        i = 0
        print('Cross Validating', str(r + 1), 'out of', str(repeats))
        x, y = shuffle(x, y, random_state=r)  # shuffle data before each repeat
        kf = KFold(n_splits=folds, random_state=i+1000)  # random split, different each time
        for train_ind, test_ind in kf.split(x):
            print('Fold', i+1, 'out of', folds)
            xtrain, ytrain = x[train_ind,:], y[train_ind]
            xtest, ytest = x[test_ind,:], y[test_ind]
            model.fit(xtrain, ytrain)
            #print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
            ypred[test_ind] = model.predict(xtest)
            i += 1
        score[r] = R2(ypred[:,r], y)
    print('\nOverall R2:', str(score))
    print('Mean:', str(np.mean(score)))
    print('Deviation:', str(np.std(score)))

cross_validate(regr, X, y, folds=5, repeats=5)
```

### 5. Numpy算法

```python
import numpy as np
import pandas as pd

### 初始化模型参数
def initialize_params(dims):
    '''
    输入：dims：训练数据变量维度
    输出：
        w：初始化权重参数值，
        b：初始化偏差参数值
    '''
    # 初始化权重参数为零矩阵
    w = np.zeros((dims, 1))
    # 初始化偏差参数为零
    b = 0
    return w, b

### 定义模型主体部分：包括线性回归公式、均方损失和参数偏导三部分
def linear_loss(X, y, w, b):
    '''
    输入: 
    	X：输入变量矩阵
        y：输出标签向量
        w：变量参数权重矩阵
        b：偏差项
    输出：
        y_hat：线性模型预测输出
        loss：均方损失值
        dw：权重参数一阶偏导
        db：偏差项一阶偏导
    '''
    # 训练样本数量
    num_train = X.shape[0]
    # 训练特征数量
    num_feature = X.shape[1]
    # 线性回归预测输出
    y_hat = np.dot(X, w) + b
    # 计算预测输出与实际标签之间的均方损失
    loss = np.sum((y_hat - y)**2)/num_train
    # 基于均方损失对权重参数的一阶偏导数
    dw = np.dot(X.T, (y_hat - y)) /num_train
    # 基于均方损失对偏差项的一阶偏导数
    db = np.sum((y_hat - y)) / num_train
    return y_hat, loss, dw, db

### 定义线性回归模型训练过程
def linear_train(X, y, learning_rate=0.01, epochs=10000):
    '''
    输入：
        X：输入变量矩阵
        y：输出标签向量
        learning_rate：学习率
        epochs：训练迭代次数
    输出：
        loss_his：每次迭代的均方损失
        params：优化后的参数字典
        grads：优化后的参数梯度字典
    '''
    # 记录训练损失的空列表
    loss_his = []
    # 初始化模型参数
    w, b = initialize_params(X.shape[1])
    # 迭代训练
    for i in range(1, epochs):
        # 计算当前迭代的预测值、损失和梯度
        y_hat, loss, dw, db = linear_loss(X, y, w, b)
        # 基于梯度下降的参数更新
        w += -learning_rate * dw
        b += -learning_rate * db
        # 记录当前迭代的损失
        loss_his.append(loss)
        # 每1000次迭代打印当前损失信息
        if i % 10000 == 0:
            print('epoch %d loss %f' % (i, loss))
        # 将当前迭代步优化后的参数保存到字典
        params = {'w': w, 'b': b}
        # 将当前迭代步的梯度保存到字典
        grads = {'dw': dw, 'db': db}     
    return loss_his, params, grads

from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle

diabetes = load_diabetes()
# 获取输入和标签
data, target = diabetes.data, diabetes.target 
# 打乱数据集
X, y = shuffle(data, target, random_state=13)
# 按照8/2划分训练集和测试集
offset = int(X.shape[0] * 0.8)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
# 将标签改为列向量的形式
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))
# 模型训练
loss_his, params, grads = linear_train(X_train, y_train, 0.01, 200000)

### 定义线性回归预测函数
def predict(X, params):
    '''
    输入：
        X：测试数据集
        params：模型训练参数
    输出：
    	y_pred：模型预测结果
    '''
    # 获取模型参数
    w = params['w']
    b = params['b']
    # 预测
    y_pred = np.dot(X, w) + b
    return y_pred

# 基于测试集的预测
y_pred = predict(X_test, params)

### 定义R2系数函数
def r2_score(y_test, y_pred):
    '''
    输入：
        y_test：测试集标签值
        y_pred：测试集预测值
    输出：
    	r2：R2系数
    '''
    y_avg = np.mean(y_test)
    # 总离差平方和
    ss_tot = np.sum((y_test - y_avg)**2)
    # 残差平方和
    ss_res = np.sum((y_test - y_pred)**2)
    r2 = 1 - (ss_res/ss_tot)
    return r2

print(r2_score(y_test, y_pred))

### 交叉验证
from random import shuffle

def k_fold_cross_validation(items, k, randomize=True):
    if randomize:
        items = list(items)
        shuffle(items)

    slices = [items[i::k] for i in range(k)]

    for i in range(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        training = np.array(training)
        validation = np.array(validation)
        yield training, validation

for training, validation in k_fold_cross_validation(data, 5): 
    X_train = training[:, :10]
    y_train = training[:, -1].reshape((-1,1))
    X_valid = validation[:, :10]
    y_valid = validation[:, -1].reshape((-1,1))
    loss5 = []

    loss, params, grads = linar_train(X_train, y_train, 0.001, 100000)
    loss5.append(loss)
    score = np.mean(loss5)
    print('five kold cross validation score is', score)
    y_pred = predict(X_valid, params)
    valid_score = np.sum(((y_pred-y_valid)**2))/len(X_valid)
    print('valid score is', valid_score)
```





## Lasco 回归

### 1. 原理

对多元线性回归的损失加上L1范式惩罚，通过加入惩罚项，将一些不重要的自变量系数调整为0，从而达到剔除变量的目的

#### 假设函数

$h_\theta(x)=\theta_0 + \theta_1x$

#### 损失函数

普通最小二乘法

$L(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2+\lambda|\theta_1|$

### 2. skleran-API

#### 参数说明

```
class sklearn.linear_model.Lasso(alpha=1.0, *, fit_intercept=True, normalize='deprecated', precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
```

| 参数          | 说明                                                         |
| ------------- | ------------------------------------------------------------ |
| fit_intercept | bool型，选择是否需要计算截距，中心化的数据可以选择false      |
| normalize     | bool型，选择是否需要标准化，减去均值再除以L2范式（将被删除） |
| copy_X        | bool型，选择是否复制原数据，如果为false则原数据会因标准化而被覆盖 |
| positive      | bool型，是否强制系数为正值                                   |
| alpha         | float型，正则化系数，数值越大，则对复杂模型的惩罚力度越大    |
| precompute    | 是否提前计算Gram矩阵来加速计算                               |
| selection     | str型，指定每次迭代时，选择权重向量的哪个分量进行更新<br />"random"：随机选择<br />"cyclic"：循环选择 |

#### 属性

| 属性             | 说明                                         |
| ---------------- | -------------------------------------------- |
| coef_            | array，系数                                  |
| intercept_       | float（0.0）或array，偏置                    |
| n_features_in_   | int，输入特征数                              |
| feature_names_in | array，输入特征名称                          |
| n_iter_          | int或list，迭代次数                          |
| dual_gap_        | float或ndarray，优化结束后的对偶间隙（没懂） |
| sparse_coef_     | array，系数矩阵的稀疏表示                    |

#### 方法

| 方法                                          | 说明                             |
| --------------------------------------------- | -------------------------------- |
| fit(X, y[, sample_weight])                    | 拟合模型                         |
| get_params([deep])                            | 获取estimator的参数              |
| predict(X)                                    | 预测                             |
| score(X, y[, sample_weight])                  | 返回相关系数                     |
| set_params(**params)                          | 设置estimator的参数              |
| path(X, y, *[, l1_ratio, eps, n_alphas, ...]) | 使用坐标下降计算elastic net path |

#### 实例

```python
# 导入线性模型模块
from sklearn import linear_model
# 创建lasso模型实例
sk_lasso = linear_model.Lasso(alpha=0.1)
# 对训练集进行拟合
sk_lasso.fit(X_train, y_train)
# 打印模型相关系数
print("sklearn Lasso intercept :", sk_lasso.intercept_)
print("\nsklearn Lasso coefficients :\n", sk_lasso.coef_)
print("\nsklearn Lasso number of iterations :", sk_lasso.n_iter_)
```

### 3. 评估指标



### 4. Numpy算法

```python
import numpy as np
import pandas as pd

data = np.genfromtxt('example.dat', delimiter = ',')

# 选择特征与标签
x = data[:,0:100] 
y = data[:,100].reshape(-1,1)
# 加一列
X = np.column_stack((np.ones((x.shape[0],1)),x)) # 为什么要加一列

# 划分训练集与测试集
X_train, y_train = X[:70], y[:70]
X_test, y_test = X[70:], y[70:]

# 定义参数初始化函数
def initialize(dims):
    w = np.zeros((dims, 1))
    b = 0
    return w, b

# 定义符号函数
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

# 利用numpy对符号函数进行向量化
vec_sign = np.vectorize(sign)

# 定义lasso损失函数
def l1_loss(X, y, w, b, alpha):
    num_train = X.shape[0]
    num_feature = X.shape[1]
    y_hat = np.dot(X, w) + b
    loss = np.sum((y_hat-y)**2)/num_train + np.sum(alpha*abs(w))
    dw = np.dot(X.T, (y_hat-y)) /num_train + alpha * vec_sign(w)
    db = np.sum((y_hat-y)) /num_train
    return y_hat, loss, dw, db

# 定义训练过程
def lasso_train(X, y, learning_rate=0.01, epochs=300):
    loss_list = []
    w, b = initialize(X.shape[1])
    for i in range(1, epochs):
        y_hat, loss, dw, db = l1_loss(X, y, w, b, 0.1)
        w += -learning_rate * dw
        b += -learning_rate * db
        loss_list.append(loss)
        
        if i % 300 == 0:
            print('epoch %d loss %f' % (i, loss))
        params = {'w': w, 'b': b}
        grads = {'dw': dw, 'db': db}
    return loss, loss_list, params, grads

# 执行训练示例
loss, loss_list, params, grads = lasso_train(X_train, y_train, 0.01, 3000)

# 定义预测函数
def predict(X, params):
    w = params['w']
    b = params['b']
    
    y_pred = np.dot(X, w) + b
    return y_pred

y_pred = predict(X_test, params)

from sklearn.metrics import r2_score
r2_score(y_pred, y_test)
```



## Ridge 回归

### 1. 原理

对多元线性回归的损失加上L2范式惩罚，通过加入惩罚项，将一些不重要的自变量系数调整为接近0

### 2. sklearn-API

#### 参数说明

```
class sklearn.linear_model.Ridge(alpha=1.0, *, fit_intercept=True, normalize='deprecated', copy_X=True, max_iter=None, tol=0.001, solver='auto', positive=False, random_state=None)
```

| 参数          | 说明                                                         |
| ------------- | ------------------------------------------------------------ |
| fit_intercept | bool型，选择是否需要计算截距，中心化的数据可以选择false      |
| normalize     | bool型，选择是否需要标准化，减去均值再除以L2范式（将被删除） |
| copy_X        | bool型，选择是否复制原数据，如果为false则原数据会因标准化而被覆盖 |
| positive      | bool型，是否强制系数为正值                                   |
| alpha         | float型，正则化系数，数值越大，则对复杂模型的惩罚力度越大    |
| solver        | str型，计算求解方法<br />'auto'：根据数据类型自动选择求解器<br />'svd'：利用X的奇异值分解来计算系数<br />'cholesky'：使用scipy.linalg.solve求解<br />'lsqr'：使用专用正规化最小二乘的常规scipy.sparse.linalg.lsqr<br />'sparse_cg'：使用在scipy.sparse.linalg.cg中发现的共轭梯度求解器<br />'sag'：随机平均梯度下降，在大型数据上优化较快<br />'saga'：随机平均梯度下降改进，在大型数据上优化较快<br />'lbfgs'：拟牛顿法 |

#### 属性

| 属性             | 说明                      |
| ---------------- | ------------------------- |
| coef_            | array，系数               |
| intercept_       | float（0.0）或array，偏置 |
| n_features_in_   | int，输入特征数           |
| feature_names_in | array，输入特征名称       |
| n_iter_          | int或list，迭代次数       |

#### 方法

| 方法                         | 说明                |
| ---------------------------- | ------------------- |
| fit(X, y[, sample_weight])   | 拟合模型            |
| get_params([deep])           | 获取estimator的参数 |
| predict(X)                   | 预测                |
| score(X, y[, sample_weight]) | 返回相关系数        |
| set_params(**params)         | 设置estimator的参数 |

#### 实例

```python
# 导入线性模型模块
from sklearn import linear_model
# 创建ridge模型实例
sk_ridge = linear_model.Ridge(alpha=0.1)
# 对训练集进行拟合
sk_ridge.fit(X_train, y_train)
# 打印模型相关系数
print("sklearn Ridge intercept :", sk_ridge.intercept_)
print("\nsklearn Ridge coefficients :\n", sk_ridge.coef_)
print("\nsklearn Ridge number of iterations :", sk_ridge.n_iter_)
```

### 3. 评估指标







### 4. Numpy 算法

```python
import numpy as np
import pandas as pd

data = np.genfromtxt('example.dat', delimiter = ',')
# 选择特征与标签
x = data[:,0:100] 
y = data[:,100].reshape(-1,1)
X = np.column_stack((np.ones((x.shape[0],1)),x))

# 划分训练集与测试集
X_train, y_train = X[:70], y[:70]
X_test, y_test = X[70:], y[70:]

# 定义参数初始化函数
def initialize(dims):
    w = np.zeros((dims, 1))
    b = 0
    return w, b

# 定义ridge损失函数
def l2_loss(X, y, w, b, alpha):
    num_train = X.shape[0]
    num_feature = X.shape[1]
    y_hat = np.dot(X, w) + b
    loss = np.sum((y_hat-y)**2)/num_train + alpha*(np.sum(np.square(w)))
    dw = np.dot(X.T, (y_hat-y)) /num_train + 2*alpha*w
    db = np.sum((y_hat-y)) /num_train
    return y_hat, loss, dw, db

# 定义训练过程
def ridge_train(X, y, learning_rate=0.01, epochs=300):
    loss_list = []
    w, b = initialize(X.shape[1])
    for i in range(1, epochs):
        y_hat, loss, dw, db = l2_loss(X, y, w, b, 0.1)
        w += -learning_rate * dw
        b += -learning_rate * db
        loss_list.append(loss)
        
        if i % 100 == 0:
            print('epoch %d loss %f' % (i, loss))
        params = {'w': w, 'b': b}
        grads = {'dw': dw, 'db': db}
    return loss, loss_list, params, grads

# 执行训练示例
loss, loss_list, params, grads = ridge_train(X_train, y_train, 0.01, 1000)

# 定义预测函数
def predict(X, params):
    w = params['w']
    b = params['b']
    
    y_pred = np.dot(X, w) + b
    return y_pred

y_pred = predict(X_test, params)

from sklearn.metrics import r2_score
r2_score(y_pred, y_test)
```



# 逻辑斯蒂回归

Logistic Regression

## 1. 原理

### 假设函数

$h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-(\theta^Tx)}}$

### 损失函数

$L(\theta)=\frac{1}{m}[\sum_{i=1}^my^{(i)}logh_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)})]$

### 优化过程

- 梯度下降法

$\theta_j:=\theta_j-\alpha\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$

### 特点

- 线性回归+sigmoid()

- 逻辑回归在线性关系的拟合效果非常好，但对非线性关系拟合效果非常差
- 计算速度快（效率高于SVM和RF）
- 在小数据上标线比树模型更好

- 一般不用PCA和SVD降维；统计方法可以用，但没必要



## 2. sklearn-API

### 参数说明

```
class sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
```

| 参数              | 说明                                                         |
| ----------------- | ------------------------------------------------------------ |
| penalty           | str，正则化参数<br />'l1'或'l2'或'none'或'elasticnet'（同时使用L1和L2正则化），数据量较大时倾向于选择L1（提高稀疏性），此时特征选择可用Embedded嵌入法完成 |
| C                 | float，正则化强度的倒数（越小则正则化惩罚越大）              |
| tol               | float，当迭代时误差范围小于tol值时，就停止迭代               |
| solver            | str，优化方法<br />'liblinear'：坐标下降法，二分类，支持L1和L2正则化，对未标准化数据较有效，对小型数据集效果较好<br />'lbfgs'：拟牛顿法，只支持L2正则化，对未标准化数据较有效<br />'newton-cg'：牛顿法，利用损失函数的二阶导数（海森矩阵）矩阵，只支持l2正则化<br />'sag'：随机平均梯度下降，只支持L2正则化，在大型数据上优化较快<br />'sage'：快速梯度下降法，支持三种正则化，在大型数据上优化较快<br />'sag'和'saga'快速收敛的条件是特征具有类似的分布（标准化） |
| fit_intercept     | bool，是否拟合偏置                                           |
| class_weight      | dict（{class_label: weight}）或'balanced'，样本权重          |
| multi_class       | str，分类种类<br />'ovr'：二分类<br />'multinomial'：多分类，solver不能选择'liblinear'<br />'auto'：自动选择 |
| intercept_scaling | float，偏置自适应，solver为'liblinear'时使用                 |
| dual              | bool，当solver为'liblinear'时的L2正则化时使用，当样本数>特征数时，令dual=False |
| warm_start        | bool，当 warm_start 为true时，现有的拟合模型属性用于在后续调用拟合中初始化新模型。<br />当在同一数据集上重复拟合估计器时，但对于多个参数值（例如在网格搜索中找到最大化性能的值），可以重用从先前参数值中学习的模型的各个方面，从而节省时间。 |
| l1_ratio          | float，在0-1之间，仅当penalty='elasticnet'时使用，           |

### 属性

| 属性             | 说明                                                      |
| ---------------- | --------------------------------------------------------- |
| classes_         | ndarray，分类标签                                         |
| coef_            | ndarray，系数                                             |
| intercept_       | float（0.0）或array，偏置                                 |
| n_features_in_   | int，输入特征数                                           |
| feature_names_in | array，输入特征名称                                       |
| n_iter_          | ndarry，迭代次数（当solver为'liblinear'时会返回多个元素） |

### 方法

| 方法                         | 说明                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| fit(X, y[, sample_weight])   | 拟合模型                                                     |
| get_params([deep])           | 获取estimator的参数                                          |
| set_params(**params)         | 设置estimator的参数                                          |
| predict(X)                   | 预测                                                         |
| predict_log_proba(X)         | 预测概率估计的对数                                           |
| predict_proba(X)             | 预测概率估计                                                 |
| score(X, y[, sample_weight]) | 返回预测结果的分数                                           |
| decision_function(X)         | 预测置信度                                                   |
| densify()                    | 将coef_ 矩阵转化为ndarray（在已经稀疏化的模型上使用才有效果） |
| sparsify()                   | 将coef_ 矩阵转换为scipy.sparse矩阵，对于L1正则化模型，它比通常的numpy.ndarray表示更节省内存和存储 |

### 实例

```python
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

data = load_breast_cancer()  # 乳腺癌数据集
x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

# l2正则化用于训练预测
lrl2 = LR(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
lrl2 = lrl2.fit(x_train, y_train)
print(accuracy_score(lrl2.predict(x_test), y_test)) 

# l1正则化用于特征选择
lrl1 = LR(penalty='l1', solver='liblinear', C=0.8, max_iter=1000)
cross_val_score(lrl1, x_test, y_test, cv=10).mean()   # 交叉验证
x_embedded = SelectFromModel(lrl1, threshold=i, norm_order=1).fit_transform(x_test, y_test)    # 使用x_embedded进行特征选择（删除0值）
```



## 3. 评估指标

### 混淆矩阵

<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220515201330249.png" alt="image-20220515201330249" style="zoom:100%;" />

- 准确率Accuracy、精确度（查准率）Precision、召回率（敏感度）Recall、F1分数（在0~1之间）

- python代码

```python
# 精确度
(y[y == clf.predict(x)] == 1).sum() / (clf.predict(x) == 1).sum()
# 召回率
(y[y == clf.predict(x)] == 1).sum() / (y == 1).sum()
```

- sklearn-API

```python
from sklearn import metrics

metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])    # 混淆矩阵（labels中把少数类写在前面）
metrics.accuracy_score      # 准确率
metrics.precision_score     # 精确度
metrics.recall_score        # 召回率
metrics.precision_recall_curve      # 精确度-召回率曲线（不同阈值下的精确率和召回率）
metrics.f1_score            # F1分数
```

### ROC曲线

#### ROC定义

ROC全称是“受试者工作特征”（Receiver Operating Characteristic）

#### ROC计算方法

- 以假阳率（FPR）为横坐标，以真阳率（TPR）为纵坐标

FPR = FP / (FP + TN)  指分类器预测的正类中实际负实例占所有负实例的比例

TPR = TP / (TP + FN)  指分类器预测的正类中实际正实例占所有正实例的比例

- 希望FPR越小越好，TPR越大越好

- ROC曲线

在二分类模型中，最后输出是一个概率值，需要一个阈值，超过这个阈值则归类为1，低于这个阈值就归类为0。所以当阈值从0开始慢慢移动到1的过程，就会形成很多对（FPR, TPR），将它们画在坐标系上即得到ROC曲线
<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220515202316336.png" alt="image-20220515202316336" style="zoom:50%;" />

- AUC

ROC曲线下的面积为AUC（Area Under the Curve），一般AUC在0.5-1之间，AUC越大越好

- python代码

```python
from sklearn.metrics import confusion_matrix as CM
import scikitplot as skplt

# 计算概率
prob = clf.predict_proba(X)

# 计算FPR和TPR
cm = CM(prob.loc[:, 'y_true'], prob.loc[:, 'y_pred'], labels=[1, 0])
frp = cm[1,0] / cm[1,:].sum()
recall = cm[0,0] / cm[0,:].sum()

# 绘图
vali_proba_df = pd.DataFrame(model.predict_proba(vali_x))
skplt.metrics.plot_roc(vali_y, vali_proba_df, plot_micro=False, figsize=(6,6), plot_macro=False)
```

- sklearn-API

```python
from sklearn.metrics import roc_curve, roc_auc_score
roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
# y_score: 置信度分数，或decision_function返回距离
# pos_label：整数或字符串，表示被认为是正样本的类别
# drop_intermediate：若设置为True，则会舍弃一些ROC曲线上不显示的阈值点
fpr, recall, thresholds = roc_curve(y, clf_proba.decision_function(X), pos_label=1)
```

- 利用ROC曲线找出最佳阈值

recall和fpr差距最大的点，约登指数



## 4. 解决样本不平衡问题

通常采用上采样

```python
import imblearn
# imblearn是一个专门处理不平衡数据的库
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
x, y = sm.fit_sample(x, y)
```



## 5. Numpy算法

### 训练和预测

```python
import numpy as np

def sigmoid(x):
    """sigmoid函数"""
    z = 1 / (1 + np.exp(-x))
    return z

def initialize_params(dims):
    """初始化参数"""
    W = np.zeros((dims, 1))
    b = 0
    return W, b

### 定义逻辑回归模型主体
def logistic(X, y, W, b):
    '''
    输入：
        X: 输入特征矩阵
        y: 输出标签向量
        W: 权值参数
        b: 偏置参数
    输出：
        a: 逻辑回归模型输出
        cost: 损失
        dW: 权值梯度
        db: 偏置梯度
    '''
    # 训练样本量和特征数
    num_train = X.shape[0]
    num_feature = X.shape[1]
    # 逻辑回归模型输出
    a = sigmoid(np.dot(X, W) + b)
    # 交叉熵损失
    cost = -1/num_train * np.sum(y*np.log(a) + (1-y)*np.log(1-a))
    # 权值、偏置梯度
    dW = np.dot(X.T, (a-y))/num_train
    db = np.sum(a-y)/num_train
    # 压缩损失数组维度
    cost = np.squeeze(cost) 
    return a, cost, dW, db

### 定义逻辑回归模型训练过程
def logistic_train(X, y, learning_rate, epochs):
    '''
    输入：
        X: 输入特征矩阵
        y: 输出标签向量
        learning_rate: 学习率
        epochs: 训练轮数
    输出：
        cost_list: 损失列表
        params: 模型参数
        grads: 参数梯度
    '''
    # 初始化模型参数
    W, b = initialize_params(X.shape[1])  
    # 初始化损失列表
    cost_list = []  
    
    # 迭代训练
    for i in range(epochs):
        # 计算当前次的模型计算结果、损失和参数梯度
        a, cost, dW, db = logistic(X, y, W, b)    
        # 参数更新
        W = W - learning_rate * dW
        b = b - learning_rate * db        
        # 记录、打印损失
        if i % 100 == 0:
            cost_list.append(cost)   
            print('epoch %d cost %f' % (i, cost)) 
               
    # 保存参数及梯度
    params = {'W': W, 'b': b}        
    grads = {'dW': dW, 'db': db}                
    return cost_list, params, grads

### 定义预测函数
def predict(X, params):
    '''
    输入：
        X: 输入特征矩阵
        params: 训练好的模型参数
    输出：
    	y_prediction: 转换后的模型预测值
    '''
    # 模型预测值
    y_prediction = sigmoid(np.dot(X, params['W']) + params['b'])
    # 基于分类阈值对概率预测值进行类别转换
    for i in range(len(y_prediction)):        
        if y_prediction[i] > 0.5:
            y_prediction[i] = 1
        else:
            y_prediction[i] = 0
            
    return y_prediction

from sklearn.datasets.samples_generator import make_classification
# 生成100*2的模拟二分类数据集
X, labels = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=2)
# 设置随机数种子
rng = np.random.RandomState(2)
# 对生成的特征数据添加一组均匀分布噪声
X += 2 * rng.uniform(size=X.shape)

labels = labels.reshape((-1, 1))
data = np.concatenate((X, labels), axis=1)

# 训练集与测试集的简单划分
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], labels[:offset]
X_test, y_test = X[offset:], labels[offset:]
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

# 训练
cost_list, params, grads = logistic_train(X_train, y_train, 0.01, 1000)

# 预测
y_pred = predict(X_test, params)

# 评估
from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

def accuracy(y_test, y_pred):
    """计算分类准确率"""
    correct_count = 0
    for i in range(len(y_test)):
        for j in range(len(y_pred)):
            if y_test[i] == y_pred[j] and i == j:
                correct_count +=1
            
    accuracy_score = correct_count / len(y_test)
    return accuracy_score

accuracy_score_test = accuracy(y_test, y_prediction)
```

### 绘制决策边界

```python
import matplotlib.pyplot as plt

### 绘制逻辑回归决策边界
def plot_decision boundary(X_train, y_train, params):
    '''
    输入：
        X_train: 训练集输入
        y_train: 训练集标签
        params：训练好的模型参数
    输出：
        决策边界图
    '''
    # 训练样本量
    n = X_train.shape[0]
    # 初始化类别坐标点列表
    xcord1, ycord1, xcord2, ycord2 = [], [], [], []
    # 获取两类坐标点并存入列表
    for i in range(n):
        if y_train[i] == 1:
            xcord1.append(X_train[i][0])
            ycord1.append(X_train[i][1])
        else:
            xcord2.append(X_train[i][0])
            ycord2.append(X_train[i][1])
    # 创建绘图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制两类散点，以不同颜色表示
    ax.scatter(xcord1, ycord1,s=32, c='red')
    ax.scatter(xcord2, ycord2, s=32, c='green')
    # 取值范围
    x = np.arange(-1.5, 3, 0.1)
    # 决策边界公式
    y = (-params['b'] - params['W'][0] * x) / params['W'][1]
    # 绘图
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
plot_logistic(X_train, y_train, params)
```



# 朴素贝叶斯

## 1. 原理

### 简介

#### 贝叶斯公式

根据$P(x, y) = P(x) · P(y|x) = P(y) · P(x|y)$可以得到

$$
P(x|y) = \frac{P(y|x) · P(x)}{P(y)}
$$

#### 朴素——条件独立

假设 $x$, $y$ 相对于 $z$ 是独立的，则有

$$
P(x,y|z) = P(x|z) · P(y|z)
$$

### 优化问题

#### 极大似然估计——后验概率最大化

$$
\mathop{argmax}_{c_k}P(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k)
$$

#### 贝叶斯估计——平滑系数

$$
P_\lambda(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum_{i=1}^NI(x_i^{(j)}=a_{jl},y_i=c_k)+\lambda}{\sum_{i=1}^NI(y_i=c_k)+S_j\lambda}
$$

$\lambda$ 等于1时为拉普拉斯平滑





```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Accuracy of GaussianNB in iris data test:", 
      accuracy_score(y_test, y_pred))
```



## numpy实现

```python
import numpy as  np
import pandas as pd

class Naive_Bayes:
    def __init__(self):
        pass

    # 朴素贝叶斯训练过程
    def nb_fit(self, X, y):
        classes = y[y.columns[0]].unique()
        class_count = y[y.columns[0]].value_counts()
        # 类先验概率
        class_prior = class_count / len(y)
        # 计算类条件概率
        prior = dict()
        for col in X.columns:
            for j in classes:
                p_x_y = X[(y == j).values][col].value_counts()
                for i in p_x_y.index:
                    prior[(col, i, j)] = p_x_y[i] / class_count[j]

        return classes, class_prior, prior

    # 预测新的实例
    def predict(self, X_test):
        res = []
        for c in classes:
            p_y = class_prior[c]
            p_x_y = 1
            for i in X_test.items():
                p_x_y *= prior[tuple(list(i) + [c])]
            res.append(p_y * p_x_y)
        return classes[np.argmax(res)]


if __name__ == "__main__":
    x1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    x2 = ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']
    y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    X = df[['x1', 'x2']]
    y = df[['y']]
    X_test = {'x1': 2, 'x2': 'S'}

    nb = Naive_Bayes()
    classes, class_prior, prior = nb.nb_fit(X, y)
    print('测试数据预测类别为：', nb.predict(X_test))
```





# k-近邻算法

K-Nearest Neighbors, KNN

## 1. 原理

### 1.1 简介

对于给定的实例数据和实例数据对应所属类别，当要对新的实例进行分类时，根据这个实例最近的 k 个实例所属的类别来决定其属于哪一类。

关键：距离度量、k值选取、归类规则

### 1.2 算法流程

1. 根据给定的距离度量，在训练集中找出与 $x$ 最近的 $k$ 个点
2. 在这些点中，根据分类决策规则（如多数表决）决定 $x$ 的类别 $y$

### 1.3 三要素

#### 距离度量

- 闵可夫斯基距离（Minkowski Distance）

$x_i,x_j$ 之间的 $L_p$ 距离为
$$
L_p(x_i,x_j)=(\sum_{l=1}^n|x_i^{(l)}-x_j^{(l)}|^p)^{\frac{1}{p}} \ \ \ p\geqslant1
$$
$p=1$ 时，即为曼哈顿距离（Manhattan distance）

$p=2$ 时，即为欧式距离（Euckidean distance）

- Python实现

```python
def MinkowskiDistance(x, y, p):
    import math
    import numpy as np
    zipped_coordinate = zip(x, y)
    return math.pow(np.sum([math.pow(np.abs(i[0]-i[1]), p) for i in zipped_coordinate]), 1/p)
```

#### k值选择

- k值的影响

较小的k值会降低近似误差，但会增加估计误差，使得预测结果对近邻点较敏感，易发生过拟合

较大的k值会减小估计误差，但会增加近似误差，较远的实例也会对预测起作用

k值过小，易受到异常值影响；k值过大，易受到样本不均衡影响

- 通常采用交叉验证来选取最优k值

#### 分类决策规则

- 多数表决

### 1.4 优化策略

线性扫描方法需要遍历所有数据，当数据集较大时非常耗时

构造kd树：不断用垂直于坐标轴的超平面将k维空间切分，构成一系列的k维超矩形区域

通常选择训练实例点在选定坐标轴上的中位数为切分点，得到平衡kd树



## 2. sklearn-API





### 示例

```python
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
y_pred = y_pred.reshape((-1, 1))
# 计算准确率
num_correct = np.sum(y_pred == y_test)
accuracy = float(num_correct) / X_test.shape[0]
print('Got %d / %d correct => accuracy: %f' % (num_correct, X_test.shape[0], accuracy))
```



## Numpy实现

```python
import numpy as np
from collections import Counter
import random
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle

class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y    
    
    def compute_distances(self, X):
        ### 定义欧氏距离
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 

        M = np.dot(X, self.X_train.T)
        te = np.square(X).sum(axis=1)
        tr = np.square(self.X_train).sum(axis=1)
        dists = np.sqrt(-2 * M + tr + np.matrix(te).T)        
        return dists    
        
    def predict_labels(self, dists, k=1):
        ### 定义预测函数
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)         
        for i in range(num_test):
            closest_y = []
            labels = self.y_train[np.argsort(dists[i, :])].flatten()
            closest_y = labels[0:k]

            c = Counter(closest_y)
            y_pred[i] = c.most_common(1)[0][0]        
        return y_pred    
        
    def cross_validation(self, X_train, y_train):
        ### 5折交叉验证
        num_folds = 5
        k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

        X_train_folds = []
        y_train_folds = []

        X_train_folds = np.array_split(X_train, num_folds)
        y_train_folds = np.array_split(y_train, num_folds)

        k_to_accuracies = {}        
        for k in k_choices:            
            for fold in range(num_folds): 
                validation_X_test = X_train_folds[fold]
                validation_y_test = y_train_folds[fold]
                temp_X_train = np.concatenate(X_train_folds[:fold] + X_train_folds[fold + 1:])
                temp_y_train = np.concatenate(y_train_folds[:fold] + y_train_folds[fold + 1:])


                self.train(temp_X_train, temp_y_train )

                temp_dists = self.compute_distances(validation_X_test)
                temp_y_test_pred = self.predict_labels(temp_dists, k=k)
                temp_y_test_pred = temp_y_test_pred.reshape((-1, 1))                #Checking accuracies
                num_correct = np.sum(temp_y_test_pred == validation_y_test)
                num_test = validation_X_test.shape[0]
                accuracy = float(num_correct) / num_test
                k_to_accuracies[k] = k_to_accuracies.get(k,[]) + [accuracy]        # Print out the computed accuracies
        
        for k in sorted(k_to_accuracies):            
            for accuracy in k_to_accuracies[k]:
                print('k = %d, accuracy = %f' % (k, accuracy))

        accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
        best_k = k_choices[np.argmax(accuracies_mean)]
        print('最佳k值为{}'.format(best_k))        
        
        return best_k    
        
    def create_train_test(self):
        X, y = shuffle(iris.data, iris.target, random_state=13)
        X = X.astype(np.float32)
        y = y.reshape((-1,1))
        offset = int(X.shape[0] * 0.7)
        X_train, y_train = X[:offset], y[:offset]
        X_test, y_test = X[offset:], y[offset:]
        y_train = y_train.reshape((-1,1))
        y_test = y_test.reshape((-1,1))        
        return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    knn_classifier = KNearestNeighbor()
    X_train, y_train, X_test, y_test = knn_classifier.create_train_test()
    best_k = knn_classifier.cross_validation(X_train, y_train)
    dists = knn_classifier.compute_distances(X_test)
    y_test_pred = knn_classifier.predict_labels(dists, k=best_k)
    y_test_pred = y_test_pred.reshape((-1, 1))
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / X_test.shape[0]
    print('Got %d / %d correct => accuracy: %f' % (num_correct, X_test.shape[0], accuracy))
```



## 优缺点

### 优点

简单，无需训练，易于实现 

### 缺点

计算量大；k值不当则分类精度无法保证





# 决策树

decision tree, DT

## 1. 原理

### 简介

- 一种分类的树形结构（也可以用作回归），在分类中，基于特征对实例进行分类，可以看做是if-then规则的集合，也可以认为是定义在特征空间与类空间上的条件概率分布
- 决策树由结点和有向边组成。结点又分为内部结点（表示特征或属性）和叶结点（表示一个类）。决策树从根结点开始对实例某一特征进行分类，并将实例分配到其子结点上，不断递归直至到达子结点。
- 主要包括三个步骤：特征选择、决策树生成、决策树修剪
- 决策树学习的思想：ID3算法、C4.5算法、CART算法

- 决策树学习主要流程

开始，构建根节点，将所有训练数据放在根节点，选择一个最优特征，按照这一特征将训练集分割为子集，使得各个子集有一个在当前条件下最好的分类。如果这些子集已经能够被基本正确分类，那么构建叶结点，并将这些子集分到所对应的叶结点中，若不能正确分类，则重新选择分类特征。如此递归执行，直至训练集子集被基本正确分类，或者没有合适的特征为止。

### ID3算法

利用信息增益来学习决策树

#### 信息熵 information entropy

设$X$是一个离散随机变量，其概率分布为
$$
P(X=x_i)=p_i, \ \ i=1,2,...,n
$$
则其熵表示为
$$
H(p)=-\sum_{i=1}^np_ilog_2p_i
$$
熵的单位是比特（以2为底）或纳特（以e为底），熵越大，随机变量的不确定性就越大

#### 信息增益 information gain

表示得知特征X的信息而使得类Y的信息的不确定性减少的程度，也叫互信息
$$
g(Y, X)=H(Y)-H(Y|X)=H(Y)-\sum_{i=1}^np_iH(Y|X=x_i)
$$
信息增益越大，则该特征对数据集确定性贡献越大，表示该特征对数据有较强的分类能力。

#### 学习过程

从根结点开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征，由该特征的不同取值建立子结点；在递归调用以上方法；直到所有特征的信息增益均很小或没有特征可以选择为止

#### numpy实现

```python
def entropy(ele):    
    '''
    计算信息熵
    input: A list contain categorical value.
    output: Entropy value.
    '''
    # Calculating the probability distribution of list value
    probs = [ele.count(i)/len(ele) for i in set(ele)]    
    # Calculating entropy value
    entropy = -sum([prob*log(prob, 2) for prob in probs])    
    return entropy

def split_dataframe(data, col):    
    '''
    根据特征和特征值进行数据划分
    input: dataframe, column name.
    output: a dict of splited dataframe.
    '''
    # unique value of column
    unique_values = data[col].unique()    
    # empty dict of dataframe
    result_dict = {elem : pd.DataFrame for elem in unique_values}    
    # split dataframe based on column value
    for key in result_dict.keys():
        result_dict[key] = data[:][data[col] == key]    
    return result_dict

def choose_best_col(df, label):    
    '''
    根据信息增益确定最优特征
    input: datafram, label
    output: max infomation gain, best column, 
            splited dataframe dict based on best column.
    '''
    # Calculating label's entropy
    entropy_D = entropy(df[label].tolist())    
    # columns list except label
    cols = [col for col in df.columns if col not in [label]]    
    # initialize the max infomation gain, best column and best splited dict
    max_value, best_col = -999, None
    max_splited = None
    # split data based on different column
    for col in cols:
        splited_set = split_dataframe(df, col)
        entropy_DA = 0
        for subset_col, subset in splited_set.items():            
            # calculating splited dataframe label's entropy
            entropy_Di = entropy(subset[label].tolist())            
            # calculating entropy of current feature
            entropy_DA += len(subset)/len(df) * entropy_Di        
        # calculating infomation gain of current feature
        info_gain = entropy_D - entropy_DA        
        if info_gain > max_value:
            max_value, best_col = info_gain, col
            max_splited = splited_set    
        return max_value, best_col, max_splited
    
class ID3Tree:    
    # define a Node class
    class Node:        
        def __init__(self, name):
            self.name = name
            self.connections = {}    
            
        def connect(self, label, node):
            self.connections[label] = node    
        
    def __init__(self, data, label):
        self.columns = data.columns
        self.data = data
        self.label = label
        self.root = self.Node("Root")    
    
    # print tree method
    def print_tree(self, node, tabs):
        print(tabs + node.name)        
        for connection, child_node in node.connections.items():
            print(tabs + "\t" + "(" + connection + ")")
            self.print_tree(child_node, tabs + "\t\t")    
    
    def construct_tree(self):
        self.construct(self.root, "", self.data, self.columns)    
    
    # construct tree
    def construct(self, parent_node, parent_connection_label, input_data, columns):
        max_value, best_col, max_splited = choose_best_col(input_data[columns], self.label)        
        if not best_col:
            node = self.Node(input_data[self.label].iloc[0])
            parent_node.connect(parent_connection_label, node)            
        return

        node = self.Node(best_col)
        parent_node.connect(parent_connection_label, node)

        new_columns = [col for col in columns if col != best_col]        
        # Recursively constructing decision trees
        for splited_value, splited_data in max_splited.items():
            self.construct(node, splited_value, splited_data, new_columns)
```



### C4.5算法

#### 信息增益比

$$
g_R(Y,X)=\frac{g(Y,X)}{H_X(Y)}, H_X(Y)=-\sum_{i=1}^n\frac{|Y_i|}{|Y|}log_2\frac{|Y_i|}{Y}
$$

#### 学习过程

- 与ID3法类似，不同的是采用信息增益比替代信息增益选择特征

#### 剪枝

- 生成的决策树对训练数据拟合效果好，但容易过拟合，需要剪枝（从已生成的树上裁减一些叶结点）

- 剪枝算法之一 —— 极小化决策树损失函数（类似正则化）

树 $T$ 的叶结点个数为 $|T|$，在叶结点 $t$ 处有 $N_t$ 个样本，其中 $k$ 类样本数有 $N_{tk}$ 个

决策树的损失函数可以定义为：
$$
C_\alpha(T)=\sum_{i=1}^{|T|}N_tH_t(T) + \alpha|T|,其中经验熵H_t(T)=-\sum_k\frac{N_tk}{N_t}log\frac{N_{tk}}{N_k}
$$
剪枝算法：计算每个叶结点回缩至父结点前后的损失，若剪枝后损失未减小则不剪枝，不断循环至不能再剪枝为止

### CART算法

classification and regression tree, CART

#### 简介

- CART是在给定输入随机变量X条件下输出随机变量Y的条件概率分布的学习方法

- 假设决策树是二叉树
- 算法流程：基于训练集生成决策树，用验证集剪枝（以损失函数最小为剪枝原则）

#### 回归树

- 回归树模型

将输入空间划分为 $M$ 个单元 ，且每个单元 $R_m$ 上有固定输出值 $c_m$
$$
f(x)=\sum_{m=1}^Mc_mI(x\in R_m)
$$

- 损失：空间划分的误差采用平方误差来衡量（最小二乘回归树）

$$
\sum_{x_i\in R_m}(y_i-f(x_i))^2
$$

#### 分类树

- 基尼系数

假设有 $K$ 类，样本属于第 $k$ 类的概率为 $p_k$，则概率分布的基尼系数定义为
$$
Gini(p)=\sum_{k=1}^Kp_k(1-p_k)=1-\sum_{k=1}^Kp_k^2
$$
基尼系数表示集合的不确定度，其值越大，则不确定度越大

- 分类树采用基尼系数选择特征，选择基尼系数最小的特征作为切分点

#### 剪枝

与ID3和C4.5的剪枝算法类似，不同的是损失采用最小二乘或基尼系数进行衡量

#### numpy实现

```python
def gini(nums):
    """计算基尼系数"""
    probs = [nums.count(i)/len(nums) for i in set(nums)]
    gini = sum([p*(1-p) for p in probs]) 
    return gini

### 定义二叉特征分裂函数
def feature_split(X, feature_i, threshold):
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    X_left = np.array([sample for sample in X if split_func(sample)])
    X_right = np.array([sample for sample in X if not split_func(sample)])
    return np.array([X_left, X_right])

### 定义树结点
class TreeNode():
    def __init__(self, feature_i=None, threshold=None,
                 leaf_value=None, left_branch=None, right_branch=None):
        # 特征索引
        self.feature_i = feature_i          
        # 特征划分阈值
        self.threshold = threshold 
        # 叶子节点取值
        self.leaf_value = leaf_value   
        # 左子树
        self.left_branch = left_branch     
        # 右子树
        self.right_branch = right_branch 

### 定义二叉决策树
class BinaryDecisionTree(object):
    ### 决策树初始参数
    def __init__(self, min_samples_split=2, min_gini_impurity=999,
                 max_depth=float("inf"), loss=None):
        # 根结点
        self.root = None  
        # 节点最小分裂样本数
        self.min_samples_split = min_samples_split
        # 节点初始化基尼不纯度
        self.mini_gini_impurity = min_gini_impurity
        # 树最大深度
        self.max_depth = max_depth
        # 基尼不纯度计算函数
        self.gini_impurity_calculation = None
        # 叶子节点值预测函数
        self._leaf_value_calculation = None
        # 损失函数
        self.loss = loss

    ### 决策树拟合函数
    def fit(self, X, y, loss=None):
        # 递归构建决策树
        self.root = self._build_tree(X, y)
        self.loss=None

    ### 决策树构建函数
    def _build_tree(self, X, y, current_depth=0):
        # 初始化最小基尼不纯度
        init_gini_impurity = 999
        # 初始化最佳特征索引和阈值
        best_criteria = None    
        # 初始化数据子集
        best_sets = None        

        # 合并输入和标签
        Xy = np.concatenate((X, y), axis=1)
        # 获取样本数和特征数
        n_samples, n_features = X.shape
        # 设定决策树构建条件
        # 训练样本数量大于节点最小分裂样本数且当前树深度小于最大深度
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # 遍历计算每个特征的基尼不纯度
            for feature_i in range(n_features):
                # 获取第i特征的所有取值
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                # 获取第i个特征的唯一取值
                unique_values = np.unique(feature_values)

                # 遍历取值并寻找最佳特征分裂阈值
                for threshold in unique_values:
                    # 特征节点二叉分裂
                    Xy1, Xy2 = feature_split(Xy, feature_i, threshold)
                    # 如果分裂后的子集大小都不为0
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # 获取两个子集的标签值
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # 计算基尼不纯度
                        impurity = self.impurity_calculation(y, y1, y2)

                        # 获取最小基尼不纯度
                        # 最佳特征索引和分裂阈值
                        if impurity < init_gini_impurity:
                            init_gini_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],   
                                "lefty": Xy1[:, n_features:],   
                                "rightX": Xy2[:, :n_features],  
                                "righty": Xy2[:, n_features:]   
                                }
        
        # 如果计算的最小不纯度小于设定的最小不纯度
        if init_gini_impurity < self.mini_gini_impurity:
            # 分别构建左右子树
            left_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            right_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return TreeNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                "threshold"], left_branch=left_branch, right_branch=right_branch)

        # 计算叶子计算取值
        leaf_value = self._leaf_value_calculation(y)

        return TreeNode(leaf_value=leaf_value)

    ### 定义二叉树值预测函数
    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root

        # 如果叶子节点已有值，则直接返回已有值
        if tree.leaf_value is not None:
            return tree.leaf_value

        # 选择特征并获取特征值
        feature_value = x[tree.feature_i]

        # 判断落入左子树还是右子树
        branch = tree.right_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.left_branch
        elif feature_value == tree.threshold:
            branch = tree.left_branch

        # 测试子集
        return self.predict_value(x, branch)

    ### 数据集预测函数
    def predict(self, X):
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

### CART回归树
class RegressionTree(BinaryDecisionTree):
    def _calculate_variance_reduction(self, y, y1, y2):
        var_tot = np.var(y, axis=0)
        var_y1 = np.var(y1, axis=0)
        var_y2 = np.var(y2, axis=0)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        # 计算方差减少量
        variance_reduction = var_tot - (frac_1 * var_y1 + frac_2 * var_y2)
        
        return sum(variance_reduction)

    # 节点值取平均
    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        self.impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)    

### CART决策树
class ClassificationTree(BinaryDecisionTree):
    ### 定义基尼不纯度计算过程
    def _calculate_gini_impurity(self, y, y1, y2):
        p = len(y1) / len(y)
        gini = calculate_gini(y)
        gini_impurity = p * calculate_gini(y1) + (1-p) * calculate_gini(y2)
        return gini_impurity
    
    ### 多数投票
    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # 统计多数
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common
    
    # 分类树拟合
    def fit(self, X, y):
        self.impurity_calculation = self._calculate_gini_impurity
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)        
    
from sklearn import datasets
data = datasets.load_iris()
X, y = data.data, data.target
y = y.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1,1), test_size=0.3)
clf = ClassificationTree()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
```



### 示例

#### ID3算法

```python
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

iris = load_iris()
# criterion选择entropy，这里表示选择ID3算法
clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best')
clf = clf.fit(iris.data, iris.target)
# score = clf.score(X_test, y_test)  # 返回预测的准确度

dot_data = tree.export_graphviz(clf, out_file=None,
                               feature_names=iris.feature_names,
                               class_names=iris.target_names,
                               filled=True, 
                               rounded=True,
                               special_characters=True)
graph = graphviz.Source(dot_data)
```

#### CART树

```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```



## 分类树  DecisionTreeClassifier

### 参数

- criterion  计算不纯度方式

1. entropy  信息熵（计算更慢一些，但对不纯度更敏感，对训练集的拟合更好，但容易过拟合）
2. gini  基尼系数（默认选项；纬度高，噪音大时使用）

- random_state  分枝中随机模式的参数，默认为None
- splitter  控制DT中的随机选项

1. best  优先选择更重要的特征进行分枝
2. random  分枝时更加随机，防止过拟合

- max_depth  限制树的最大深度（高维度低样本量时有效；可从3开始尝试）
- min_samples_leaf   一个节点分枝后的子节点必须至少包含xxx个训练样本，这个节点才被允许分枝（可从5开始尝试；设置太小容易过拟合；也可输入浮点数表示百分比）
- min_samples_split  一个节点必须至少包含xxx个训练样本，这个节点才被允许分枝
- class_weight & min_weight_fraction_leaf  给样本标签一定权重（此时剪枝需要搭配min_weight_fraction_leaf使用）

### 属性

- classes_    输出所有标签
- feature_importances_    特征重要性
- max_feature_    参数max_feature的推断值
- n_classes_    标签类别的数据
- n_features_    训练时使用的特征个数
- n_outputs_    训练时输出的结果个数
- tree_    可以访问树的结构和低级属性

### 接口

- clf.apply(X_test)    返回测试样本所在的叶子节点的索引
- clf.predict(X_test)    返回测试样本的分类/回归结果
- clf.fit(x_train, y_train)
- clf.score(X_test, y_test)

不接受一维矩阵作为特征输入，必须转化为二维矩阵（reshape(-1, 1)）

### 画DT树

```
from sklearn import tree
import graphvize

feature_names = []  # 定义特征名
label_names = []  # 标签名
dot_data = tree.export_graphviz(clf, feature_names, class_names, filled=True, rounded=True)  # 定义DT
graph = graphvize.Source(dot_data)  # 绘图
```

filled和rounded定义了框的填充颜色和形状

### 查看DT树属性

```
clf.feature_importance_
[*zip(feature_names, clf.feature_importance_)]
```

### 其他

DT对环形数据的分类效果较差

## 回归树  DecisionTreeRegressor

### 参数

- criterion  衡量分枝质量的指标

1. mse    均方误差（也是最常用的回归树回归质量的指标）
2. fredman_mse    费尔德曼均方误差
3. mae    绝对平均误差

### 属性&接口

和分类树一致

### 交叉验证

```
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
regressor = DecisionTreeRegressor(random_state=0)
cross_val_score(regressor, x_data, y_data, cv=10, scoring='neg_mean_squared_error')
```

不需要额外划分数据集和测试集

### 网格搜索

```
from sklearn.model_selection import GridSearchCV
parameters = {
    'criterion': ('gini', 'entropy'),
    'splitter': ('best', 'random'),
    'max_depth': [*range(1, 10)],
    'min_samples_leaf': [*range(1, 50, 5)],
    'min_impurity_decrease': np.linspace(0, 0.5, 50)
}
GS = GridSearchCV(clf, parameters, cv=10)
GS = GS.fit(x_train, y_train)
```

计算量大！

网格参数的属性

- GS.best_parms_    返回参数的最佳组合
- GS.best_score_    模型的评估指标





## 优缺点

### 优点

- 模型具有可读性
- 分类速度快

### 缺点



# AdaBoost算法

## 1. 原理

#### 提升方法

以分类问题为例，先训练一个弱分类器，再通过改变训练数据的权值分布学习一系列弱分类器，再将这些弱分类器进行组合得到一个强分类器

#### AdaBoost算法

- 步骤

在均匀的权值分布训练样本上训练一个基分类器 $G_m{x}$

计算 $G_m{x}$ 在训练数据上的分类误差率
$$
e_m=\sum_{i=1}^NP(G_m(x_i)\neq y_i)=\sum_{i=1}^Nw_{mi}I(G_m(x_i)\neq y_i)
$$
计算 $G_m(x)$ 系数
$$
\alpha_m=\frac{1}{2}ln\frac{1-e_m}{e_m}
$$
更新训练数据集的权值分布
$$
w_{m+1,i}=\frac{w_m,i}{\sum_{i=1}^Nexp(-\alpha_my_iG_m(x_i))}exp(-\alpha_my_iGm(x_i)), i=1,2,...,N
$$
构建基本分类器的线性组合
$$
G(x)=sign(\sum_{m=1}^M\alpha_mG_m(x))
$$

- 解释

AdaBoost 算法还被认为是一种模型为加法模型、损失函数为指数函数、学习算法为前向分布算法时的二分类学习方法

- 提升树

以分类树或回归树为基本分类器的提升方法



## 2. sklearn-API





### 示例

```python
from sklearn.ensemble import AdaBoostClassifier
clf_ = AdaBoostClassifier(n_estimators=5, random_state=0)
# train
clf_.fit(X_train, y_train)
# valid
y_pred_ = clf_.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_)
print ("Accuracy:", accuracy)
```





## Numpy 代码实现

```python
class DecisionStump():
    """定义一个决策树桩，本质上就是一个带有阈值划分的决策树结点"""
    def __init__(self):
        # 基于划分阈值决定样本分类为1还是-1
        self.polarity = 1
        # 特征索引
        self.feature_index = None
        # 特征划分阈值
        self.threshold = None
        # 指示分类准确率的值
        self.alpha = None
        
class Adaboost():
    # 弱分类器个数
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
    # Adaboost拟合算法
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # (1) 初始化权重分布为均匀分布 1/N
        w = np.full(n_samples, (1/n_samples))
        
        self.estimators = []
        # (2) for m in (1,2,...,M)
        for _ in range(self.n_estimators):
            # (2.a) 训练一个弱分类器：决策树桩
            clf = DecisionStump()
            # 设定一个最小化误差
            min_error = float('inf')
            # 遍历数据集特征，根据最小分类误差率选择最优划分特征
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                # 尝试将每一个特征值作为分类阈值
                for threshold in unique_values:
                    p = 1
                    # 初始化所有预测值为1
                    prediction = np.ones(np.shape(y))
                    # 小于分类阈值的预测值为-1
                    prediction[X[:, feature_i] < threshold] = -1
                    # 2.b 计算误差率
                    error = sum(w[y != prediction])
                    
                    # 如果分类误差大于0.5，则进行正负预测翻转
                    # E.g error = 0.8 => (1 - error) = 0.2
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # 一旦获得最小误差则保存相关参数配置
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error
                        
            # 2.c 计算基分类器的权重
            clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            # 初始化所有预测值为1
            predictions = np.ones(np.shape(y))
            # 获取所有小于阈值的负类索引
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            # 将负类设为 '-1'
            predictions[negative_idx] = -1
            # 2.d 更新样本权重
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            # 保存该弱分类器
            self.estimators.append(clf)
    
    # 定义预测函数
    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))
        # 计算每个弱分类器的预测值
        for clf in self.estimators:
            # 初始化所有预测值为1
            predictions = np.ones(np.shape(y_pred))
            # 获取所有小于阈值的负类索引
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            # 将负类设为 '-1'
            predictions[negative_idx] = -1
            # 2.e 对每个弱分类器的预测结果进行加权
            y_pred += clf.alpha * predictions

        # 返回最终预测结果
        y_pred = np.sign(y_pred).flatten()
        return y_pred
```





随机森林 Random Forest

集成模型

三类集成算法：bagging (RF), boosting (Adaboost, GBDT), stacking

bagging    模型独立，相互平行

boosting    按顺序逐步提升特征权重

sklearn.ensemble模块

随机森林分类器  RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

单个树的错误率不能超过50%

参数

n_estimators    基评估器的数量（一般在0~200之内），一般随着该值的升高，准确度迅速升高，然后持平（该值太高时计算量较大）

random_state    生成森林的随机性参数

bootstrap    默认为True，表示有放回随机抽样

oob_score    袋外数据（没有被抽中的数据），可用来测试（即不需划分训练集和测试集），设置为True

其他参数与DecisionTreeClassifier一致

API

from sklearn.ensemble import RandomForestClassifier rfc = RandomForestClassifier() model = rfc.fit(x_train, y_train) score = model.score(x_test, y_test)

Copy

Caption



Plain Text

接口：apply, fit, predict, score

属性

rfc.estimators_    查看森林中树的状况

rfc.oob_score_    查看袋外数据的测试结果

随机森林回归器  RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

可以用于填补缺失值

遍历所有特征，从缺失值最少的特征开始填充，此时其他缺失特征用0填充

from sklearn.impute import SimpleImputer import numpy as np import pandas as pd from sklearn.ensemble import RandomForestRegressor # 返回缺失值数量从小到大排序所对应的索引 sortindex = np.argsort(X_missing.isnull().sum(axis=0)).values for i in sortindex:    df = X_missing.copy()    # 构造新标签    fillc = df.iloc[:, i]    # 构造新特征矩阵    df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_full)], axis=1)    # 缺失值零填充    df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)     # 选择训练集和测试集    y_train = fillc[fillc.notnull()]    y_test = fillc[fillc.isnull()]    x_train = df_0[y_train.index, :]    x_test = df_0[y_test.index, :]     # 用随机森林填补缺失值    rfc = RandomForestRegressor(n_estimators=100)    rfc = rfc.fit(x_train, y_train)    y_predict = rfc.predict(x_test)     # 将填补好的值返回到原始的特征矩阵中    X_missing.loc[X_missing.iloc[:, i].isnull(), i] = y_predict

Copy

Caption



Plain Text



# 梯度提升决策树

Gradient Boosting Decision Tree, GBDT

## 1. 原理

### 简介

- GBDT 是由决策树、提升模型和梯度下降一起构成的
- GBDT 中使用 CART 作为弱分类器

### 流程

- 初始化弱分类器

$$
f_0(x)=argmin_c\sum_{i=1}^NL(y_i,c)
$$

- 计算样本的负梯度（残差）

$$
r_{im}=-[\frac{\partial L(y_i,f(x_i))}{\partial f(x_i)}]_{f(x)=f_{m-1}(x)}
$$

- 将残差与数据作为下棵树的训练数据，重复以上过程，得到最终学习器





# 支持向量机

## 1. 原理

### 简介

- support vector machines, SVM

特征空间上的间隔最大的线性分类器，通过核技巧也可用于非线性分类

学习策略为间隔最大化——求解凸二次规划问题（convex quadratic programming）

- 种类

线性可分支持向量机——硬间隔最大化

线性支持向量机——软间隔最大化

非线性支持向量机——核方法

### 线性可分支持向量机

#### 分类决策函数

$$
f(x)=sign(w^*x+b^*)
$$

找到两类数据正确划分并且间隔最大的超平面

#### 函数间隔与几何间隔

在超平面确定前提下，$|wx+b|$ 能够表示点 $x$ 距离超平面远近，$wx+b$ 的符号则表示是否分类正确

超平面关于点 $x$ 的函数间隔（function margin）定义为：
$$
\hat\gamma_i = y_i(wx_i+b)
$$
超平面关于训练集的函数间隔定义为：
$$
\hat\gamma = \mathop{min}_{i=1,...,N}\hat\gamma_i
$$
几何间隔则表示为：（实例点到超平面的带符号的距离的最小值）
$$
\hat\gamma = min(y_i(\frac{w}{||w||}x_i+\frac{b}{||w||}))
$$

#### 最优化问题

最大化 $\hat\gamma$，转变为：
$$
\mathop{min}_{w,b}\ \frac{1}{2}||w||^2  \\
s.t. \ y_i(wx_i+b) -1 \geqslant 0, i=1,2,...,N
$$

#### 支持向量与间隔

- 支持向量：两类数据中距离决策边界最近的点
- 间隔 margin：两个 支持向量所在超平面间的距离

<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220611222439615.png" alt="image-20220611222439615" style="zoom:50%;" />

#### 算法实现

```python
from cvxopt import matrix, solvers
### 实现线性可分支持向量机
### 硬间隔最大化策略
class Hard_Margin_SVM:
    ### 线性可分支持向量机拟合方法
    def fit(self, X, y):
        # 训练样本数和特征数
        m, n = X.shape

        # 初始化二次规划相关变量：P/q/G/h
        self.P = matrix(np.identity(n + 1, dtype=np.float))
        self.q = matrix(np.zeros((n + 1,), dtype=np.float))
        self.G = matrix(np.zeros((m, n + 1), dtype=np.float))
        self.h = -matrix(np.ones((m,), dtype=np.float))

        # 将数据转为变量
        self.P[0, 0] = 0
        for i in range(m):
            self.G[i, 0] = -y[i]
            self.G[i, 1:] = -X[i, :] * y[i]
        
        # 构建二次规划求解
        sol = solvers.qp(self.P, self.q, self.G, self.h)

        # 对权重和偏置寻优
        self.w = np.zeros(n,) 
        self.b = sol['x'][0] 
        for i in range(1, n + 1):
            self.w[i - 1] = sol['x'][i]
        return self.w, self.b

    ### 定义模型预测函数
    def predict(self, X):
        return np.sign(np.dot(self.w, X.T) + self.b)
```



### 线性支持向量机

#### 松弛系数

存在特异点（outlier）导致非线性可分，故引入一个大于零的松弛变量 $\xi_i$ 使得约束条件变为：
$$
y_i(wx_i+b)\geqslant 1- \xi_i
$$

#### 最优化问题

$$
\mathop{min}_{w,b}\ \frac{1}{2}||w||^2 +C\sum_{i=1}^N\xi_i \\
s.t. \ y_i(wx_i+b) -1 \geqslant 0, i=1,2,...,N
$$

$C$ 为惩罚系数

也可以利用合页损失函数进行优化

#### 软间隔与支持向量

软间隔的支持向量 $x_i$ 或在间隔边界上，或在间隔边界与分离超平面之间，或在超平面误分一侧

<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220611222512879.png" alt="image-20220611222512879" style="zoom:50%;" />

#### 算法实现

```python
from cvxopt import matrix, solvers

### 定义一个线性核函数
def linear_kernel(x1, x2):
    '''
    输入:
    x1: 向量1
    x2: 向量2
    输出:
    np.dot(x1, x2): 两个向量的点乘
    '''
    return np.dot(x1, x2)

### 定义近似线性可分支持向量机
### 软间隔最大化策略
class Soft_Margin_SVM:
    ### 定义基本参数
    def __init__(self, kernel=linear_kernel, C=None):
        # 软间隔svm核函数，默认为线性核函数
        self.kernel = kernel
        # 惩罚参数
        self.C = C
        if self.C is not None: 
            self.C = float(self.C)
    
    ### 定义线性支持向量机拟合方法
    def fit(self, X, y):
        # 训练样本数和特征数
        m, n = X.shape
        
        # 基于线性核计算Gram矩阵
        K = self._gram_matrix(X)
                
        # 初始化二次规划相关变量：P/q/G/h
        P = matrix(np.outer(y,y) * K)
        q = matrix(np.ones(m) * -1)
        A = matrix(y, (1, m))
        b = matrix(0.0)
        
        # 未设置惩罚参数时的G和h矩阵
        if self.C is None:
            G = matrix(np.diag(np.ones(m) * -1))
            h = matrix(np.zeros(m))
        # 设置惩罚参数时的G和h矩阵
        else:
            tmp1 = np.diag(np.ones(m) * -1)
            tmp2 = np.identity(m)
            G = matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(m)
            tmp2 = np.ones(m) * self.C
            h = matrix(np.hstack((tmp1, tmp2)))

        # 构建二次规划求解
        sol = solvers.qp(P, q, G, h, A, b)
        # 拉格朗日乘子
        a = np.ravel(sol['x'])

        # 寻找支持向量
        spv = a > 1e-5
        ix = np.arange(len(a))[spv]
        self.a = a[spv]
        self.spv = X[spv]
        self.spv_y = y[spv]
        print('{0} support vectors out of {1} points'.format(len(self.a), m))

        # 截距向量
        self.b = 0
        for i in range(len(self.a)):
            self.b += self.spv_y[i]
            self.b -= np.sum(self.a * self.spv_y * K[ix[i], spv])
        self.b /= len(self.a)

        # 权重向量
        self.w = np.zeros(n,)
        for i in range(len(self.a)):
            self.w += self.a[i] * self.spv_y[i] * self.spv[i]

    ### 定义Gram矩阵计算函数
    def _gram_matrix(self, X):
        m, n = X.shape
        K = np.zeros((m, m))
        # 遍历计算Gram矩阵
        for i in range(m):
            for j in range(m):
                K[i,j] = self.kernel(X[i], X[j])
        return K
    
    ### 定义映射函数
    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
    
    ### 定义模型预测函数
    def predict(self, X):
        return np.sign(np.dot(self.w, X.T) + self.b)
```



### 非线性支持向量机

#### 核技巧 kernel trick

将非线性问题通过变化转化为线性问题
$$
映射函数：\phi(x):\chi → \varkappa \\
使得对所有 x,z \in \chi，函数 K(x,z)满足条件：K(x, z)=\phi(x)\phi(z)
$$

#### 常用核函数

- 多项式核函数

$$
K(x, z)=(xz+1)^p
$$

- 高斯核函数

$$
K(x, z)=exp(-\frac{||x-z||^2}{2\sigma^2})
$$

- 字符串核函数

定义在字符串集合上的核函数，在文本分类、信息检索、生物信息学方面有应用

#### 算法实现

```python
from cvxopt import matrix, solvers

### 定义高斯核函数
def gaussian_kernel(x1, x2, sigma=5.0):
    '''
    输入:
    x1: 向量1
    x2: 向量2
    输出:
    两个向量的高斯核
    '''
    return np.exp(-1 * np.linalg.norm(x1-x2)**2 / (2 * (sigma ** 2)))

### 定义线性不可分支持向量机
### 借助于高斯核函数转化为线性可分的情形
class Non_Linear_SVM:
    ### 定义基本参数
    def __init__(self, kernel=gaussian_kernel):
        # 非线性可分svm核函数，默认为高斯核函数
        self.kernel = kernel
    
    ### 定义非线性可分支持向量机拟合方法
    def fit(self, X, y):
        # 训练样本数和特征数
        m, n = X.shape
        
        # 基于线性核计算Gram矩阵
        K = self._gram_matrix(X)
                
        # 初始化二次规划相关变量：P/q/A/b/G/h
        P = matrix(np.outer(y,y) * K)
        q = matrix(np.ones(m) * -1)
        A = matrix(y, (1, m))
        b = matrix(0.0)
        G = matrix(np.diag(np.ones(m) * -1))
        h = matrix(np.zeros(m))

        # 构建二次规划求解
        sol = solvers.qp(P, q, G, h, A, b)
        # 拉格朗日乘子
        a = np.ravel(sol['x'])

        # 寻找支持向量
        spv = a > 1e-5
        ix = np.arange(len(a))[spv]
        self.a = a[spv]
        self.spv = X[spv]
        self.spv_y = y[spv]
        print('{0} support vectors out of {1} points'.format(len(self.a), m))

        # 截距向量
        self.b = 0
        for i in range(len(self.a)):
            self.b += self.spv_y[i]
            self.b -= np.sum(self.a * self.spv_y * K[ix[i], spv])
        self.b /= len(self.a)

        # 权重向量
        self.w = None

    ### 定义Gram矩阵计算函数
    def _gram_matrix(self, X):
        m, n = X.shape
        K = np.zeros((m, m))
        # 遍历计算Gram矩阵
        for i in range(m):
            for j in range(m):
                K[i,j] = self.kernel(X[i], X[j])
        return K
    
    ### 定义映射函数
    def project(self, X):
        y_pred = np.zeros(len(X))
        for i in range(X.shape[0]):
            s = 0
            for a, spv_y, spv in zip(self.a, self.spv_y, self.spv):
                s += a * spv_y * self.kernel(X[i], spv)
            y_pred[i] = s
        return y_pred + self.b
    
    ### 定义模型预测函数
    def predict(self, X):
        return np.sign(self.project(X))
```



## 2. sklearn-API

### 参数说明

class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)

- kernel：核函数

1. 'linear'：线性核

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/10b8bc8b-1f98-432b-8018-3483b92ecf50/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/10b8bc8b-1f98-432b-8018-3483b92ecf50/Untitled.png)

1. 'poly'：多项式核

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/336b8bde-c6ff-4891-942c-f79a79ffed76/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/336b8bde-c6ff-4891-942c-f79a79ffed76/Untitled.png)

degree代表d，表示多项式的次数；gamma为多项式的系数；coef0代表r，表示多项式的偏置

1. 'sigmoid'：双曲正切核

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3e340d6d-2ba0-4f8a-a6ed-f030393e66c2/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3e340d6d-2ba0-4f8a-a6ed-f030393e66c2/Untitled.png)

1. 'rbf'：高斯径向基

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/770fdbe9-7ba0-487c-83a1-8f13078615b7/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/770fdbe9-7ba0-487c-83a1-8f13078615b7/Untitled.png)

线性核函数对线性可分的分类效果较好，rbf对非线性分类效果较好；线性核函数的计算效率较低，可以使用poly（degree=1）来替代；poly的degree越高，越耗时；rbf和poly都不擅长处理未归一化数据

### 示例

#### 线性可分SVM

```python
# 导入sklearn线性SVM分类模块
from sklearn.svm import LinearSVC
# 创建模型实例
clf = LinearSVC(random_state=0, tol=1e-5)
# 训练
clf.fit(X_train, y_train)
# 预测
y_pred = clf.predict(X_test)
# 计算测试集准确率
print(accuracy_score(y_test, y_pred))
```

#### 线性SVM

```python
from sklearn import svm

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy of soft margin svm based on sklearn: ', 
      accuracy_score(y_test, y_pred))
```

#### 非线性SVM

```python
from sklearn import svm

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Accuracy of soft margin svm based on sklearn: ', 
      accuracy_score(y_test, y_pred))
```



```python
from sklearn.svm import SVC    # SVC类形式
from sklearn import svm     # svc的函数形式

clf = SVC(kernel='linear').fit(x, y)
clf.predict(x)    # 根据决策边界对样本进行分类
clf.score(x, y)    # 准确度
clf.support_vectors_    # 返回支持向量
clf.n_support_      # 每个类返回的支持向量的个数
```

- 核函数的参数

1. degree：默认为3，若核函数非poly，则忽略这个参数
2. gamma：浮点数；'auto'——1/n_features，默认；'scale'——1/(n_features*X.std())
3. coef0：浮点数，默认为0.0；核函数的常数项

```python
# 利用网格搜索寻找poly核函数的最佳参数
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
import numpy as np

gamma_range = np.logspace(-10, 1, 20)   # 生成等对数间距的数列
coef0_range = np.logspace(1, 5, 10)

param_grid = dict(gamma = gamma_range, coef0 = coef0_range)     # 生成网格参数

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=1)
clf = SVC(kernel='poly', degree=1, cache_size=5000)     # cache_size表示使用缓存大小
grid = GridSearchCV(clf, param_grid=param_grid, cv=cv)

grid.fit(x, y)

print(grid.best_params_, grid.best_score_)
```

- C：松弛系数的惩罚项系数；默认为1，大于0；C较大时，svc会选择边际较小，能够更好得包括所有分类样本（硬间隔），C越小，软间隔
- class_weight：样本不均衡；默认None；字典——class_weight[i] * C；'balanced'——n_samples/(n_classes * np.bincount(y))；也可在fit中设置sample_weight；在做样本均衡之后，对少数类的分类效果更好（但对多数类的分类效果更差）

## 评估指标

- probability：启用概率估计，默认为False

在二分类情况下，SVC使用Platt缩放生成概率（在decision_function生成的距离上进行sigmoid压缩，并附加训练数据的交叉验证你和，生成类逻辑回国的SVM分数）；多分类时参考Wu et al. (2004)发表论文。

```python
svm = svm.SVC(kernel='linear', C=1.0, probability=True).fit(X, y)
svm.decision_function(X)
svm.predict_proba(X)
```





# 线性判别分析

Linear Discriminant Analysis，LDA

## 1. 原理

### 1.1 简介

一种监督学习的降维技术，将数据在低维度上进行投影，使得同一类数据尽可能接近，不同类数据尽可能疏远

<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220603190053359.png" alt="image-20220603190053359" style="zoom:50%;" />

### 1.2 公式推导

#### 目标函数

以二分类为例：

给定数据集 $D={(X_I,Y_I)}_{i=1}^m, y_i\in\{0, 1\}$

定义 $X_i、\mu_i、\Sigma_i$ 分别表示第 $i\in\{0, 1\}$ 类数据的集合、均值向量、协方差矩阵，$w^T$表示投影矩阵

保证同类样本在投影后协方差尽可能小，类中心距离尽可能大

故最大化目标函数：
$$
J=\frac{||w^T\mu_0-w^T\mu_1||_2^2}{w^T\Sigma_0w+w^T\Sigma_1w}=\frac{w^T(\mu_0-\mu_1)(\mu_0-\mu_1)^Tw}{w^T(\Sigma_0+\Sigma_1)w}=\frac{w^TS_bw}{w^TS_ww}
$$
类内散度矩阵$S_w$，类间散度矩阵$S_b$

根据条件约束优化求解的拉格朗日乘子法可以得到
$$
w=S_w^{-1}(\mu_0-\mu_1)
$$
其中$S_w^{-1}$可由SVD求解

#### 算法流程

1. 对数据按类别分组，分别计算每组样本的均值和协方差
2. 计算类内散度矩阵 $S_w$
3. 计算均值差 $\mu_0-\mu_1$
4. SVD方法计算类内散度矩阵的逆 $S_w^{-1}$
5. 计算投影矩阵 $w$。
6. 计算投影后的数据点 $Y = S_w^TX$



## 2. sklearn-API

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```



## 4. Numpy实现

### LDA模型

```python
import numpy as np

class LDA():
    def __init__(self):
        # 初始化权重矩阵
        self.w = None
        
    # 计算协方差矩阵
    def calc_cov(self, X, Y=None):
        m = X.shape[0]
        # 数据标准化
        X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
        Y = X if Y == None else (Y - np.mean(Y, axis=0))/np.std(Y, axis=0)
        return 1 / m * np.matmul(X.T, Y)
    
    # 对数据进行投影
    def project(self, X, y):
        self.fit(X, y)
        X_projection = X.dot(self.w)
        return X_projection
    
    # LDA拟合过程
    def fit(self, X, y):
        # 按类分组
        X0 = X[y == 0]
        X1 = X[y == 1]

        # 分别计算两类数据自变量的协方差矩阵
        sigma0 = self.calc_cov(X0)
        sigma1 = self.calc_cov(X1)
        # 计算类内散度矩阵
        Sw = sigma0 + sigma1

        # 分别计算两类数据自变量的均值和差
        u0, u1 = np.mean(X0, axis=0), np.mean(X1, axis=0)
        mean_diff = np.atleast_1d(u0 - u1)

        # 对类内散度矩阵进行奇异值分解
        U, S, V = np.linalg.svd(Sw)
        # 计算类内散度矩阵的逆
        Sw_ = np.dot(np.dot(V.T, np.linalg.pinv(np.diag(S))), U.T)
        # 计算w
        self.w = Sw_.dot(mean_diff)
    
    # LDA分类预测
    def predict(self, X):
        y_pred = []
        for sample in X:
            h = sample.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred
```

### 示例

```python
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = datasets.load_iris()
X = data.data
y = data.target
X = X[y != 2]
y = y[y != 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

lda = LDA()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```



## 5. 与PCA之间的异同点

### 相同点

- 两者均可以对数据进行降维

- 两者在降维时均使用了矩阵特征分解的思想。

- 两者都假设数据符合高斯分布。

### 不同点

- LDA是有监督的降维方法，而PCA是无监督的降维方法

- LDA降维最多降到类别数k-1的维数，而PCA没有这个限制。

- LDA除了可以用于降维，还可以用于分类。

- LDA选择分类性能最好的投影方向，而PCA选择样本点投影具有最大方差的方向。



## 6. 优缺点

### 优点

- 在降维过程中可以使用类别的先验知识经验

- LDA在样本分类信息依赖均值而不是方差的时候，比PCA之类的算法较优。

 ### 缺点

- LDA不适合对非高斯分布样本进行降维，PCA也有这个问题。
- LDA降维最多降到类别数k-1的维数，如果我们降维的维度大于k-1，则不能使用LDA。当然目前有一些LDA的进化版算法可以绕过这个问题。
- LDA在样本分类信息依赖方差而不是均值的时候，降维效果不好。
- LDA可能过度拟合数据



# 贝叶斯网络

