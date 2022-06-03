# 有监督算法篇

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

- 通常采用交叉验证来选取最优k值

#### 分类决策规则

- 多数表决



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
