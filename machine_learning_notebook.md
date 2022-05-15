# 有监督算法

## 线性回归

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



## 逻辑斯蒂回归

Logistic Regression

### 1. 原理

#### 假设函数

$h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-(\theta^Tx)}}$

#### 损失函数

$L(\theta)=\frac{1}{m}[\sum_{i=1}^my^{(i)}logh_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)})]$

#### 优化过程

- 梯度下降法

$\theta_j:=\theta_j-\alpha\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$

#### 特点

- 线性回归+sigmoid()

- 逻辑回归在线性关系的拟合效果非常好，但对非线性关系拟合效果非常差
- 计算速度快（效率高于SVM和RF）
- 在小数据上标线比树模型更好

- 一般不用PCA和SVD降维；统计方法可以用，但没必要



### 2. sklearn-API

#### 参数说明

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

#### 属性

| 属性             | 说明                                                      |
| ---------------- | --------------------------------------------------------- |
| classes_         | ndarray，分类标签                                         |
| coef_            | ndarray，系数                                             |
| intercept_       | float（0.0）或array，偏置                                 |
| n_features_in_   | int，输入特征数                                           |
| feature_names_in | array，输入特征名称                                       |
| n_iter_          | ndarry，迭代次数（当solver为'liblinear'时会返回多个元素） |

#### 方法

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

#### 实例

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

### 3. 评估指标

#### 混淆矩阵

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

#### ROC曲线

- ROC定义

ROC全称是“受试者工作特征”（Receiver Operating Characteristic）

- ROC计算方法

以假阳率（FPR）为横坐标，以真阳率（TPR）为纵坐标

FPR = FP / (FP + TN)  指分类器预测的正类中实际负实例占所有负实例的比例

TPR = TP / (TP + FN)  指分类器预测的正类中实际正实例占所有正实例的比例

希望FPR越小越好，TPR越大越好

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



### 4. 解决样本不平衡问题

通常采用上采样

```python
import imblearn
# imblearn是一个专门处理不平衡数据的库
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
x, y = sm.fit_sample(x, y)
```



### 5. Numpy算法

#### 训练和预测

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

#### 绘制决策边界

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



## Lasco 回归

### 1. 原理





# 无监督算法

## K-means算法

### 1. 原理

#### 算法原理

1. 随机选择k个中心
2. 遍历所有样本，把样本划分到距离最近的一个中心
3. 划分之后就有K个簇，计算每个簇的平均值作为新的质心
4. 重复步骤2，直到达到停止条件
5. 停止：聚类中心不再发生变化；所有的距离最小；迭代次数达到设定值

#### 算法复杂度

k-means运用了 Lioyd’s 算法,平均计算复杂度是 O(k*n*T)，其中n是样本量，T是迭代次数。

计算复杂读在最坏的情况下为 O(n^(k+2/p))，其中n是样本量，p是特征个数。

### 2. sklearn API

#### 参数说明

```
KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto')
```

| 参数                | 含义                                                         |
| ------------------- | ------------------------------------------------------------ |
| n-cluster           | int，分类簇的数量                                            |
| max_iter            | int，执行一次k-means算法所进行的最大迭代数                   |
| n_init              | 用不同的质心初始化值运行算法的次数，最终解是在inertia意义下选出的最优结果 |
| init                | string或数组，初始化策略<br />'kmeans++'表示初始均值向量之间距离比较远，效果较好<br />random表示从数据中随机选择K个样本作为初始均值向量<br />（n_cluster,n_features）数组作为初始均值向量 |
| precompute_distance | Bool或者'auto'，预计算距离，计算速度快但占用内存<br />'auto'表示如果n_samples*k>12 million，则不提前计算（在版本0.22中已弃用） |
| tol                 | float，算法收敛的阈值，与inertia结合来确定收敛条件           |
| n_jobs              | int，计算所用的进程数，内部原理是同时进行n_init指定次数的计算<br />-1表示用所有的CPU进行运算 <br />1表示不进行并行运算<br />值小于-1表示用到的CPU数为(n_cpus + 1 + n_jobs) |
| random_state        | int或numpy.RandomState类型，可选用于初始化质心的随机数生成器 |
| verbose             | int，日志模式<br />0表示不输出日志信息<br />1表示每隔一段时间打印一次日志信息<br />如果大于1，打印次数频繁 |
| copy_x              | bool，当我们precomputing distances时，将数据中心化会得到更准确的结果<br />True则原始数据不会被改变<br />False则会直接在原始数据上做修改并在函数返回值时将其还原 |
| algorithm           | float，算法类型<br />"full"：经典的EM风格算法<br /> "elkan"：使用三角形不等式算法，对于定义良好的聚类的数据更有效，但是分配了额外的形状数组（n_samples，n_clusters），因此需要更多的内存。 <br />"auto"（保持向后兼容性）选择"elkan" |

#### 属性

| 属性             | 含义                                            |
| ---------------- | ----------------------------------------------- |
| cluster_centers_ | 向量，[n_clusters, n_features] (聚类中心的坐标) |
| Labels_          | 每个点的分类                                    |
| inertia_         | float，每个点到其簇的质心的距离之和             |

#### 方法

- fit(X[,y]): 计算k-means聚类。
- fit_predict(X[,y]): 计算簇质心并给每个样本预测类别。
- fit_transform(X[,y])：计算簇并 transform X to cluster-distance space。
- get_params([deep])：取得估计器的参数。
- predict(X): 给每个样本估计最接近的簇。
- score(X[,y]): 计算聚类误差
- set_params(**params): 为这个估计器手动设定参数。
- transform(X[,y]): 将X转换为群集距离空间。 　
  在新空间中，每个维度都是到集群中心的距离。请注意，即使X是稀疏的，转换返回的数组通常也是密集的。

#### 实例

```python
from sklearn.datasets import load_iris
import xlwt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# 加载鸢尾花数据
iris = load_iris()
iris_data = iris['data']
iris_target = iris['target']
iris_names = iris['feature_names']

# 标准化
data_zs = (iris_data - iris_data.mean()) / iris_data.std()
# minmax标准化
scale = MinMaxScaler().fit(iris_data)
iris_datascale = scale.transform(iris_data)

# 聚类
kmeans = KMeans(n_clusters=3, random_state=123).fit(iris_datascale)

# 预测，预测的数据需要使用和训练数据同样的标准化才行。
result = kmeans.predict([[5.6,2.8,4.9,2.0]])
```

### 3. 评估指标

#### 评估体系

| 方法              | 真实值 | 最佳值     | sklearn接口                |
| ----------------- | ------ | ---------- | -------------------------- |
| ARI(兰德系数)     | 需要   | 1.0        | adjusted_rand_score        |
| AMI(互信息)       | 需要   | 1.0        | adjusted_mutual_info_score |
| V-measure         | 需要   | 1.0        | completeness_score         |
| FMI               | 需要   | 1.0        | fowlkes_mallows_score      |
| 轮廓系数          | 不需要 | 畸变程度大 | silhouette_score           |
| Calinski_ Harabaz | 不需要 | 最大值     | calinski_harabaz_score     |

#### FMI评价法

```python
from sklearn.metrics import fowlkes_mallows_score
for i in range(2, 7):
    kmeans = KMeans(n_clusters=i, random_state=123).fit(iris_data)
    score = fowlkes_mallows_score(iris_target, kmeans.labels_)
    print("聚类%d簇的FMI分数为：%f" % (i, score))
```

#### 轮廓系数

```python
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
silhouettescore=[]
for i in range(2, 15):
    kmeans = KMeans(n_clusters=i, random_state=123).fit(iris_data)
    score = silhouette_score(iris_data, kmeans.labels_)
    silhouettescore.append(score)
plt.figure(figsize=(10, 6))
plt.plot(range(2, 15), silhouettescore, linewidth=1.5, linestyle='-')
plt.show()
```

变化快的部分（斜率大）就是分类的最佳选择

#### Calinski-Harabasz指数评价

```python
from sklearn.metrics import calinski_harabaz_score
for i in range(2, 7):
    kmeans = KMeans(n_clusters=i, random_state=123).fit(iris_data)
    score = calinski_harabaz_score(iris_data, kmeans.labels_)
    print("聚类%d簇的calinski_harabaz分数为：%f" % (i, score))
```

### 4. Numpy算法

```python
import numpy as np
# 定义欧式距离
def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return np.sqrt(distance)

# 定义中心初始化函数
def centroids_init(k, X):
    m, n = X.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        # 每一次循环随机选择一个类别中心
        centroid = X[np.random.choice(range(m))]
        centroids[i] = centroid
    return centroids

# 定义样本的最近质心点所属的类别索引
def closest_centroid(sample, centroids):
    closest_i = 0
    closest_dist = float('inf')
    for i, centroid in enumerate(centroids):
        # 根据欧式距离判断，选择最小距离的中心点所属类别
        distance = euclidean_distance(sample, centroid)
        if distance < closest_dist:
            closest_i = i
            closest_dist = distance
    return closest_i

# 定义构建类别过程
def build_clusters(centroids, k, X):
    clusters = [[] for _ in range(k)]
    for x_i, x in enumerate(X):
        # 将样本划分到最近的类别区域
        centroid_i = closest_centroid(x, centroids)
        clusters[centroid_i].append(x_i)
    return clusters

# 根据上一步聚类结果计算新的中心点
def calculate_centroids(clusters, k, X):
    n = X.shape[1]
    centroids = np.zeros((k, n))
    # 以当前每个类样本的均值为新的中心点
    for i, cluster in enumerate(clusters):
        centroid = np.mean(X[cluster], axis=0)
        centroids[i] = centroid
    return centroids

# 获取每个样本所属的聚类类别
def get_cluster_labels(clusters, X):
    y_pred = np.zeros(X.shape[0])
    for cluster_i, cluster in enumerate(clusters):
        for X_i in cluster:
            y_pred[X_i] = cluster_i
    return y_pred

# 根据上述各流程定义kmeans算法流程
def kmeans(X, k, max_iterations):
    # 1.初始化中心点
    centroids = centroids_init(k, X)
    # 遍历迭代求解
    for _ in range(max_iterations):
        # 2.根据当前中心点进行聚类
        clusters = build_clusters(centroids, k, X)
        # 保存当前中心点
        prev_centroids = centroids
        # 3.根据聚类结果计算新的中心点
        centroids = calculate_centroids(clusters, k, X)
        # 4.设定收敛条件为中心点是否发生变化
        diff = centroids - prev_centroids
        if not diff.any():
            break
    # 返回最终的聚类标签
    return get_cluster_labels(clusters, X)
```

