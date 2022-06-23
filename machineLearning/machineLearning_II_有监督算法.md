有监督算法篇



# 1. 感知机

perception

## 1.1 原理

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



## 1.2 算法实现

 [1. perceptron.ipynb](code\1. perceptron.ipynb) 



# 2. 线性回归

## 2.1 多元线性回归模型

### 2.1.1 原理

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

### 2.1.2 sklearn-API

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

### 2.1.3 评估指标

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

### 2.1.4 交叉验证

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

### 2.1.5 Numpy算法

 [2. linear_regression.ipynb](code\2. linear_regression.ipynb) 



## 2.2 Lasco 回归

### 2.2.1 原理

对多元线性回归的损失加上L1范式惩罚，通过加入惩罚项，将一些不重要的自变量系数调整为0，从而达到剔除变量的目的

#### 假设函数

$h_\theta(x)=\theta_0 + \theta_1x$

#### 损失函数

普通最小二乘法

$L(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2+\lambda|\theta_1|$

### 2.2.2 skleran-API

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

### 2.2.3 评估指标



### 2.2.4. 算法实现

 [2. lasso.ipynb](code\2. lasso.ipynb) 



## 2.3 Ridge 回归

### 2.3.1 原理

对多元线性回归的损失加上L2范式惩罚，通过加入惩罚项，将一些不重要的自变量系数调整为接近0

### 2.3.2 sklearn-API

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

### 2.3.3 评估指标





### 2.3.4 算法实现

 [2. ridge.ipynb](code\2. ridge.ipynb) 



# 3. 逻辑斯蒂回归

Logistic Regression

## 3.1 原理

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



## 3.2 sklearn-API

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



## 3.3 评估指标

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



## 3.4 解决样本不平衡问题

通常采用上采样

```python
import imblearn
# imblearn是一个专门处理不平衡数据的库
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
x, y = sm.fit_sample(x, y)
```



## 3.5 算法实现

 [3. logistic_regression.ipynb](code\3. logistic_regression.ipynb) 



# 4. 支持向量机

## 4.1 原理

### 4.1.1 简介

- support vector machines, SVM

特征空间上的间隔最大的线性分类器，通过核技巧也可用于非线性分类

学习策略为间隔最大化——求解凸二次规划问题（convex quadratic programming）

- 种类

线性可分支持向量机——硬间隔最大化

线性支持向量机——软间隔最大化

非线性支持向量机——核方法



### 4.1.2 线性可分支持向量机

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

 [4_1_hard_margin_svm.ipynb](code\4_1_hard_margin_svm.ipynb) 



### 4.1.3 线性支持向量机

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

 [4_2_soft_margin_svm.ipynb](code\4_2_soft_margin_svm.ipynb) 



### 4.1.4 非线性支持向量机

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

 [4_3_non-linear_svm.ipynb](code\4_3_non-linear_svm.ipynb) 



## 4.2 sklearn-API

### 参数说明

class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)

- kernel：核函数

1. 'linear'：线性核

1. 'poly'：多项式核

degree代表d，表示多项式的次数；gamma为多项式的系数；coef0代表r，表示多项式的偏置

1. 'sigmoid'：双曲正切核

1. 'rbf'：高斯径向基



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



## 4.3 评估指标

- probability：启用概率估计，默认为False

在二分类情况下，SVC使用Platt缩放生成概率（在decision_function生成的距离上进行sigmoid压缩，并附加训练数据的交叉验证你和，生成类逻辑回国的SVM分数）；多分类时参考Wu et al. (2004)发表论文。

```python
svm = svm.SVC(kernel='linear', C=1.0, probability=True).fit(X, y)
svm.decision_function(X)
svm.predict_proba(X)
```





# 5. 树相关模型



## 5.1 决策树

decision tree, DT

### 5.1.1 原理

#### 简介

- 一种分类的树形结构（也可以用作回归），在分类中，基于特征对实例进行分类，可以看做是if-then规则的集合，也可以认为是定义在特征空间与类空间上的条件概率分布
- 决策树由结点和有向边组成。结点又分为内部结点（表示特征或属性）和叶结点（表示一个类）。决策树从根结点开始对实例某一特征进行分类，并将实例分配到其子结点上，不断递归直至到达子结点。
- 主要包括三个步骤：特征选择、决策树生成、决策树修剪
- 决策树学习的思想：ID3算法、C4.5算法、CART算法

- 决策树学习主要流程

开始，构建根节点，将所有训练数据放在根节点，选择一个最优特征，按照这一特征将训练集分割为子集，使得各个子集有一个在当前条件下最好的分类。如果这些子集已经能够被基本正确分类，那么构建叶结点，并将这些子集分到所对应的叶结点中，若不能正确分类，则重新选择分类特征。如此递归执行，直至训练集子集被基本正确分类，或者没有合适的特征为止。

#### ID3算法

利用信息增益来学习决策树

##### 信息熵 information entropy

设$X$是一个离散随机变量，其概率分布为
$$
P(X=x_i)=p_i, \ \ i=1,2,...,n
$$
则其熵表示为
$$
H(p)=-\sum_{i=1}^np_ilog_2p_i
$$
熵的单位是比特（以2为底）或纳特（以e为底），熵越大，随机变量的不确定性就越大

##### 信息增益 information gain

表示得知特征X的信息而使得类Y的信息的不确定性减少的程度，也叫互信息
$$
g(Y, X)=H(Y)-H(Y|X)=H(Y)-\sum_{i=1}^np_iH(Y|X=x_i)
$$
信息增益越大，则该特征对数据集确定性贡献越大，表示该特征对数据有较强的分类能力。

##### 学习过程

从根结点开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征，由该特征的不同取值建立子结点；在递归调用以上方法；直到所有特征的信息增益均很小或没有特征可以选择为止

##### numpy实现

 [5_1_ID3.ipynb](code\5_1_ID3.ipynb) 



#### C4.5算法

##### 信息增益比

$$
g_R(Y,X)=\frac{g(Y,X)}{H_X(Y)}, H_X(Y)=-\sum_{i=1}^n\frac{|Y_i|}{|Y|}log_2\frac{|Y_i|}{Y}
$$

##### 学习过程

- 与ID3法类似，不同的是采用信息增益比替代信息增益选择特征

##### 剪枝

- 生成的决策树对训练数据拟合效果好，但容易过拟合，需要剪枝（从已生成的树上裁减一些叶结点）

- 剪枝算法之一 —— 极小化决策树损失函数（类似正则化）

树 $T$ 的叶结点个数为 $|T|$，在叶结点 $t$ 处有 $N_t$ 个样本，其中 $k$ 类样本数有 $N_{tk}$ 个

决策树的损失函数可以定义为：
$$
C_\alpha(T)=\sum_{i=1}^{|T|}N_tH_t(T) + \alpha|T|,其中经验熵H_t(T)=-\sum_k\frac{N_tk}{N_t}log\frac{N_{tk}}{N_k}
$$
剪枝算法：计算每个叶结点回缩至父结点前后的损失，若剪枝后损失未减小则不剪枝，不断循环至不能再剪枝为止

#### CART算法

classification and regression tree, CART

##### 简介

- CART是在给定输入随机变量X条件下输出随机变量Y的条件概率分布的学习方法

- 假设决策树是二叉树
- 算法流程：基于训练集生成决策树，用验证集剪枝（以损失函数最小为剪枝原则）

##### 回归树

- 回归树模型

将输入空间划分为 $M$ 个单元 ，且每个单元 $R_m$ 上有固定输出值 $c_m$
$$
f(x)=\sum_{m=1}^Mc_mI(x\in R_m)
$$

- 损失：空间划分的误差采用平方误差来衡量（最小二乘回归树）

$$
\sum_{x_i\in R_m}(y_i-f(x_i))^2
$$

##### 分类树

- 基尼系数

假设有 $K$ 类，样本属于第 $k$ 类的概率为 $p_k$，则概率分布的基尼系数定义为
$$
Gini(p)=\sum_{k=1}^Kp_k(1-p_k)=1-\sum_{k=1}^Kp_k^2
$$
基尼系数表示集合的不确定度，其值越大，则不确定度越大

- 分类树采用基尼系数选择特征，选择基尼系数最小的特征作为切分点

##### 剪枝

与ID3和C4.5的剪枝算法类似，不同的是损失采用最小二乘或基尼系数进行衡量

##### numpy实现

 [5_1_CART.ipynb](code\5_1_CART.ipynb) 



### 5.1.2 分类树 sklearn-API

#### 参数说明

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

#### 属性

- classes_    输出所有标签
- feature_importances_    特征重要性
- max_feature_    参数max_feature的推断值
- n_classes_    标签类别的数据
- n_features_    训练时使用的特征个数
- n_outputs_    训练时输出的结果个数
- tree_    可以访问树的结构和低级属性

#### 接口

- clf.apply(X_test)    返回测试样本所在的叶子节点的索引
- clf.predict(X_test)    返回测试样本的分类/回归结果
- clf.fit(x_train, y_train)
- clf.score(X_test, y_test)

不接受一维矩阵作为特征输入，必须转化为二维矩阵（reshape(-1, 1)）



### 5.1.3 回归树 sklearn-API

#### 参数

- criterion  衡量分枝质量的指标

1. mse    均方误差（也是最常用的回归树回归质量的指标）
2. fredman_mse    费尔德曼均方误差
3. mae    绝对平均误差

#### 属性&接口

和分类树一致



### 5.1.4 其他相关代码实现

#### 交叉验证

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
regressor = DecisionTreeRegressor(random_state=0)
cross_val_score(regressor, x_data, y_data, cv=10, scoring='neg_mean_squared_error')
```

不需要额外划分数据集和测试集

#### 网格搜索

```python
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

#### 画DT树

```python
from sklearn import tree
import graphvize

feature_names = []  # 定义特征名
label_names = []  # 标签名
dot_data = tree.export_graphviz(clf, feature_names, class_names, filled=True, rounded=True)  # 定义DT
graph = graphvize.Source(dot_data)  # 绘图
```

filled和rounded定义了框的填充颜色和形状

#### 查看DT树属性

```python
clf.feature_importance_
[*zip(feature_names, clf.feature_importance_)]
```

#### 其他

DT对环形数据的分类效果较差



#### 示例

- ID3算法

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

- CART树

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







### 5.1.5 优缺点

#### 优点

- 模型具有可读性
- 分类速度快

#### 缺点





## 5.2 AdaBoost算法

### 5.2.1 原理

#### 集成模型

三类集成算法：bagging (RF), boosting (Adaboost, GBDT), stacking

bagging    模型独立，相互平行

boosting    按顺序逐步提升特征权重

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



### 5.2.2 sklearn-API





#### 示例

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





### 5.2.3 Numpy 代码实现

 [5_2_adaboost.ipynb](code\5_2_adaboost.ipynb) 





## 5.3 梯度提升决策树 GBDT

Gradient Boosting Decision Tree, GBDT

### 5.3.1 原理

#### 简介

- 一种基于决策树的集成算法
- GBDT 中使用 CART 作为弱分类器，通过梯度下降来对新的学习器进行迭代

#### 优化方法

##### 损失函数

- 回归问题：残差平方和

$$
L=\frac{1}{2}(y_i-c)^2, c为预测值
$$

- 分类问题：交叉熵

$$
L=-\sum_{i=1}^N[y_ilog(p)+(1-y_i)log(1-p)]=-\sum_{i=1}^N[y_ilog(odds)-log(1+e^{log(odds)})]
$$

##### 优化流程

- 初始化弱分类器（回归问题则取标签的平均值，仅为一片叶子）

$$
f_0(x)=argmin_{\gamma}\sum_{i=1}^NL(y_i,\gamma)
$$

该求解过程采用梯度下降法求解

- 计算样本的负梯度（残差）

$$
r_{im}=-[\frac{\partial L(y_i,f(x_i))}{\partial f(x_i)}]_{f(x)=f_{m-1}(x)}
$$

根据损失函数可得上式即为计算残差（$y_{predict}-y_{mean}$）

- 将残差作为标签，用特征拟合得到一棵决策树

$$
r_{jm}=argmin_\gamma\sum_{i=1}^NL(y_i,f_{m-1}(x_i) + \gamma)
$$

其结果即为残差平均值

- 在弱分类器的基础上加上乘以学习率的新决策树

- 重复以上过程，残差越来越小，得到最终学习器





### 5.3.3 算法实现



### 5.3.4 参考资料

[Gradient Boost Part 1 (of 4): Regression Main Ideas - YouTube](https://www.youtube.com/watch?v=3CC4N4z3GJc) 系列视频





## 5.4 GBoost

### 5.4.1 原理

#### 简介

eXtreme Gradient Boosting，即极端梯度提升树



#### 损失函数及优化过程

##### 回归问题

- 损失函数组成

$$
L(y_i, p_i)=\frac{1}{2}\sum_{i=1}^N(y_i-p_i)^2+\frac{1}{2}\lambda O_{value}^2+\gamma T=\frac{1}{2}\sum_{i=1}^N[y_i-(p_0+O_{value})]^2+\frac{1}{2}\lambda O_{value}^2
$$

第一项表示了真实值与预测值之间的残差平方和（预测值初始值为0.5）

第二项为正则化，类似于岭回归，$O_{value}$ 表示树的输出结果，$\lambda$ 越大，$O_{value}$ 越接近于0

第三项中，$\gamma$ 表示惩罚项，$T$ 表示结点数，该项为剪枝选项，但由于剪枝发生在树生成之后，故可不计入损失

- 最小化损失函数

在优化过程中，对第一项使用了二阶泰勒近似
$$
L(y, p_i+O_{value})\approx L(y,p_i)+[\frac{d}{dp_i}L(Y,p_i)]O_{value}+\frac{1}{2}[\frac{d^2}{dp_i^2}L(y,p_i)]O_{value}^2=L(y,p_i)+gO_{value}+\frac{1}{2}hO_{value}^2
$$
故 $O_{value}$ 的最优解为
$$
O_{value}=-\frac{\sum_{i=1}^Ng_i}{\sum_{i=1}^Nh_i+\lambda}
$$
由于 $g_i=-(y_i-p_i),h_i=1$，故
$$
O_{value}=\frac{Sum\ of\ residuals}{Number\ of\ residuals + \lambda}
$$

- 分枝

计算每片叶子及根节点的 Similarity Score
$$
Similarity\ Score=\frac{(\sum_{i=1}^ng_i)^2}{\sum_{i=1}^nh_i+\lambda}=\frac{(Sum\ of\ Residual)^2}{Number\ of \ Residual+\lambda}
$$
计算 Gain 值：分支的 Similarity Score 之和减去节点的 Similarity Score 

- 剪枝：判断Similarity Socre是否超过阈值，若低于阈值则剪枝



##### 分类问题

- 损失函数

$$
L(y_i,p_i)=-[y_ilog(p_i)+(1-y_i)log(1-p_i)]
$$

$$
L(y_i,log(odds)_i)=-y_ilog(odds)_i+log(1+e^{log(odds)_i})
$$

- 优化

求一阶导数和二阶导数
$$
g_i=\frac{d}{dlog(odds)}L(y_i,log(odds)_i)=-y_i+\frac{e^{log(odds)_i}}{1+e^{log(odds)_i}}=-(y_i-p_i)
$$

$$
h_i=\frac{d^2}{dlog(odds)^2}L(y_i,log(odds)_i)=\frac{e^{log(odds)_i}}{1+e^{log(odds)_i}}×\frac{1}{1+e^{log(odds)_i}}=p_i(1-p_i)
$$

可以求出 $O_{value}$
$$
O_{value}=\frac{\sum Residual_i}{\sum[Previous\ Probability_i×(1-Previous\ Probability_i)] + \lambda}
$$

- 分枝

依次选择不同分枝策略，进行如下操作，找出 Gain 值最大的分枝策略：

计算每片叶子及根节点的 Similarity Score
$$
Similarity\ Score=\frac{(\sum Residual_i)^2}{\sum[Previous\ Probability_i×(1-Previous\ Probability_i)] + \lambda}
$$
计算 Gain 值：分支的 Similarity Score 之和减去节点的 Similarity Score 

- 剪枝：判断Similarity Socre是否超过阈值，若低于阈值则剪枝

#### 在大数据上的优化策略

- 近似贪心算法

在构建树的每一层时，不考虑下一层分枝带来的影响

近似贪心算法：采用分位数来作为限值而非每两个样本的平均值

- Quantile Sketch Algorithm

使用 Sketch 算法近似得到每个特征的 quantile

- Weight Quantile

每个 quantile 的权重为损失函数的二阶导数（Hessian矩阵）

对回归问题而言，每个样本权重相等（均为1）

对分类问题而言，$weight=p_i(1-p_i)$

quantile 分段标准是保证每段的累积 weight 尽量相等

- Parallel Learning

可以用多个计算机绘制数据的特征分布直方图，然后累加成近似直方图

#### 其他优化策略

- 应对缺失数据策略 —— Sparsity-Aware Split Finding

在分枝时将缺失数据分别放至左侧和右侧，计算 Gain 值，寻找最优策略。

预测时将缺失值选入最优分枝进行预测

- Cache-Aware Access

使用CPU缓存存储梯度 Gradients 及 Hessians，加快计算

- Blocks for Out-of-Core Computation

同时从硬盘读取数据



### 5.4.2 代码实现

 [5_4_xgboost.ipynb](code\5_4_xgboost.ipynb) 

### 5.4.3 参考资料

[XGBoost Part 3 (of 4): Mathematical Details - YouTube](https://www.youtube.com/watch?v=ZVFeW798-2I)



## 5.5 LightGBM

轻量的梯度提升机，Light Gradient Boosting Machine

### 5.5.1 原理

#### 简介

- 总体上仍然属于GBDT算法框架

- XGBoost通过预排序的算法来寻找特征的最佳分裂点，但占用空间的代价太大。XGBoost寻找最佳分裂点的算法复杂度可以估计为：复杂度=特征数量\*特征分裂点的数量\*样本数量
- LightGBM则主要针对特征数量、特征分裂点的数量、样本数量等方面进行优化，提升算法运行效率

#### 优化方法

##### Histogram算法

- 区别于 XGBoost 的预排序算法，采用 Histogram 直方图的算法寻找最佳特征分裂点。
- 其基本想法是将连续的浮点特征值进行离散化为k个整数并构造一个宽度为k的直方图。

##### GOSS 算法

- 全称为单边梯度抽样算法，Gradient-based One-Side Sampling
- 从减少样本角度进行的优化
- 将训练过程中大部分权重较小的样本剔除，仅对剩余样本数据计算信息增益

- 基本做法：先将需要进行分裂的特征按绝对值大小降序排序，取绝对值最大的前a%个数据，假设样本大小为n，在剩下的(1-a)%个数据中随机选择b%个数据，将这b%个数据乘以一个常数(1-a)/b，这种做法会使得算法更加关注训练不够充分的样本，并且原始的数据分布不会有太大改变。最后使用a+b个数据来计算该特征的信息增益。

##### EFB 算法

- 互斥特征捆绑算法，Exclusive Feature Bundling
- 针对于特征的优化
- 通过将两个互斥的特征捆绑在一起，合为一个特征，在不丢失特征信息的前提下，减少特征数量，从而加速模型训练。
- 大多数时候两个特征都不是完全互斥的，可以用定义一个冲突比率对特征不互斥程度进行衡量，当冲突比率较小时，可以将不完全互斥的两个特征捆绑，对模型精度也没有太大影响。

##### Leaf-Wise

- 区别于XGBoost的按层生长的叶子节点生长方法（Level-wise），LightGBM 使用带有深度限制的按叶子节点生长（Leaf-Wise）的决策树生成算法。

##### 其他

- 除了以上四点改进算法之外，LightGBM在工程实现上也有一些改进和优化
- 比如可以直接支持类别特征（不需要再对类别特征进行one-hot等处理）、高效并行和Cache命中率优化等。



### 5.5.2 代码实现

使用lightgbm包，提供了分类和回归两大类接口

#### 安装依赖

```shell
pip install lightgbm
```

#### 实现

 [5_5_lightgbm.ipynb](code\5_5_lightgbm.ipynb) 



### 5.5.3 参考资料

https://lightgbm.readthedocs.io/en/latest/



## 5.6 CatBoost

能够高效处理数据中的类别特征、排序提升

### 5.6.1 原理

#### 改进类别特征处理方法

##### 常规处理方法

- 对于类别型特征，以往最通用的方法就是 one-hot 编码，但当特征取值数较多时容易产生大量冗余数据

- 一种折中的方法是将类别数目进行重新归类，使其类别数目降到较少数目再进行one-hot编码。
- 另一种最常用的方法则是目标变量统计（Target Statisitics，TS），TS计算每个类别对于的目标变量的期望值并将类别特征转换为新的数值特征。
- 在 GBDT 中，对类别特征的处理方式为 Greedy Target-based Statistics（Greedy TS），即使用类别对应的标签平均值来进行替换，其计算公式为

$$
x_k^i=\frac{\sum_{j=1}^n[x_{j,k}=x_{i,k}]Y_i}{\sum_{j=1}^n[x_{j,k}=x_{i,k}]}
$$

- 该方法的缺点：会造成训练集中 label 的泄露，因为对于某个样本来说，其数值编码计算过程中已经把这个样本的 label 值纳入了计算过程中；且训练集和测试集可能会因为数据分布不一样而产生条件偏移问题

##### 改进方法

- 改进的 Greedy TS 方法：添加先验分布项，用以减少噪声和低频类别型数据对于数据分布的影响

$$
x_k^i=\frac{\sum_{j=1}^{p-1}[x_{j,k}=x_{i,k}]Y_i+ap}{\sum_{j=1}^{p-1}[x_{j,k}=x_{i,k}]+a}
$$

$p$ 为添加的先验项，$a$ 为权重系数

- Ordered TS：先将样本随机打乱，然后每个样本只使用它排序在它前面的样本来计算其类别特征的数值编码。同时设计多个样本随机排列（默认4个）

- 其他：Holdout TS、Leave-one-out TS

#### 基于贪心的特征组合

- 构建任意几个类别型特征的任意组合为新的特征

- 基于贪心策略：生成tree的第一次分裂，CatBoost不使用任何交叉特征。在后面的分裂中，CatBoost会使用生成tree所用到的全部原始特征和交叉特征 跟 数据集中的全部 类别特征进行交叉。

#### 避免预测偏移的 Ordered Boosting 方法

- 可以有效地减少梯度估计的误差，缓解预测偏移。但是会增加较多的计算量，影响训练速度。

- 先将样本随机打乱，然后每个样本只使用排序在它前面的样本来训练模型。用这样的模型来估计这个样本预测结果的一阶和二阶梯度。然后用这些梯度构建一棵tree的结构，最终tree的每个叶子节点的取值，是使用全体样本进行计算的。

#### 使用对称二叉树作为基模型

- XGBoost和LightGBM采用的基模型是普通的二叉树
- 这种对树结构上的约束有一定的正则作用。更为重要的是，它可以让CatBoost模型的推断过程极快。



### 5.6.2 算法实现

使用开源的 catboost 库

 [5_6_catboost.ipynb](code\5_6_catboost.ipynb) 



### 5.6.3 参考资料

[数学推导+纯Python实现机器学习算法19：CatBoost (qq.com)](https://mp.weixin.qq.com/s?__biz=MzI4ODY2NjYzMQ==&mid=2247487676&idx=1&sn=45d717cc543175d47c7f92cac0b0e25e&chksm=ec3bb5d4db4c3cc29493bbea9c82f9cf34f8a2eca48a5cb48679b9e3d0ea9b41a40c52554007&scene=178&cur_album_id=1369989062744211457#rd)

[30分钟学会CatBoost - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/460986009)

https://catboost.ai/docs/concepts/tutorials.html



## 5.7 随机森林

Random Forest

### 5.7.1 原理

#### Bagging 方法

其核心概念在于自助采样（Bootstrap Sampling），给定包含 m 个样本的数据集，有放回的随机抽取一个样本放入采样集中，经过 m 次采样，可得到一个和原始数据集一样大小的采样集。我们可以采样得到 T 个包含 m 个样本的采样集，然后基于每个采样集训练出一个基学习器，最后将这些基学习器进行组合。

#### 随机

- 随机选取样本
- 随机选取特征

#### 训练过程

- 假设有M个样本，有放回的随机选择M个样本（每次随机选择一个放回后继续选）。
- 假设样本有N个特征，在决策时的每个节点需要分裂时，随机地从这N个特征中选取n个特征，满足n<<N，从这n个特征中选择特征进行节点分裂。
- 基于抽样的M个样本n个特征按照节点分裂的方式构建决策树。
- 按照1~3步构建大量决策树组成随机森林，然后将每棵树的结果进行综合（分类使用投票法，回归可使用均值法）。



### 5.7.2 sklearn-API



#### 示例

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
```



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



接口：apply, fit, predict, score

属性

rfc.estimators_    查看森林中树的状况

rfc.oob_score_    查看袋外数据的测试结果











### 5.7.3 Numpy 算法实现

 [5_7_random_forest.ipynb](code\5_7_random_forest.ipynb) 



### 5.7.4 应用——缺失值填充

遍历所有特征，从缺失值最少的特征开始填充，此时其他缺失特征用0填充

```python
from sklearn.impute 
import SimpleImputer 
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor 
# 返回缺失值数量从小到大排序所对应的索引 
sortindex = np.argsort(X_missing.isnull().sum(axis=0)).values 
for i in sortindex:    
	df = X_missing.copy()    
    # 构造新标签    
    fillc = df.iloc[:, i]    
    # 构造新特征矩阵    
    df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_full)], axis=1) 
    # 缺失值零填充    
    df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)     
    # 选择训练集和测试集    
    y_train = fillc[fillc.notnull()]    
    y_test = fillc[fillc.isnull()]    
    x_train = df_0[y_train.index, :]    
    x_test = df_0[y_test.index, :]     
    # 用随机森林填补缺失值    
    rfc = RandomForestRegressor(n_estimators=100)    
    rfc = rfc.fit(x_train, y_train)    
    y_predict = rfc.predict(x_test)     
    # 将填补好的值返回到原始的特征矩阵中    
    X_missing.loc[X_missing.iloc[:, i].isnull(), i] = y_predict
```









# 6. 概率相关模型

## 6.1 朴素贝叶斯

### 6.1.1 原理

#### 简介

##### 贝叶斯公式

根据$P(x, y) = P(x) · P(y|x) = P(y) · P(x|y)$可以得到

$$
P(x|y) = \frac{P(y|x) · P(x)}{P(y)}
$$

##### 朴素——条件独立

假设 $x$, $y$ 相对于 $z$ 是独立的，则有

$$
P(x,y|z) = P(x|z) · P(y|z)
$$

#### 优化问题

##### 极大似然估计——后验概率最大化

$$
\mathop{argmax}_{c_k}P(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k)
$$

##### 贝叶斯估计——平滑系数

$$
P_\lambda(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum_{i=1}^NI(x_i^{(j)}=a_{jl},y_i=c_k)+\lambda}{\sum_{i=1}^NI(y_i=c_k)+S_j\lambda}
$$

$\lambda$ 等于1时为拉普拉斯平滑



### 6.1.2 sklearn-API



#### 示例

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



### 6.1.3 numpy实现

 [6_1_naive_bayes.ipynb](code\6_1_naive_bayes.ipynb) 







## 6.2 EM 算法

一种用于包含隐变量概率模型参数的极大似然估计方法

### 6.2.1 原理

主要步骤

给定观测变量数据 $Y$，隐变量数据 $Z$，联合概率分布 $P(Y,Z|\theta)$ 以及关于隐变量的条件分布 $P(Z|Y,\theta)$，使用 EM 算法对模型参数 $\theta$ 进行估计

初始化模型参数 $\theta^{(0)}$

E 步：计算 Q 函数
$$
Q(\theta,\theta^{(i)})=\sum_ZlogP(Y,Z|\theta)P(Z|Y,\theta^{(i)})
$$
M 步：求使 Q 函数最大化的参数 $\theta$

重复迭代 E 步和 M 步直至收敛



### 6.2.3 代码实现

 [6_2_em.ipynb](code\6_2_em.ipynb) 



## 6.3 隐马尔可夫模型



### 6.3.3 代码实现

 [6_3_hmm.ipynb](code\6_3_hmm.ipynb) 





## 6.4 CRF条件随机场

一种能够考虑相邻时序信息的模型

词性标注就是CRF最常用的一个场景之一



### 6.4.3 代码实现

 [6_4_crf.ipynb](code\6_4_crf.ipynb) 



# 7. 聚类算法

## 7.1 k-近邻算法

K-Nearest Neighbors, KNN

### 7.1.1 原理

#### 简介

对于给定的实例数据和实例数据对应所属类别，当要对新的实例进行分类时，根据这个实例最近的 k 个实例所属的类别来决定其属于哪一类。

关键：距离度量、k值选取、归类规则

#### 算法流程

1. 根据给定的距离度量，在训练集中找出与 $x$ 最近的 $k$ 个点
2. 在这些点中，根据分类决策规则（如多数表决）决定 $x$ 的类别 $y$

#### 三要素

##### 距离度量

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

##### k值选择

- k值的影响

较小的k值会降低近似误差，但会增加估计误差，使得预测结果对近邻点较敏感，易发生过拟合

较大的k值会减小估计误差，但会增加近似误差，较远的实例也会对预测起作用

k值过小，易受到异常值影响；k值过大，易受到样本不均衡影响

- 通常采用交叉验证来选取最优k值

##### 分类决策规则

- 多数表决

#### 优化策略

线性扫描方法需要遍历所有数据，当数据集较大时非常耗时

构造kd树：不断用垂直于坐标轴的超平面将k维空间切分，构成一系列的k维超矩形区域

通常选择训练实例点在选定坐标轴上的中位数为切分点，得到平衡kd树



### 7.1.2 sklearn-API





#### 示例

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



### 7.1.3 Numpy实现

 [7_1_knn.ipynb](code\7_1_knn.ipynb) 





### 7.1.4 优缺点

#### 优点

简单，无需训练，易于实现 

#### 缺点

计算量大；k值不当则分类精度无法保证



## 7.2 线性判别分析

Linear Discriminant Analysis，LDA

### 7.2.1 原理

####  简介

一种监督学习的降维技术，将数据在低维度上进行投影，使得同一类数据尽可能接近，不同类数据尽可能疏远

<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220603190053359.png" alt="image-20220603190053359" style="zoom:50%;" />

#### 公式推导

##### 目标函数

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

##### 算法流程

1. 对数据按类别分组，分别计算每组样本的均值和协方差
2. 计算类内散度矩阵 $S_w$
3. 计算均值差 $\mu_0-\mu_1$
4. SVD方法计算类内散度矩阵的逆 $S_w^{-1}$
5. 计算投影矩阵 $w$。
6. 计算投影后的数据点 $Y = S_w^TX$



### 7.2.2 sklearn-API



#### 示例

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```



### 7.2.3 Numpy实现

 [7_2_LDA.ipynb](code\7_2_LDA.ipynb) 







### 7.2.4 与PCA之间的异同点

#### 相同点

- 两者均可以对数据进行降维

- 两者在降维时均使用了矩阵特征分解的思想。

- 两者都假设数据符合高斯分布。

#### 不同点

- LDA是有监督的降维方法，而PCA是无监督的降维方法

- LDA降维最多降到类别数k-1的维数，而PCA没有这个限制。

- LDA除了可以用于降维，还可以用于分类。

- LDA选择分类性能最好的投影方向，而PCA选择样本点投影具有最大方差的方向。



### 7.2.5 优缺点

#### 优点

- 在降维过程中可以使用类别的先验知识经验

- LDA在样本分类信息依赖均值而不是方差的时候，比PCA之类的算法较优。

#### 缺点

- LDA不适合对非高斯分布样本进行降维，PCA也有这个问题。
- LDA降维最多降到类别数k-1的维数，如果我们降维的维度大于k-1，则不能使用LDA。当然目前有一些LDA的进化版算法可以绕过这个问题。
- LDA在样本分类信息依赖方差而不是均值的时候，降维效果不好。
- LDA可能过度拟合数据









