# machineLearning III 无监督算法篇

# 1. 聚类算法

## 1.1 K-means算法

### 1.1.1 原理

#### 算法原理

1. 随机选择k个中心
2. 遍历所有样本，把样本划分到距离最近的一个中心
3. 划分之后就有K个簇，计算每个簇的平均值作为新的质心
4. 重复步骤2，直到达到停止条件
5. 停止：聚类中心不再发生变化；所有的距离最小；迭代次数达到设定值

#### 算法复杂度

k-means运用了 Lioyd’s 算法,平均计算复杂度是 O(k*n*T)，其中n是样本量，T是迭代次数。

计算复杂读在最坏的情况下为 O(n^(k+2/p))，其中n是样本量，p是特征个数。



### 1.1.2 sklearn API

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

#### 示例

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



### 1.1.3 评估指标

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



### 1.1.4 Numpy算法

 [1_1_kmeans.ipynb](unsupervised_code\1_1_kmeans.ipynb) 





## 1.2 主成分分析 PCA

### 1.2.1 原理

#### 奇异值分解

##### 定义

singular value decomposition, SVD: 矩阵因子分解方法，将一个m×n矩阵分解为m阶正交矩阵、由降序排列的非负对角线m×n矩阵和n阶正交矩阵

##### 公式

$A=U\Sigma V^T$

且有 $UU^T=I, VV^T=I$

$\Sigma=diag(\sigma_1,\sigma_2,...,\sigma_p),\sigma_1\geqslant\sigma_2\geqslant ...\geqslant\sigma_p, p=min(m,n)$

$\sigma_p$ 称为矩阵 $A$ 的奇异值

##### 截断奇异值分解

矩阵 $A$ 的秩为 $r$ ，分解后 $\Sigma$ 为 $k$ 阶对角矩阵（$k<r$）

可以起到数据压缩的作用（紧奇异值分解对应着无损压缩，截断奇异值分解则为有损压缩）

截断奇异值分解是在平方损失（弗罗贝尼乌斯范数）意义下对矩阵的最优近似

奇异值分解还可以看做将一个线性变换分解成旋转变换、缩放变换及旋转变换的组合

##### Numpy实现

```python
import numpy as np
A = np.array([[0,1],[1,1],[1,0]])
u, s, vt = np.linalg.svd(A, full_matrices=True)
```

##### 图片压缩

```python
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# 定义恢复函数，由分解后的矩阵恢复到原矩阵
def restore(u, s, v, K): 
    '''
    u:左奇异矩阵
    v:右奇异矩阵
    s:奇异值矩阵
    K:奇异值个数
    '''
    m, n = len(u), len(v[0])
    a = np.zeros((m, n))
    for k in range(K):
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        # 前k个奇异值的加总
        a += s[k] * np.dot(uk, vk)   
    a = a.clip(0, 255)
    return np.rint(a).astype('uint8')

A = np.array(Image.open("./louwill.jpg", 'r'))
# 对RGB图像进行奇异值分解
u_r, s_r, v_r = np.linalg.svd(A[:, :, 0])    
u_g, s_g, v_g = np.linalg.svd(A[:, :, 1])
u_b, s_b, v_b = np.linalg.svd(A[:, :, 2])

# 使用前50个奇异值
K = 50 
output_path = r'./svd_pic'
for k in tqdm(range(1, K+1)):
    R = restore(u_r, s_r, v_r, k)
    G = restore(u_g, s_g, v_g, k)
    B = restore(u_b, s_b, v_b, k)
    I = np.stack((R, G, B), axis=2)   
    Image.fromarray(I).save('%s\\svd_%d.jpg' % (output_path, k))
```



#### 主成分分析

principal component analysis, PCA

##### 定义

- 利用正交变换把由线性相关变量表示的观测数据转换为少数几个由线性无关变量（主成分）表示的数据

- 通常需要对数据标准化，此时主成分的协方差矩阵即为相关矩阵
- 可以通过相关矩阵的特征值分解或样本矩阵的奇异值分解进行
- 第一主成分，即方差最大的成分（包含信息最多）

##### 相关矩阵的特征值分解实现PCA

1. 去平均值
2. 计算协方差矩阵
3. 计算协方差矩阵的特征值与特征向量
4. 对特征值从大到小排序
5. 保留k个最大的特征向量
6. 将数据转换到k个特征向量构成的新空间中

##### SVD实现PCA

1. 构造新的$n×m$矩阵$X'$（每列均值为零）

$$
X'=\frac{1}{\sqrt{n-1}}X^T
$$

2. 截断奇异值分解（有 $k$ 个奇异值，矩阵 $V$ 的前 $k$ 列构成 $k$ 个样本主成分）

$$
X'=U\Sigma V^T
$$

3. 求 $k×n$ 样本主成分矩阵

$$
Y=V^TX
$$



### 1.2.2 sklearn-API

#### 参数说明

```
sklearn.decomposition.PCA(n_components=None, *, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', n_oversamples=10, power_iteration_normalizer='auto', random_state=None)
```

| 参数                       | 含义                                                         |
| -------------------------- | ------------------------------------------------------------ |
| n_components               | int/float/'mle'，保留的特征数<br />为'mle'时（且svd_solver='full'）时会自动计算保留的特征<br />0 < n_components < 1代表满足最低的主成分方差累计贡献率 |
| whiten                     | bool，降低输入数据的冗余性，使得经过白化处理的输入数据特征之间相关性较低且所有特征具有相同的方差 |
| svd_solver                 | str: 'auto', 'full', 'arpack', 'randomized'<br />'auto': 自动选择<br />'full': 使用了 scipy 库对应的实现<br />'arpack': 使用 scipy 库的 sparse SVD 实现，严格遵循0 < n_components < min(X.shape)<br />'randomized': 适用于数据量大，数据维度多同时主成分数目比例又较低的 PCA 降维（Halko et al.） |
| tol                        | float, 当svd_solver='arpack'时奇异值计算的容差               |
| iterated_power             | int, svd_solver='randomized'时计算幂方法的迭代次数           |
| n_oversamples              | int, svd_solver="randomized" 时额外随机向量的数量            |
| power_iteration_normalizer | 'auto'/'QR'/'LU'/None, 随机 SVD 求解器的幂迭代归一化器       |

#### 属性

| 属性                      | 类型                               | 含义                                                         |
| ------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| components_               | ndarray (n_components, n_features) | 含有最大方差的主成分                                         |
| explained_variance_       | ndarray (n_components,)            | 降维后的各主成分的方差值                                     |
| explained_variance_ratio_ | ndarray (n_components,)            | 降维后的各主成分的方差值占总方差值的比例（主成分方差贡献率） |
| singular_values_          | ndarray (n_components,)            | 所被选主成分的奇异值                                         |
| mean_                     | ndarray (n_components,)            | 每个特征的经验平均值，由训练集估计                           |
| n_components_             | int                                | 主成分的估计数量                                             |
| n_features_               | int                                | 训练数据中的特征数                                           |
| n_samples_                | int                                | 训练数据中的样本数量                                         |
| noise_variance_           | float                              | 噪声协方差                                                   |
| n_features_in_            | int                                | 拟合时的特征数                                               |
| feature_names_in_         | ndarray (n_components,)            | 拟合时的特征名称                                             |

#### 方法

| 方法                                    | 含义                         |
| --------------------------------------- | ---------------------------- |
| fit(X，Y=None)                          | 拟合                         |
| transform(X)                            | 降维                         |
| fit_transform(X,Y=None)                 | 拟合并降维                   |
| get_covariance()                        | 获得协方差数据               |
| get_params(deep=True)                   | 返回模型的参数               |
| set_params(**params)                    | 为这个估计器手动设定参数     |
| inverse_transform(X)                    | 将降维后的数据转换成原始数据 |
| score(X, Y=None)                        | 计算所有样本的log似然平均值  |
| score_samples(X)                        | 计算样本的log似然值          |
| get_precision()                         | 计算数据精度矩阵             |
| get_feature_names_out([input_features]) | 获取模型的输出特征名         |

#### 示例

```python
# 导入sklearn降维模块
from sklearn import decomposition
# 创建pca模型实例，主成分个数为3个
pca = decomposition.PCA(n_components=3)
# 模型拟合
pca.fit(X)
# 拟合模型并将模型应用于数据X
X_trans = pca.transform(X)
```



### 1.2.3 评估指标





### 1.2.4 Numpy实现

 [1_2_pca.ipynb](unsupervised_code\1_2_pca.ipynb) 





### 1.2.5 优缺点

#### 优点

仅仅需要以方差衡量信息量，不受数据集以外的因素影响

各主成分之间正交，可消除原始数据成分间的相互影响的因素

计算方法简单，易于实现

#### 缺点

主成分各个特征维度的含义具有一定的模糊性，不如原始样本特征的解释性强

方差小的非主成分也可能含有对样本差异的重要信息，因降维丢弃可能对后续数据处理有影响





# 2. 概率相关算法

## 2.1 潜在狄利克雷分配

Latent Dirichlet allocation, LDA

基于贝叶斯学习的话题模型，潜在语义分析、概率潜在语义分析的扩展

2002年由Blei等提出，用于文本数据挖掘、图像处理、生物信息处理等领域

### 2.2.1 原理

#### 多项分布与狄利克雷分布

##### 多项分布

- 多元离散随机变量$X=(X_1, X_2, ...X_K)$的概率函数为

$$
P(X_1=n_1, X_2=n_2,...,X_k=n_k)=\frac{n!}{\prod_{i=1}^kn_i!}\prod_{i=1}^kp_i^{n_i}
$$

其中$\sum_{i=1}^kp_i=1, \sum_{i=1}^kn_i=n$

称随机变量服从参数为$(n, p)$的多项分布，记作 $X \sim Mult(n, p)$

##### 狄利克雷分布

- 一种多元连续随机变量的概率分布

多元离散随机变量$\theta=(\theta_1, \theta_2, ...\theta_K)$的概率函数为
$$
P(\theta|\alpha_1,\alpha_2,...,\alpha_k)=\frac{\Gamma(\sum_{i=1}^k\alpha_i)}{\prod_{i=1}^k\Gamma(\alpha_i)}\prod_{i=1}^k\theta_i^{\alpha_i-1}
$$
其中$\sum_{i=1}^k\theta_i=1, \sum_{i=1}^kn_i=n$

称随机变量$\theta$服从参数为$\alpha$的狄利克雷分布，记作 $\theta \sim Dir(\alpha)$

$\Gamma(s)$是伽马函数（阶乘在实数上的推广），定义为：
$$
\Gamma(s)=\int_0^\infty{x^{s-1}e^{-x}dx}, s>0
$$

当概率分布比较均匀时，可以将$\alpha$调低（$\alpha$较低时取某个值的概率较大，较高时取相同值概率较大）

##### 二项分布和贝塔分布

- 二项分布

 多项分布的特殊情况
$$
P(X=m)=\left(\begin{array}{c} n \\ m \end{array}\right)p^m(1-p)^{n-m}
$$

- 贝塔分布

狄利克雷分布的特殊情况
$$
p(x)=\left\{ \begin{array}{cc} \frac{1}{B(s,t)x^{s-1}}(1-x)^{t-1} & 0<x<1 \\ 0 & other \end{array}\right.
$$
其中$B(s, t)=\frac{\Gamma(s)\Gamma(t)}{\Gamma(s+t)}$为贝塔函数

##### 各种分布之间的关系

<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220602213319462.png" alt="image-20220602213319462" style="zoom:50%;" />

#### 潜在狄利克雷分配模型

##### 模型假设

模型假设话题由单词的多项分布表示，文本由话题的多项分布表示，单词和话题分布的先验分布都是狄利克雷分布。

##### LDA生成文本的过程

首先随机生成一个文本的话题分布，在该文本的每个位置依据话题分布随机生成一个话题，并依据该话题的单词分布随机生成一个单词，直至生成整个文本。

##### 与PLSA（概率潜在语义分析）的异同

均假设话题是单词的多项分布、文本是话题的多项分布

LDA使用狄利克雷分布作为先验分布，PLSA不使用先验分布（或者说先验分布为均匀分布）

学习过程LDA基于贝叶斯学习，PLSA基于极大似然估计

LDA使用先验概率分布可以防止学习过程中产生的过拟合

##### 模型定义

每个话题由一个单词的条件概率分布决定，分布服从多项分布，其参数又服从狄利克雷分布，超参数为$\beta$

每个文本由一个话题的条件概率分布决定，分布服从多项分布，其参数又服从狄利克雷分布，超参数为$\alpha$

##### LDA的参数估计

###### 吉布斯抽样算法

- 特点

算法简单，但迭代次数较多

- 原理——蒙特卡洛原理

生成单词的联合概率分布为：
$$
p(\vec{w}_m,\vec{z}_m,\vec{\theta}_m,\Phi(\vec\alpha,\vec\beta))=\prod_{n=1}^{N_m}p(w_{m,n}|\vec\varphi_{z_{m,n}})p(z_{m,n}|\vec{\theta}_m)p(\vec{\theta}_m|\vec\alpha)p(\Phi|\vec\beta)
$$


<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220603085302532.png" alt="image-20220603085302532" style="zoom:30%;" />

根据 $\alpha$ 生成长度为 $K$ 的话题分布向量 $\vec\theta_m$，采样后得到某个单词位置的话题 $z_{m, n}$。

根据 $\beta$ 生成 $K$ 个长度为 $V$ （词库长度）的单词分布向量 $\vec\varphi_k$，选择第 $z_{m,n}$ 个分布，采样后得到单词 $w_{m,n}$

依次生成文本中所有单词

- 流程

输入：文本的单词序列

输出：文本的话题序列的后验概率分布的样本技术，模型参数的估计值

参数：超参数 $\alpha/\beta$，话题个数 $K$

流程：

​		初始随机给文本中每个词分配主题，然后统计每个主题下出现词的数量以及每个文本出现主题的数量

​		根据其他词的主题分布估计当前词分配各个主题的概率，得到当前词的主题分布，并采样得到新主题
$$
p(\vec{w},\vec{z}|\vec\alpha,\vec\beta)=p(\vec{w}|\vec{z},\vec\beta)p(\vec{z}|\vec\alpha)
$$
​		同样方法更新下一个词的主题

​		直至每个文本的主题分布和每个主题的词分布收敛

<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220603105009861.png" alt="image-20220603105009861" style="zoom:50%;" />

<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220603110319494.png" alt="image-20220603110319494" style="zoom:50%;" />

- 超参数的选择

交叉验证

经验估计（$\alpha=50/K, \beta=200/W$）

​	$\alpha$表达了不同文本间主题是否鲜明，$\beta$度量了有多少近义词能够属于同一个类别

迭代法：T.Minka

<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220603111816060.png" alt="image-20220603111816060" style="zoom:33%;" />

###### 变分推理——EM算法

优化较快

<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220602220727504.png" alt="image-20220602220727504" style="zoom:50%;" />



### 2.3.2 代码实现

#### gensim库实现

```python
from gensim import corpora, models, similarities

if __name__ == '___main_':
    f = open('22.LDA_test.txt')
    stop_list = set('for a of the and to in'.split())
    # texts = [line.strip().split() for line in f]
    # print(texts)
    texts = [[word for word in line.strip().lower().split() if word not in stop_list] for line in f]
    print('Text = ', texts)
    
    dictionary = corpora.Dictionary(texts)
    V = len(dictionary)
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpus_tfidf = models.TfidfModel(corpus)[corpus]

    print('TF-IDF: ')
    for c in corpus_tfidf:
        print(c)

    print('An LDA Mode1: ')
    num_topics = 2
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary, \
        alpha='auto', eta='auto', minimum_probability=0.001)
    doc_topic = [doc_t for doc_t in lda[corpus_tfidf]]

    print('Document-Topic: ')
    print(doc_topic)
    for doc_topic in lda.get_document_topics(corpus_tfidf):
        print(doc_topic)
    for topic_id in range(num_topics):
        print('Topic', topic_id)
        #print(lda.get_topic_terms(topicid=topic_id))
        print(lda.show_topic(topic_id))
    
    similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])
    print('similarity: ')
    print(list(similarity))
```



#### numpy实现

[LDA主题模型试验_LJBlog2014的博客-CSDN博客](https://blog.csdn.net/LJBlog2014/article/details/50539253)



### 2.2.3 优缺点

由于加入主题概念，可以解决一词多义和多次一义问题；可以看成降维/聚类过程

对短文本效果不好

