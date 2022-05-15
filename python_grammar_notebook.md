# Python基础篇

## 01_数据类型

### 1. 变量

#### 变量命名

- 只能包括数字、字母、下划线，但不能以数字开头
- 不能与python的关键字重名

<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220512154939122.png" alt="image-20220512154939122" style="zoom:75%;" />

- 推荐命名：驼峰式命名法，如MyName，或myName

#### 变量赋值

```python
a = 3
b = 'zyt'
c = 4.6
d = True
```

#### 变量的数据类型

- 基本类型

  int  整型

  float  浮点型

  str  字符型

  bool  布尔型（True/False）

- 复杂类型

  list  列表

  tuple  元组

  dict  字典

  set  集合



### 2. 字符串

#### 定义

引号：单引号、双引号、三引号（三引号用来表示注释）

#### 字符串的索引和切片

- 索引均从零开始
- 切片包含冒号左边，但不包含冒号右边
- 两个冒号，最后一个冒号表示的间隔，如果是负数，则表示倒着的间隔

```python
a = 'xgglovebb'
print(a[5])
print(a[1:2])
print(a[3:])
print(a[:4])
print(a[1:6:2])
print(a[-2])
print(a[6:1:-1])
```

#### 转义字符

| 转义字符 | 含义                             |
| -------- | -------------------------------- |
| \        | 续行符（一行展示不下，换行展示） |
| \\\      | 反斜杠符号                       |
| \\'      | 单引号                           |
| \\"      | 双引号                           |
| \n       | 换行                             |
| \t       | 横向制表符                       |
| \r       | 回车                             |
| \other   | 其它的字符以普通格式输出         |

#### 字符串运算：拼接、重复、是否包含

```python
a, b = 'xgg', 'bb'
print(a + b)
print(a * 2)
print('x' in a)
print('b' not in b)
```

#### 格式化字符串

| 格式化符号 | 含义                                 |
| ---------- | ------------------------------------ |
| %s         | 格式化字符串                         |
| %d         | 格式化整数                           |
| %f         | 格式化浮点数字，可指定小数点后的精度 |

```python
a = 'zhuyating'
b = 8
print('%s is %d years old.' % (a, b))
print(a + ' is ' + str(b) + ' years old.')
print('{} is {} years old.'.format(a, b))
```

#### 字符串内置函数

- python的内置函数：函数名（变量名）—— print()、input()、type()、len()

- 判断型

| 内置函数                                                     | 含义                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| string.isalnum()                                             | 如果 string 至少有一个字符并且所有字符都是字母或数字则返 回 True,否则返回 False |
| string.isalpha()                                             | 如果 string 至少有一个字符并且所有字符都是字母则返回 True, 否则返回 False |
| string.isdigit()                                             | 如果 string 只包含数字则返回 True 否则返回 False.            |
| string.islower() / string.isupper()                          | 如果 string 都是小/大写，是则返回 True，否则返回 False       |
| string.isnumeric()                                           | 如果 string 中只包含数字字符，则返回 True，否则返回 False    |
| string.isspace()                                             | 如果 string 中只包含空格，则返回 True，否则返回 False.       |
| string.istitle()                                             | 如果 string 是标题化的，则返回 True，否则返回 False          |
| string.endswith(obj, beg=0, end=len(string)) / string.startswith(obj, beg=0,end=len(string)) | 检查字符串是否以 obj 结束/开始，如果beg 或者 end 指定则检查指定的范围内是否以 obj 结束/开始，如果是，返回 True,否则返回 False. |

- 大小写

| 内置函数                        | 含义                                  |
| ------------------------------- | ------------------------------------- |
| string.capitalize()             | 把字符串的第一个字符大写              |
| string.lower() / string.upper() | 转换 string 中所有大写字符为小写/大写 |
| string.swapcase()               | 翻转 string 中的大小写                |

- 统计

| 内置函数                                  | 含义                                                         |
| ----------------------------------------- | ------------------------------------------------------------ |
| string.count(str, beg=0, end=len(string)) | 返回 str 在 string 里面出现的次数，如果 beg 或者 end 指定则返回指定范围内 str 出现的次数 |
| max(str)/min(str)                         | 返回字符串 *str* 中最大/小的字母。                           |

- 编码

| 内置函数                                         | 含义                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ |
| string.decode(encoding='UTF-8', errors='strict') | 以encoding指定的编码格式解码string，如果出错默认报一个ValueError的异常，除非errors指定的是'ignore'或者'replace' |
| string.encode(encoding='UTF-8', errors='strict') | 以 encoding 指定的编码格式编码 string，如果出错默认报一个ValueError 的异常，除非 errors 指定的是'ignore'或者'replace' |
| string.format()                                  | 格式化字符串                                                 |

- 查找替换

| 内置函数                                            | 含义                                                         |
| --------------------------------------------------- | ------------------------------------------------------------ |
| string.find(str, beg=0, end=len(string))            | 检测 str 是否包含在 string 中，如果 beg 和 end 指定范围，则检查是否包含在指定范围内，如果是返回开始的索引值，否则返回-1 |
| string.index(str, beg=0, end=len(string))           | 跟find()方法一样，只不过如果str不在 string中会报一个异常.    |
| string.rfind(str, beg=0,end=len(string) )           | 类似于 find() 函数，返回字符串最后一次出现的位置，如果没有匹配项则返回 -1。 |
| string.rindex( str, beg=0,end=len(string))          | 类似于 index()，不过是返回最后一个匹配到的子字符串的索引号。 |
| string.replace(str1, str2,  num=string.count(str1)) | 把 string 中的 str1 替换成 str2,如果 num 指定，则替换不超过 num 次. |

- 删除转换

| 内置函数                         | 含义                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| string.lstrip()                  | 截掉 string 左边的空格                                       |
| string.rstrip()                  | 删除 string 字符串末尾的空格.                                |
| string.strip([obj])              | 在 string 上执行 lstrip()和 rstrip()                         |
| string.translate(str, del="")    | 根据 str 给出的表(包含 256 个字符)转换 string 的字符, 要过滤掉的字符放到 del 参数中 |
| string.maketrans(intab, outtab]) | maketrans() 方法用于创建字符映射的转换表，对于接受两个参数的最简单的调用方式，第一个参数是字符串，表示需要转换的字符，第二个参数也是字符串表示转换的目标。 |

- 格式调整

| 内置函数                     | 含义                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| string.expandtabs(tabsize=8) | 把字符串 string 中的 tab 符号转为空格，tab 符号默认的空格数是 8。 |
| string.ljust(width)          | 返回一个原字符串左对齐,并使用空格填充至长度 width 的新字符串 |
| string.rjust(width)          | 返回一个原字符串右对齐,并使用空格填充至长度 width 的新字符串 |
| string.zfill(width)          | 返回长度为 width 的字符串，原字符串 string 右对齐，前面填充0 |

- 其他

| 内置函数                                    | 含义                                                         |
| ------------------------------------------- | ------------------------------------------------------------ |
| string.split(str="", num=string.count(str)) | 以 str 为分隔符切片 string，如果 num 有指定值，则仅分隔 num+1 个子字符串 |
| string.splitlines([keepends])               | 按照行('\r', '\r\n', \n')分隔，返回一个包含各行作为元素的列表，如果参数 keepends 为 False，不包含换行符，如果为 True，则保留换行符。 |
| string.join(seq)                            | 以 string 作为分隔符，将 seq 中所有的元素(的字符串表示)合并为一个新的字符串 |
| string.partition(str)                       | 有点像 find()和 split()的结合体,从 str 出现的第一个位置起,把 字 符 串 string 分 成 一 个 3 元 素 的 元 组 (string_pre_str,str,string_post_str),如果 string 中不包含str 则 string_pre_str == string. |
| string.rpartition(str)                      | 类似于 partition()函数,不过是从右边开始查找                  |



### 3. 序列

python中的序列包括字符串、列表、元组、集合和字典

#### 均支持以下操作

- 索引

sname[x]

索引从零开始

- 切片

sname[start : end : step]

- 相加和相乘

```python
a = [1, 2, 3]
b = [4, 5, 6]
print(a + b)
print(a * 2)
```

- 检查元素是否包含在序列中

```python
print(1 in a)
print(1 not in a)
```

字典的 in 操作只能用来匹配 key

- 内置函数

| 内置函数    | 含义                                                         |
| ----------- | ------------------------------------------------------------ |
| len()       | 计算序列的长度，即返回序列中包含多少个元素。                 |
| max()       | 找出序列中的最大元素。                                       |
| min()       | 找出序列中的最小元素。                                       |
| sum()       | 计算元素和，做加和操作的必须都是数字，不能是字符或字符串，否则该函数将抛出异常 |
| sorted()    | 对元素进行排序。                                             |
| reversed()  | 反向序列中的元素。数据类型是发生变化的                       |
| enumerate() | 将序列组合为一个索引序列，多用在 for 循环中。                |
| list()      | 将序列转换为列表。                                           |
| str()       | 将序列转换为字符串。                                         |
| tuple()     | 将序列转化为元组                                             |
| set()       | 将序列转化为集合                                             |
| dict()      | 将序列转化为字典                                             |

#### 例外

集合和字典不支持索引、切片、相加和相乘操作



### 4. 列表

#### 定义列表

两种方法

```python
# 赋值号定义
a = [1, 2, 3, 'xgg', 5.0, True]
# list()函数定义
b = list('abcd')
c = list(range(4))
```

#### 列表的增删改查

- 添加元素——四种方法

```python
# 1. + 号 —— 生成一个新的列表，原有的列表不会被改变
d = b + c
# 2. append()函数 —— 末尾追加元素、元组或列表（整体添加）
a.append(5)
# 3. extend()函数 —— 末尾追加元素、元组或列表（逐个添加）
b.extend(['e', 'f'])
# 4. insert()函数 —— 指定位置添加元素、列表或元组（整体添加）
c.insert(2, 9) # (索引，值)
```

- 删除

删除整个列表：del listname —— 结果会删除变量

删除元素：四种方法

```python
# 1. del 语句
del a[2]
del b[2:3]
# 2. pop()函数 —— 根据索引删除
c.pop(2)
d.pop()   # 不指定index时默认删除最后一个元素
# 3. remove()函数 —— 根据值删除
e = [9, 8, 7]
e.remove(2)
# 4. clear()函数 —— 删除列表所有元素，结果为空列表
e.clear()
```

- 修改

```python
a[2] = 2
b[1:3] = ['a', 'b']
```

- 查询

索引：listname[i]

切片：listname[start : end : step]

查找位置：listname.index(obj, start, end) —— 返回索引值

- 其他

排序：list.sort(key=None, reverse=False) —— 升序排列

反转：list.reversed()

### 5. 元组

#### 创建元组

```python
s = (1, 2, 3)
t = tuple('abcd')
```

#### 和列表的区别

元组元素不可变！！！



### 6. 字典

#### 特点

- 键值对形式存在
- 无序序列
- 字典是可变的，可以修改
- 键必须唯一
- 键必须不可变：只能使用数字、字符串或者元组

#### 创建字典

```python
# 1. {}
scores = {'数学': 95, '英语': 92, '语文': 84}
# 2. dict.fromkeys(key, value)
knowledge = ['语文', '数学', '英语']
scores = dict.fromkeys(knowledge, 60)   # 所有的 value 都一样
# 3. dict()函数
a = dict(str1=value1, str2=value2, str3=value3)
b = dict([('two',2), ('one',1), ('three',3)])
c = dict([['two',2], ['one',1], ['three',3]])
d = dict((('two',2), ('one',1), ('three',3)))
e = dict((['two',2], ['one',1], ['three',3]))
# 4. dict(zip())
keys = ['one', 'two', 'three']
values = [1, 2, 3]
a = dict(zip(keys, values))
```

#### 增删改查

- 删除

删除字典

```python
del dictname
dictname.clear()
```

删除字典元素：三种方法

```python
del dictname[key]
dictname.pop(key)
dictname.popitem()  # 随机删除
```

- 增加

```python
dictname[key] = value
dictname.update(dictname2)
```

- 修改

key 的名字不能被修改，只能修改 value

```python
dictname[key] = value
```

- 查找

```python
dictname[key] # 返回 key 所对应的 value
dictname.get(key[,default]) # default 用于指定要查询的键不存在时，此方法返回的默认值，如果不手动指定，会返回 None
dictname.keys()  # 返回字典中所有的 keys
dictname.values()  # 返回字典中所有的 values
dictname.items()  # 返回字典中所有的键值对（key-value）
```

### 7. 集合

#### 创建集合

```python
# 1. {}创建
a = {1,'c',1,(1,2,3),'c'}
# 2. set()函数创建
print(set([1,2,3,4,5]))
# 3. 集合推导式
set4 = {num for num in range(1, 100) if num % 3 == 0 or num % 5 == 0}
```

#### 增删改查

- 查询

无法使用下标查询（集合无序），一般使用循环

- 删除

del setname —— 删除集合

setname.remove(element) —— 删除集合中指定元素

setname.discard(element) —— 删除集合中指定元素

setname.pop() —— 从集合中“随机”删除某个元素

- 添加

setname.add(element)

setname.update([list])

#### 集合运算

| 运算符   | 符号 | 含义                              |
| -------- | ---- | --------------------------------- |
| 交集     | &    | 取两集合公共的元素                |
| 并集     | \|   | 取两集合全部的元素                |
| 差集     | -    | 取一个集合中另一集合没有的元素    |
| 对称差集 | ^    | 取集合 A 和 B 中不属于 A&B 的元素 |
| 子集     | <=   | 判断左集合是否为右集合的子集      |
| 超集     | >=   | 判断左集合是否为右集合的超集      |

```
set1.intersection(set2)  # 交集
set1.union(set2)  # 并集
set1.difference(set2)  # 差集
set1.symmetric_difference(set2)  # 对称差集
set2.issubset(set1)  # 子集
set1.issuperset(set2)  # 超集
```



## 02_流程控制语句

### 1. 条件语句

#### if 语句

```python
age = int(input("请输入你的年龄：") )
if age < 18 :
    print("你还未成年，建议在家人陪同下使用该软件！")
```

#### if-else 语句

```python
b = False
if b:
    print('b是True')
else:
    print('b是False')
```

#### if-elif-else 语句

```python
height = float(input("输入身高（米）："))
weight = float(input("输入体重（千克）："))
bmi = weight / (height * height)  #计算BMI指数
if bmi < 18.5:
    print("BMI指数为：" + str(bmi))
    print("体重过轻")
elif bmi >= 18.5 and bmi < 24.9:
    print("BMI指数为："+str(bmi))
    print("正常范围，注意保持")
elif bmi>=24.9 and bmi<29.9:
    print("BMI指数为："+str(bmi))
    print("体重过重")
else:
    print("BMI指数为："+str(bmi))
    print("肥胖")
```

#### if 后面的表达式

当 if 后面的条件为如下元素时，也表示 False

- "" # 空字符串
- [ ] # 空列表
- ( ) # 空元组
- { } # 空字典
- None # 空值
- 0  # 零值
- 不带返回值的函数：对于没有 return 语句的函数，返回值为空，也即 None

#### if 语句的嵌套

```python
proof = int(input("输入驾驶员每 100 mL 血液酒精的含量："))
if proof < 20:
    print("驾驶员不构成酒驾")
else:
    if proof < 80:
        print("驾驶员已构成酒驾")
    else:
        print("驾驶员已构成醉驾")
```

#### pass 语句

表示忽略

#### assert 语句

- 格式：assert 表达式, 注释
- 相当于以下条件语句

```python
if 表达式 == True:
    程序继续执行
else:
    程序报 AssertionError 错误
```



### 2. 循环语句

#### while 语句

注意对条件变量的更新，避免死循环

通常是在满足一定条件但不确定执行多少次的时候

```python
# 循环的初始化条件
num = 1
# 当 num 小于100时，会一直执行循环体
while num < 100:
    print("num=", num)
    # 迭代语句
    num += 1
print("循环结束!")
```

#### for 语句

- 语法格式

for 迭代变量 in 字符串|列表|元组|字典|集合：代码块

```python
letter = "xgg love xbb"
#for循环，遍历 add 字符串
for e in letter:
    print(e, end="")
```

- 应用1 —— 遍历列表/元组/字典

```python
my_list = [1,2,3,4,5]
for ele in my_list:
    print('ele =', ele)
```

- 应用2 —— 数值循环

```python
result = 0
#逐个获取从 1 到 100 这些值，并做累加操作
for i in range(101):
    result += i
print(result)
```

#### 循环嵌套

```python
i = 0
while i<10:
    for j in range(10):
        print("i=",i," j=",j)       
    i=i+1
```

#### break 语句

跳出循环

```python
# 获取字符串逗号前的所有单词
letter = 'Hello, Mary!'
for i in letter:
		if i == ',':
				break
    print(i)
```

#### continue 语句

跳出当前循环，继续下一次循环

```python
# 获取列表中不能被2整除的所有元素
list1 = [1, 4, 6, 3, 7, 8]
for i in list1:
    if i % 2 == 0:
				continue
		print(i)
```



## 03_函数

### 不带参数的函数

- 函数结构

```python
# 函数定义
def <函数名>(<形式参数>):
		<函数体>

# 函数调用
<函数名>（<实际参数>）
```

- 示例

```python
# 函数定义
def my_print():
		"""
		输出'Hello, my friend' （三引号通常表示注释）
		"""
		print('Hello, my friend')

# 函数调用
my_print()
```

### 函数参数

```python
# 函数定义
def odd_or_even(num):
		"""
		判断数字是奇数还是偶数
		"""
		if num % 2 == 0:
				print('偶数')
		else:
				print('奇数')

# 函数调用
odd_or_even(3)
odd_or_even(4)
```

- 形参和实参

形式参数：在定义函数时，函数名后面括号中的参数就是形式参数

实际参数：在调用函数时，函数名后面括号中的参数称为实际参数

实参和形参的区别，就如同剧本选主角，剧本中的角色相当于形参，而演角色的演员就相当于实参

### 函数返回值

```python
def my_len(str):
		"""
		计算字符串的长度
		"""
    length = 0
    for c in str:
       length += 1
    return length

length = my_len("xgglovexbb")
print(length)
```

### 位置参数

实参和形参位置必须一致

```python
def area(height, width):
    return height * width / 2
print(area(4,3))
```

### 关键字参数和默认参数

使用关键字参数调用时，可以任意调换参数传参的位置

调用函数时若没有给拥有默认值的形参传递参数，该参数可以直接使用定义函数时设置的默认值

当定义一个有默认值参数的函数时，**有默认值的参数必须位于所有没默认值参数的后面**。

关键字参数——调用时候使用

1. 关键字参数必须放在位置参数之后；默认参数——定义时候使用

1. 默认参数定义时必须放在位置参数之后
2. 默认参数调用时可以不给，但给了新的值则会用新值赋值

```python
def func1(a, b, c=0, d=0):
		"""
		输出 a ^ b + c * d
		"""
		print(a ** b + c * d)

# 函数调用
func1(b=1, a=3)
func1(2, 3, c=3)
func1(2, 3, c=3, b=3)
```

### 可变参数

- *args

```python
# * 表示将元素放在一个容器中
a, *b, c = 1，2，3, 4, 5
print(a)
print(b) # 元素放入列表中
print(c)
def my_print(a, b, *args):
		"""打印函数"""
		print(a)
		print(b)
		print(*args)  # 以元组形式输入
```

- **kwargs

函数调用时必须传递关键字参数

```python
def show_book(**kwargs):
		print(kwargs)  # {} 以字典形式输入

show_book()
show_book(bookname='西游记', author='吴承恩', number=5)
show_book(bookname='西游记')
show_book(bookname='西游记', author='吴承恩')
```

### lambda 匿名函数

函数体可以用一行简单的表达式表示

```python
# 求平方函数
my_square = lambda x : x * x
print(my_square(2))  # 4

def my_square(x):
	return x * x

# 求和函数
my_sum = lambda a, b : a + b
print(my_sum(2, 3))
```



## 04_面向对象编程

### 1. 类和对象

- 类（Class）**:** 用来描述具有相同的属性和方法的对象的集合。它定义了该集合中每个对象所共有的属性和方法。
- 对象：类的实例（如对象 object）。
- 方法：对象可调用的函数（可通过 object.func() 调用）。
- 属性：对象固有的属性（可通过 object.attr() 获取）。

```python
>>> a = [1, 3, 2, 4]
>>> type(a)
<class 'list'>
>>> a.index(3)
1
import matplotlib.pyplot as plt

image = plt.imread('./a.png')
print(type(image))
print(image.shape)

>>> <class 'numpy.ndarray'>
>>> (100, 100, 4)
```

- 已经接触的类：字符串、列表、Nonetype等

### 2. 类的创建

创建类的格式：

```python
class EasyClass():
    """创建一个xxx类"""
    def func(self, y):
        """这是类方法，可被类的实例对象调用"""
        print(y)
```

类名称首字母要大写，建议驼峰式命名

### 3. 类的基本组成

类属性、对象属性、方法

```python
class Family():
    """创建一个家庭类"""
    description = '这是相亲相爱的一家人'        # 类属性
    family_number = 0
    
    def __init__(self, name, gender, age):
        """初始化方法，定义实例的初始化属性"""
        self.name = name        # 对象属性
        self.gender = gender
        self.age = age
        Family.family_number += 1

    def eat(self, food):
        """对象方法"""
        print('{}爱吃{}'.format(self.name, food))
    
    def play(self, place):
        print('{}爱去{}'.format(self.name, place))

# 输出类属性
print('Family_description:', Family.description)

# 实例化类对象（附带初始化属性）
husband = Family('xgg', 'male', 27)
wife = Family('xbb', 'female', 27)
print('Family_number:', Family.family_number)

daughter = Family('czx', 'female', 5)
son = Family('clz', 'male', 3)
print('Family_number:', Family.family_number)

# 输出对象属性
print('husband_gender:', husband.gender)
print('wife_age:', wife.age)
print('daughter_name:', daughter.name)

# 可以对对象属性进行修改
daughter.age = 6   
print('daughter_age:', daughter.age)

# 调用对象方法
husband.eat('麦当劳')
wife.play('看展览')
```

类属性可以被类和实例化对象调用；对象属性只能被实例化对象调用。

**init**()方法为初始化方法，定义对象属性

self 是所有方法的第一个“参数”

### 4. 类的继承

```python
# 定义父类
class Parent():        
    parentAttr = 100
    def __init__(self):
        print("调用父类构造函数")

    def parentMethod(self):
        print('调用父类方法')

    def setAttr(self, attr):
        Parent.parentAttr = attr

    def getAttr(self):
        print("父类属性:", Parent.parentAttr)

# 定义子类
class Child(Parent): 
    def __init__(self):
        print("调用子类构造方法")

    def childMethod(self):
        print('调用子类方法')

c = Child()          # 实例化子类
c.childMethod()      # 调用子类的方法
c.parentMethod()     # 调用父类方法
```

如果子类和父类定义了相同的属性或方法，则实例化子类对象执行子类方法时使用的是子类的属性及方法

## 05_异常处理

### 1. Python常见异常类型

AssertionError：当 assert 关键字后的条件为假时，程序运行会停止并抛出 AssertionError 异常

AttributeError：当试图访问的对象属性不存在时抛出的异常

IndexError：索引超出序列范围会引发此异常

KeyError：字典中查找一个不存在的关键字时引发此异常

NameError：尝试访问一个未声明的变量时，引发此异常

TypeError：不同类型数据之间的无效操作

ZeroDivisionError：除法运算中除数为 0 引发此异常

### 2. 异常捕获

#### try语句

```python
try:
    可能产生异常的代码块
except [ (Error1, Error2, ... ) [as e] ]:
    处理异常的代码块1
except [ (Error3, Error4, ... ) [as e] ]:
    处理异常的代码块2
except  [Exception]:
    处理其它异常
try:
    a = int(input("输入被除数："))
    b = int(input("输入除数："))
    c = a / b
    print("您输入的两个数相除的结果是：", c )
except (ValueError, ArithmeticError):
    print("程序发生了数字格式异常、算术异常之一")
except :
    print("未知异常")
print("程序继续运行")
```

#### 获取特定异常的有关信息

- args：返回异常的错误编号和描述字符串；
- str(e)：返回异常信息，但不包括异常信息的类型；
- repr(e)：返回较全的异常信息，包括异常信息的类型。

```python
try:
    1/0
except Exception as e:
    # 访问异常的错误编号和详细信息
    print(e.args)
    print(str(e))
    print(repr(e))
```

#### else语句和finally语句

- else语句

只有当 try 块没有捕获到任何异常时，才会执行 else 语句；若 try 块捕获到异常则执行 except

- finally 语句

通常用作 try 块中程序的扫尾工作；无论 try 块是否发生异常 finally 语句都会执行

在某些情况下，当 try 块中的程序打开了一些物理资源（文件、数据库连接等）时，由于这些资源必须手动回收，而回收工作通常就放在 finally 块中

```python
try:
    a = int(input("请输入 a 的值:"))
    print(20/a)
except:
    print("发生异常！")
else:
    print("执行 else 块中的代码")   
finally :
    print("执行 finally 块中的代码")
```

### 3. 引发异常

#### raise语句

```python
raise [exceptionName [(reason)]]
```

事实上，raise 语句引发的异常通常用 try except（else finally）异常处理结构来捕获并进行处

```python
try:
    a = input("输入一个数：")
    #判断用户输入的是否为数字
    if(not a.isdigit()):
        raise ValueError("a 必须是数字")
except ValueError as e:
    print("引发异常：",repr(e))
```



## 06_模块和包

### 模块 Module

标准模块：time, os, random, math, datetime

第三方模块：numpy, pandas

安装第三方模块

```bash
pip install module
pip install module==x.x.x
pip uninstall module  # 卸载
```

### 导入模块

#### import 语句

- 语法1

```python
import 模块名1 [as 别名1], 模块名2 [as 别名2]，…
```

- 语法2

```python
from 模块名 import 成员名1 [as 别名1]，成员名2 [as 别名2]，…
```

从模块中导致指定成员

也可以导入全部成员

```python
form 模块名 import ＊
```

#### 导入自定义模块

导入`demo.py`文件中的 test() 函数

```python
from demo import test
```

#### **if** **name** == '**main**'

Python 内置的 **name** 变量。当直接运行一个模块时，**name** 变量的值为 **main**；而将模块被导入其他程序中并运行该程序时，处于模块中的 **name** 变量的值就变成了模块名。

```python
# demo.py
from demo2 import func2
def func1():
	print('function1')

if __name__ == '__main__':
	func1()
	func2()
# demo2.py
from demo import func1
def func2():
	print('function2')

if __name__ == '__main__':
	func1()
	func2()
```

#### 导入模块时的路径问题

- 通常情况下，当使用 import 语句导入模块后，Python 会按照以下顺序查找指定的模块文件：

1. 在当前目录，即当前执行的程序文件所在目录下查找；
2. 到 PYTHONPATH（环境变量）下的每个目录中查找；
3. 到 Python 默认的安装目录下查找。

- 方法1——临时添加模块完整路径

将模块文件的存储位置临时添加到 sys.path 变量

```python
import sys
sys.path.append('D:\\\\python_module')
```

- 方法2——使用相对路径

```python
from _dir._path import demo
```



# Python进阶篇

## 01_高级函数

### 1. map函数

**语法：**map(function, iterable, ...)

- function：函数
- iterable：一个或多个序列

接收两个参数，一个是函数，一个是序列，map将传入的函数依次作用到序列的**每个元素**。

```python
list(map(abs,[-1,3,-5,8]))
>>> [1, 3, 5, 8]

# 使用 lambda 匿名函数
list(map(lambda x: x ** 3, [1, 2, 3, 4, 5]))
>>> [1, 8, 27, 64, 125]

# 提供了两个列表，对相同位置的列表数据进行相加
list(map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10]))
>>> [3, 7, 11, 15, 19]
```

### 2. reduce函数

**语法：**reduce(function,iterable[,initial])

- function：函数
- iterable：一个或多个序列
- initial：表示初始值

reduce方法，顾名思义就是减少，假设你有一个由数字组成的可迭代对象，并希望将其缩减为单个值。把一个函数作用在一个序列上，这个函数必须接收两个参数，reduce把结果继续和序列的下一个元素做累积计算

```python
from functools import reduce
nums = [6,9,4,2,4,10,5,9,6,9]
print(nums)
>>> [6, 9, 4, 2, 4, 10, 5, 9, 6, 9]
# 累加
print(sum(nums))
>>> 64
print(reduce(lambda x,y:x+y, nums, initial=3))
>>> 64
# 累计减法
reduce(lambda x,y:x-y,[1,2,3,4])
>>> -8
#累计乘法
def multi(x,y):
    return x*y
reduce(multi,[1,2,3,4])
>>> 24
reduce(lambda x,y:x*y,[1,2,3,4])
>>> 24
```

### 3. filter函数

**语法：**filter(function, iterable)

- function：判断函数。
- iterable ：可迭代对象。

**filter()** 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。

```python
fil = filter(lambda x: x>10, [1,11,2,45,7,6,13])
fil 
>>>  <filter at 0x28b693b28c8> # 可迭代对象，不能直接查看

list(fil)
>>> [11, 45, 13]

def isodd(num):
    if num % 2 == 0:
        return True
    else:
        return False
list(filter(isodd,range(1,14)))
>>> [2, 4, 6, 8, 10, 12]
```

### 4. sorted函数

**语法：**sorted(iterable,  key=None, reverse=False)

- iterable:可迭代对象。
- key: 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
- reverse: 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。

**sort 与 sorted 区别**：

sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作；list 的 sort 方法返回的是对已经存在的列表进行操作，无返回值，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。

```python
a = [5,7,6,3,4,1,2]
b = sorted(a)   # 保留原列表
a 
>>> [5, 7, 6, 3, 4, 1, 2]
b
>>> [1, 2, 3, 4, 5, 6, 7]

# 利用key
L=[('b',2),('a',1),('c',3),('d',4)]
sorted(L, key=lambda x:x[1])  
>>> [('a', 1), ('b', 2), ('c', 3), ('d', 4)]

# 降序排列
a = [1,4,2,3,1]
sorted(a,reverse=True) 
>>> [4, 3, 2, 1, 1]
c = [[1, 4, 7], [2, 3, 6], [4, 1, 2]]
print(sorted(c, key=lambda x:x[0]))
print(sorted(c, key=lambda x:x[1]))
print(sorted(c, key=lambda x:x[2]))

c = [{'name':'cj', 'age':10}, {'name':'zyt', 'age':8}]
print(sorted(c, key=lambda x:x['name']))
print(sorted(c, key=lambda x:x['age']))
```



## 02_引用、赋值、浅拷贝与深拷贝

### 1. 基本概念

#### 对象

Python中一切皆对象，可以是变量、函数、类

对象具有三个属性：id、type、value

```python
a = 123
print(id(a))
print(type(a))
```

### 可变对象与不可变对象

可变对象：列表、字典、集合（可变对象的值能改变，但是身份不变）

不可变对象：数字、字符串、元组（身份和值都不能改变）

```python
a = 123
print('id(a)-1', id(a))
a = 234
print('id(a)-2', id(a))

b = [1, 2, 3]
print('id(b)-1', id(b))
b.append(5)
print('id(b)-2', id(b))
b = [1, 2, 3, 4]
print('id(b)-3', id(b))
```

当新创建的对象被关联到原来的变量名，旧对象被丢弃，垃圾回收器会在适当的时机回收这些对象

### 2. 引用

每个对象都会在内存中申请开辟一块空间来保存该对象，该对象在内存中所在位置的地址被称为引用。在开发程序时，所定义的变量名实际就对象的地址引用。

```python
c = 18
print('id(c)', id(c))
print('id(18)', id(18))
```

### 3. 赋值

赋值的本质就是让多个变量同时引用同一个对象的地址。

#### 不可变对象的赋值

在内存中开辟一片空间指向新的对象，原不可变对象不会被修改

```python
d = 1
e = d
print('d, e', d, e)
print('id(d), id(e)', id(d), id(e))
d = 2
print('d, e', d, e)
print('id(d), id(e)', id(d), id(e))
```

#### 可变对象的赋值

对可变对象进行赋值时，只是将可变对象中保存的引用指向了新的对象

```python
f = [1, 2, 3]
g = f
print('f, g', f, g)
print('id(f), id(g)', id(f), id(g))
f[0] = 2
print('f, g', f, g)
print('id(f), id(g)', id(f), id(g))
```

### 4. 浅拷贝

拷贝一份副本，但原变量改变时，副本未必不会改变

#### 不可变对象的浅拷贝

让多个对象同时指向一个引用，和对象的赋值没区别

```python
import copy
h = 3
i = copy.copy(h)
print('h, i', h, i)
print('id(h), id(i)', id(h), id(i))
h = 4
print('h, i', h, i)
print('id(h), id(i)', id(h), id(i))
```

#### 可变对象的浅拷贝

可变对象的拷贝，会在内存中开辟一个新的空间来保存拷贝的数据。当再改变之前的对象时，对**拷贝之后的对象中的不可变元素**没有任何影响

```python
import copy
j = [1, 2, 3]
k = copy.copy(j)
print('j, k', j, k)
print('id(j), id(k)', id(j), id(k))
j[0] = 3
print('j, k', j, k)
print('id(j), id(k)', id(j), id(k))
```

浅拷贝在拷贝时，只拷贝第一层中的引用，如果元素是可变对象，并且被修改，那么拷贝的对象也会发生变化

```python
import copy
j = [1, 2, 3, [1, 2, 3]]
k = copy.copy(j)
print('j, k', j, k)
print('id(j), id(k)', id(j), id(k))
j[3].append(2)
print('j, k', j, k)
print('id(j), id(k)', id(j), id(k))
```

### 5. 深拷贝

```python
import copy
j = [1, 2, 3, [1, 2, 3]]
k = copy.deepcopy(j)
print('j, k', j, k)
print('id(j), id(k)', id(j), id(k))
j[3].append(2)
print('j, k', j, k)
print('id(j), id(k)', id(j), id(k))
```



## 03_文件操作

### 1. 打开及关闭文件

#### open()函数

用python读取文件（如txt、csv等），第一步要用open函数打开文件，它会返回一个文件对象，这个文件对象拥有read、readline、write、close等方法。

#### 语法：

open(file, mode)

- file：需要打开的文件路径
- mode（可选）：打开文件的模式，如只读、追加、写入等

#### **mode常用的模式：**

- r：表示文件只能读取
- w：表示文件只能写入（原有的东西会被删除）
- a：表示打开文件，在原有内容的基础上追加内容，在末尾写入
- r+/w+：表示可以对文件进行读写双重操作
- b：以二进制读取文件，可加在上述各种模式之后，如 rb

#### close()函数

关闭打开的文件

```python
file = 'C:/Users/27110/Desktop/lesson_file00_case00.txt'
f = open(file) # 打开文件
f.close() # 关闭文件
```

### 2. 读写操作

#### read()函数

read()会读取一些数据并将其作为字符串（在文本模式下）或字节对象（在二进制模式下）返回。

语法：f.read(size)

参数 size（可选）为数字，表示从已打开文件中读取的字节计数，默认情况下为读取全部

```python
file = 'C:/Users/27110/Desktop/lesson_file00_case00.txt'
f = open(file, 'r')
content = f.read()
print(content)
f.close()
```

#### **readline()函数**

从文件中读取整行，包括换行符'\n’

语法：f.readline(size)

```python
file = 'C:/Users/27110/Desktop/lesson_file00_case00.txt'
f = open(file, 'r')
print(f.readline())
print(f.readline(3))
print(f.readline())
f.close()
```

#### **readlines()函数**

读取所有行，返回的是所有行组成的列表，没有参数

#### **write()函数**

将字符串写入到文件里

语法：f.write(str)

```python
file = 'C:/Users/27110/Desktop/lesson_file00_case01.txt'
f = open(file, 'w')
f.write('This is the second line.')
f.write('This is the third line.')
f.close()
file = 'C:/Users/27110/Desktop/lesson_file00_case02.txt'
f = open(file, 'a')
f.write('This is the second line.\\n')
f.write('This is the third line.\\n')
f.close()
```

### 3. 上下文管理器

#### 内存泄露问题与上下文管理器

内存泄露的根本原因在于创建了某个对象，却没有及时的释放掉，直到程序结束前，这个未被释放的对象一直占着内存；如果量大那么就会直接把内存占满，导致程序被kill掉。

在任何一门编程语言中，文件的输入输出、数据库的连接断开等，都是很常见的资源管理操作。但资源都是有限的，在写程序时，我们必须保证这些资源在使用过后得到释放，不然就容易造成资源泄露，轻者使得系统处理缓慢，重则会使系统崩溃。

在 Python 中，对应的解决方式便是上下文管理器（context manager），能够自动分配并且释放资源，其中最典型的应用便是 with 语句

#### **上下文管理器的实现**

- 基于类的上下文管理器

```python
# 定一个类进行文件管理
class FileManager:
    def __init__(self, name, mode):
        print('calling __init__ method')
        self.name = name
        self.mode = mode 
        self.file = None
    # 在类中实现__enter__，并完成文件的打开操作
    def __enter__(self):
        print('calling __enter__ method')
        self.file = open(self.name, self.mode)
        return self.file
    # 在类中实现__exit__，并完成文件的关闭操作
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('calling __exit__ method')
        if self.file:
            self.file.close()

# 使用with语句来执行上下文管理器
with FileManager('test.txt', 'w') as f:
    print('ready to write to file')
    f.write('hello world')

# 当我们用 with 语句，执行这个上下文管理器时：
# 1. 方法`__init__()`被调用，程序初始化对象 FileManager，使得文件名（name）是"test.txt"，
#    文件模式 (mode) 是'w'；
# 2. 方法`__enter__()`被调用，文件“test.txt”以写入的模式被打开，并且返回 FileManager 对象
#    赋予变量 f；
# 3. 字符串“hello world”被写入文件“test.txt”；
# 4. 方法`__exit__()`被调用，负责关闭之前打开的文件流。

# 最终的输出结果：
>>> calling __init__ method
>>> calling __enter__ method
>>> ready to write to file
>>> calling __exit__ method
```

`__exit__()`方法中的参数“exc_type, exc_val, exc_tb”，分别表示 exception_type、exception_value 和 traceback。当我们执行含有上下文管理器的 with 语句时，如果有异常抛出，异常的信息就会包含在这三个变量中，传入方法`__exit__()`。

```python
# 定义一个类用于数据库操作管理
class DBCM: 
   # 负责对数据库进行初始化，也就是将主机名、接口（这里是 localhost 和 8080）分别赋予变量 hostname 和 port；
    def __init__(self, hostname, port): 
        self.hostname = hostname 
        self.port = port 
        self.connection = None
   # 连接数据库，并且返回对象 DBCM；
    def __enter__(self): 
        self.connection = DBClient(self.hostname, self.port) 
        return self
   # 负责关闭数据库的连接
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.connection.close() 
  
with DBCM('localhost', '8080') as db_client: 
    ....
```

- 基于生成器的上下文管理器

使用装饰器 contextlib.contextmanager，来自定义基于生成器的上下文管理器，用以支持 with 语句。

```python
from contextlib import contextmanager

@contextmanager
def file_manager(name, mode):
    try:
        f = open(name, mode)
        yield f
    finally:
        f.close()
        
with file_manager('test.txt', 'w') as f:
    f.write('hello world')
```

#### 基本语法

```python
with EXPR as VAR:
    BLOCK
file = 'C:/Users/27110/Desktop/lesson_file00_case03.txt'
with open(file, 'a') as f:
	f.write('This is the second line.\\n')
	f.write('This is the third line.\\n')
```



## 04_os模块

### 1. 简介

os模块提供了多数操作系统的功能接口函数。当os模块被导入后，它会自适应于不同的操作系统平台，根据不同的平台进行相应的操作。

dir 目录：“D:\\Python_study\\module8\\ab\\”

path 路径："D:\\Python_study\\module8\\ab\\cd.py"

### 2. os 模块使用

#### os模块——系统操作

- os.sep

获取系统路径分割符

- os.name

获取当前操作系统的名字，如Windows 返回 'nt'; Linux 返回'posix’

- os.getenv('path')

读取 path 环境变量

- os.getcwd()

获取当前工作的目录

#### os模块——目录操作（增删改查）

- os.listdir(dir)

列出目录下所有的文件和目录名（以列表的形式全部列举出来）

若不给参数，则会在执行文件的dir下去枚举

```python
# 假设当前目录为"D:\\python_study\\folder1"
print(os.listdir())  # 输出"D:\\python_study\\folder1"目录下所有的文件和目录名
print(os.listdir(r'D:\\python_study\\folder2'))  # 输出"D:\\python_study\\folder2"目录下所有的文件和目录名
```

- os.mkdir(dir)

创建目录

- os.rmdir(dir)

删除**空目录**（目录中有文件则无法删除）

- os.makedirs(dir)

递归创建目录，若目录全部存在，则创建失败

- os.removedirs(dir)

递归删除**空目录**，若目录中有文件则无法删除

- os.chdir(dir)

改变当前工作目录到指定目录

- os.remove(path)

删除文件

- os.rename(old_path_or_dir, new_path_or_dir)

对目录或文件重命名，若重命名后文件存在则重命名失败

#### os.path模块——判断

- os.path.isfile(path)

判断对象是否是文件

- os.path.isdir(dir)

判断对象是否是目录

- os.path.exists(path)

检验指定的对象是否存在

#### os.path模块——目录操作

- os.path.basename(path)

返回文件名

- os.path.dirname(path)

获得绝对目录，path本身必须是一个绝对路径

- os.path.getsize(path)

获得文件的大小，如果为目录，返回0

- os.path.abspath(path)

返回文件的绝对路径，path本身必须是一个相对路径

- os.path.split(path)

返回路径的目录和文件名

- os.path.splitext(path)

返回路径的文件名和后缀

- os.path.join(path, name)

连接目录和文件名，与 os.path.split(path) 相对

#### 其他

- os.system(cmd)

执行shell命令

```python
"""
删除一个目录及目录下的所有文件
"""
import os

###### 方法1：逐个删除

# 1. 删除目录下的所有文件
os.remove(r'D:\\Python_study\\module8\\ab\\ab.py')
os.remove(r'D:\\Python_study\\module8\\ab\\cd.py')
# 2. 删除目录
os.rmdir('ab')

###### 方法2：批量删除

dir_name = r'D:\\Python_study\\module8\\folder3'
file_list = os.listdir(dir_name)
# print(file_list)
for i in file_list:
    path = os.path.join(dir_name, i)
    # print(path)
    if os.path.isfile(path):
        os.remove(path)

    else:
        new_file_list = os.listdir(path)
        # print(new_file_list)
        for j in new_file_list:
            new_path = os.path.join(path, j)
            print(new_path)
            os.remove(new_path)
        os.rmdir(path)

os.rmdir(dir_name)

###### 方法2改进：批量删除（递归函数）

def remove_all(dir_name):
    file_list = os.listdir(dir_name)

    for i in file_list:
        path = os.path.join(dir_name, i)

        if os.path.isfile(path):
            os.remove(path)

        else:
            remove_all(path)
            # os.remove(path)
    os.rmdir(dir_name)

remove_all(r'D:\\Python_study\\module8\\folder3')
```



## 05_shutil 模块

### 1. 简介

主要用于文件或目录的复制或归档的操作

### 2. shutil 模块使用

#### 文件复制

- **shutil.copyfile(src, dst)**

src 和 dst 均为文件路径，非目录

若 dst 已存在则直接覆盖

保证文件后缀一致

```python
import shutil
shutil.copyfile('C:/a.txt', 'D:/b.txt')
```

- **shutil.copy(src, dst)**

src 为文件路径，dst 可为文件或目录

若 dst 已存在则直接覆盖

若给的是目录，则保留原来的文件名

- **shutil.copy2(src, dst)**

和 copy 操作类似，但同时拷贝了原文件的元数据（包括最后修改时间等）

- **shutil.copytree(olddir, newdir, symlinks=False, ignore=None, copy_function=copy2, ignore_dangling_symlinks=False, dirs_exist_ok=False)**

拷贝目录下的所有文件（空目录也不会报错）

若 newdir 存在，则无法覆盖

- **shutil.copyfileobj(open(src, 'r'), open(dst, 'w’))**

文件内容复制

#### 文件移动/重命名

- **shutil.move(src, dst)**

将文件或目录从 src 移动到 dst

1. dst可以是一个文件路径，也可以是一个文件夹，若给了文件名，则可以重命名
2. 若 dst 目录中存在 src 文件，则不可覆盖；若 dst 为文件路径，则可以覆盖
3. 若 dst 目录中存在 src 目录，则不可覆盖（文件夹无法重命名）

#### 文件删除

- **shutil.rmtree(dir)**

移除目录，不论是否为空目录

#### 归档和解包

- **shutil.make_archive(base_name, format, base_dir)**

归档函数，将多个文件合并到一个文件中

base_name：压缩包的文件名，也可以是压缩包的路径；只是文件名时，则保存至当前目录下，否则保存至指定路径；

format：压缩包种类，“zip”, “tar”, “bztar”，“gztar”；

base_dir：指定要压缩文件的路径，可以指定路径下的文件名，也可以指定路径；

```python
import shutil
# 将path_1处的文件归档到path_2处
path_1 = r'C:\\Users\\hasee\\Desktop\\test007'
path_2 = r'C:\\Users\\hasee\\Desktop\\new'
new_path = shutil.make_archive(path_2, 'zip', path_1)
print(new_path)
--->C:\\Users\\hasee\\Desktop\\new.zip
```

此操作可能会出现递归拷贝压缩导致文件损坏

- **shutil.unpack_archive(filename, extract_dir)**

解包函数，将归档的文件进行释放

filename：需要解包的文件，需要写明文件的后缀

extract_dir：解包后文件存放位置

- **shutil.get_archive_formats()**

获取当前系统已注册的归档文件格式（后缀）

- **shutil.get_unpack_formats()**

获取当前系统已经注册的解包文件格式(后缀)



## 06_time 模块

### 1. 简介

在Python中，与**时间处理**相关的模块有：time、datetime 以及 calendar 。

### 2. Python 中的时间表示方式

- 时间戳

时间戳表示是从1970年1月1号 00:00:00开始到现在按秒计算的偏移量

- 元组方式：struct_time元组共有9个元素

```python
tm_year ：年
tm_mon ：月（1-12）
tm_mday ：日（1-31）
tm_hour ：时（0-23）
tm_min ：分（0-59）
tm_sec ：秒（0-59）
tm_wday ：星期几（0-6,0表示周日）
tm_yday ：一年中的第几天（1-366）
tm_isdst ：是否是夏令时（默认为-1）
```

- UTC（世界协调时），就是格林威治天文时间，即世界标准时间。在中国为UTC+8、DST夏令时

### 3. time 模块使用

#### 返回当前时间

- **time.time()**

返回当前时间的时间戳

- **time.localtime()**

返回当前时间的 struct_time

- **time.gmtime()**

返回当前时间的 UTC 时间（结果也为 struct_time 结构）

#### 时间类型转化——时间戳、元组、字符串

- **time.localtime(secs)**

将时间戳转化为 struct_time

- **time.gmtime(secs)**

将时间戳转化为 UTC 的 struct_time

- **time.mktime(t)**

将 struct_time 转化为时间戳

- **time.asctime([t])**

将 struct_time 转化为‘Sun Aug 23 14:31:59 2015’ 这种形式，如果未指定参数，会将time.localtime()作为参数传入

- **time.ctime([secs])**

把一个时间戳转化为 time.asctime() 的形式。如果未指定参数，将会默认使用time.time()作为参数。

- **time.strftime(format[, t])**

把一个代表时间的 struct_time 转化为格式化的时间字符串，格式由参数format决定。如果 t 未指定，将传入time.localtime()。

```python
%a  本地星期名称的简写（如星期四为Thu）      
%A  本地星期名称的全称（如星期四为Thursday）      
%b  本地月份名称的简写（如八月份为agu）    
%B  本地月份名称的全称（如八月份为august）       
%c  本地相应的日期和时间的字符串表示（如：15/08/27 10:20:06）       
%d  一个月中的第几天（01 - 31）  
%f  微妙（范围0.999999）    
%H  一天中的第几个小时（24小时制，00 - 23）       
%I  第几个小时（12小时制，0 - 11）       
%j  一年中的第几天（001 - 366）     
%m  月份（01 - 12）    
%M  分钟数（00 - 59）       
%p  本地am或者pm的相应符      
%S  秒（00 - 61）    
%U  一年中的星期数。（00 - 53）星期天是一个星期的开始，第一个星期天之前的所有天数都放在第0周。     
%w  一个星期中的第几天（0 - 6，0是星期天）    
%W  和%U基本相同，不同的是%W以星期一为一个星期的开始。    
%x  本地相应日期字符串（如15/08/01）     
%X  本地相应时间字符串（如08:08:10）     
%y  去掉世纪的年份（00 - 99）两个数字表示的年份       
%Y  完整的年份（4个数字表示年份）
%z  与UTC时间的间隔（如果是本地时间，返回空字符串）
%Z  时区的名字（如果是本地时间，返回空字符串）       
%%  ‘%’字符
>>> time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
'2022-01-04 10:38:14'
```

- **time.strptime(string[,format])**

time.strftime() 函数的逆操作，根据指定的格式把一个时间字符串解析为时间元组，返回的是struct_time对象。

```python
>>> stime = "2015-08-24 13:01:30"
>>> formattime = time.strptime(stime,"%Y-%m-%d %H:%M:%S")
>>> print formattime
time.struct_time(tm_year=2015, tm_mon=8, tm_mday=24, tm_hour=13, tm_min=1, tm_sec=30, tm_wday=0, tm_yday=236, tm_isdst=-1)
```

#### 其他

- **time.sleep(s)**

程序休息 s 秒



## 07_re模块

### 1. 正则表达式的作用

匹配文本

### 2. re模块简介

- re模块常用函数：

re.match()

re.search()

re.findall()

re.sub()

re.split()

- 以上函数使用语法：

re.match(pattern, string, flags)

string —— 要匹配的字符串

pattern —— 正则表达式（匹配规则）

flags —— 匹配方式（是否区分大小写、多行匹配）

- flags标记

re.I 忽略大小写（常用）

re.L 做本地化识别（locale-aware）匹配

re.M 多行匹配，影响 ^ 和 $

re.S 使 . 匹配包括换行在内的所有字符

re.U 根据Unicode字符集解析字符。这个标志影响 \w, \W, \b, \B.

re.X 该标志通过给予更灵活的格式，以便将正则表达式写得更易于理解。

### 3. match函数

match()函数从指定字符串的开始位置进行匹配，开始位置匹配成功则继续匹配，否则输出None。

结果是返回一个正则匹配对象。

1. group()方法获取匹配内容
2. span()方法获取匹配位置（开始及结束的索引位置）

```python
import re
content = 'hello world'
result = re.match('heli', content)
print(result)
# result.group(), result.span()
```

```python
content = 'heiio world'
result = re.match('he.', content)  # . 表示匹配任意字符（除换行符\n）
result.group(), result.span()
```

```python
content = 'hello world'
result = re.match('he[a-m]', content)  # [] 表示匹配的字符，常见的有[a-z][A-Z][0-9]
result.group(), result.span()
```

```python
# 若无匹配内容，则返回None
content = 'hello world'
result = re.match('he[a-c]', content)
print(result)
```

```python
content = """heeeeeeeecj"""
result1 = re.match('he*cj', content)  # * 表示对前一个字符匹配零次或任意次
print('result1:', result1)
```

```python
# flags使用——re.S 将多行看成是一行匹配
content = """hello world
hello zyt
hello cj
"""
result1 = re.match('he.*cj', content)  # * 表示对前一个字符匹配零次或任意次
print('result1:', result1)

result2 = re.match('he.*cj', content, re.S)
print('result2:', result2)
print('result2.group:', result2.group())
print('result2.span:', result2.span())
```

```python
# flags使用——re.I 不区分大小写
content = """hello world"""
result1 = re.match('HE', content)  # * 表示对前一个字符匹配零次或任意次
print('result1:', result1)

result2 = re.match('He', content, re.I)
print('result2:', result2)
print('result2.group:', result2.group())
print('result2.span:', result2.span())
```

```python
# group()方法获取内容的时候，索引符号从1开始
content = 'hello world, your name is zyt, my name is cj'
result = re.match('(he.*o).*(yo.*e).*(my.*e).*', content)
print(result)
print(result.group())  # 等价于result.group(0)，返回匹配正则表达式的全部内容
print(result.group(1))  # 返回匹配正则表达式的第一部分内容（即第一个括号），group(N)中的参数N不能超过正则表达式中括号的个数
print(result.group(2))
print(result.group(3))
```

### 4. search函数

re.search()方法扫描整个字符串，返回的是第一个成功匹配的字符串，否则就返回None

```python
content = 'QQ: 1993466, WeChat: 1993466'
result = re.search('QQ: [0-9]*', content)
print(result)
print(result.group())
print(result.span())
```

### 5. findall函数

re.findall()扫描整个字符串，通过列表形式返回所有符合的字符串

```python
content = 'No.1, QQ: 2711090036, WeChat: 19921275330; No.2, qq: 27110, WeChat: 1992127'
result = re.findall('QQ: ([0-9]*)', content, re.I)
print(result)
```

```python
content = """No.1, QQ: 1993466, WeChat: 1993466, 
No.2, qq: 1993466, wechat: 1993466,
No.3, qq: 1993466, wechat: 1993466,
"""
result = re.findall(': (.*),.*: (.*),', content)
print(result)
```

### 6. sub函数

re.sub()方法是用来替换字符串中的某些内容

```python
# 将空格替换成-
content = 'My name is Anderw     Chen.'
# result = content.replace(' ', '-')
result = re.sub('\s+', '-', content)  # \s 表示空格；+ 表示前一个字符匹配一次或多次
print(result)
```

### 7. 正则表达式匹配规则

#### 贪婪模式与非贪婪模式

贪婪模式在整个表达式匹配成功的前提下，尽可能多的匹配；而非贪婪模式在整个表达式匹配成功的前提下，尽可能少的匹配

. 表示匹配除换行符外的任意字符

\*表示匹配0次或多次

? 表示匹配0次或1次

.\*? —— 非贪婪模式

.\* —— 贪婪模式

```python
content = 'abbbbaaaabbbabbba'
result1 = re.findall('a.*b', content)
print('result1:', result1)
result2 = re.findall('a.*?b', content)
print('result2:', result2)
result1: ['abbbbaaaabbbabbb']
result2: ['ab', 'aaaab', 'ab']
```

#### 预定义字符

\d  数字
\D  非数字
\s  空白字符，包括空格、\t、\n等
\S  非空白字符
\w  单词字符，包括0-9a-zA-Z
\W  非单词字符

#### 数量词

\* 匹配0次及以上

\+ 匹配1次及以上

?  匹配0次或1次

{m}  匹配m次

{m,n}  匹配m次到n次

```python
content = """No.1, QQ: 66666, WeChat: 2556430, 
No.2, qq: 754365427, wechat: 1993466,
No.3, qq: 3426476403, wechat: 199,
"""
result = re.findall('QQ: [0-9]{6,12}', content, re.I)
print(result)
['QQ: 123456', 'qq: 271109003']
```

#### 边界匹配符

^  匹配字符串开头（多行模式时匹配每一行开头）

$  匹配字符串结尾（多行模式时匹配每一行结尾）

\A  匹配开头

\Z  匹配结尾

```python
content = 'hello world'
result = re.match('.*d.*', content)
print(result)
<re.Match object; span=(0, 11), match='hello world'>
```

#### 逻辑分组

(...)  分组

a|b  或（优先匹配左侧，注意在[]内则表示其本身）

```python
content = """No.1, QQ: 1993466, WeChat: 1993466, 
No.2, qq: 1993466, 微信: 1993466,
No.3, Qq: 1993466, wechat: 1993466,
"""
result = re.findall('(WeChat|微信|wechat): ([0-9]*)', content)
print(result)
[('WeChat', '19921275330'), ('微信', '1992127'), ('wechat', '199')]
```



# 其他

## Conda常用命令

- 新建环境

conda create -n name python=3.7 -y

- 删除环境

conda remove -n name --all

- 查看环境

conda info -e

- 激活环境

conda activate name

conda deactivate

