# Python Grammar Book I 基础篇



# Part I 数据类型

## 1. 变量

### 变量命名

- 只能包括数字、字母、下划线，但不能以数字开头
- 不能与python的关键字重名

<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220512154939122.png" alt="image-20220512154939122" style="zoom:75%;" />

- 推荐命名：驼峰式命名法，如MyName，或myName

### 变量赋值

```python
a = 3
b = 'zyt'
c = 4.6
d = True
```

### 变量的数据类型

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



## 2. 字符串

### 定义

引号：单引号、双引号、三引号（三引号用来表示注释）

### 索引和切片

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

### 转义字符

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

### 字符串运算：拼接、重复、是否包含

```python
a, b = 'xgg', 'bb'
print(a + b)
print(a * 2)
print('x' in a)
print('b' not in b)
```

### 格式化字符串

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

### 字符串内置函数

- python的内置函数：函数名（变量名）—— print()、input()、type()、len()

#### 判断型

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

#### 大小写

| 内置函数                        | 含义                                  |
| ------------------------------- | ------------------------------------- |
| string.capitalize()             | 把字符串的第一个字符大写              |
| string.lower() / string.upper() | 转换 string 中所有大写字符为小写/大写 |
| string.swapcase()               | 翻转 string 中的大小写                |

#### 统计

| 内置函数                                  | 含义                                                         |
| ----------------------------------------- | ------------------------------------------------------------ |
| string.count(str, beg=0, end=len(string)) | 返回 str 在 string 里面出现的次数，如果 beg 或者 end 指定则返回指定范围内 str 出现的次数 |
| max(str)/min(str)                         | 返回字符串 *str* 中最大/小的字母。                           |

#### 编码

| 内置函数                                         | 含义                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ |
| string.decode(encoding='UTF-8', errors='strict') | 以encoding指定的编码格式解码string，如果出错默认报一个ValueError的异常，除非errors指定的是'ignore'或者'replace' |
| string.encode(encoding='UTF-8', errors='strict') | 以 encoding 指定的编码格式编码 string，如果出错默认报一个ValueError 的异常，除非 errors 指定的是'ignore'或者'replace' |
| string.format()                                  | 格式化字符串                                                 |

#### 查找替换

| 内置函数                                            | 含义                                                         |
| --------------------------------------------------- | ------------------------------------------------------------ |
| string.find(str, beg=0, end=len(string))            | 检测 str 是否包含在 string 中，如果 beg 和 end 指定范围，则检查是否包含在指定范围内，如果是返回开始的索引值，否则返回-1 |
| string.index(str, beg=0, end=len(string))           | 跟find()方法一样，只不过如果str不在 string中会报一个异常.    |
| string.rfind(str, beg=0,end=len(string) )           | 类似于 find() 函数，返回字符串最后一次出现的位置，如果没有匹配项则返回 -1。 |
| string.rindex( str, beg=0,end=len(string))          | 类似于 index()，不过是返回最后一个匹配到的子字符串的索引号。 |
| string.replace(str1, str2,  num=string.count(str1)) | 把 string 中的 str1 替换成 str2,如果 num 指定，则替换不超过 num 次. |

#### 删除转换

| 内置函数                         | 含义                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| string.lstrip()                  | 截掉 string 左边的空格                                       |
| string.rstrip()                  | 删除 string 字符串末尾的空格.                                |
| string.strip([obj])              | 在 string 上执行 lstrip()和 rstrip()                         |
| string.translate(str, del="")    | 根据 str 给出的表(包含 256 个字符)转换 string 的字符, 要过滤掉的字符放到 del 参数中 |
| string.maketrans(intab, outtab]) | maketrans() 方法用于创建字符映射的转换表，对于接受两个参数的最简单的调用方式，第一个参数是字符串，表示需要转换的字符，第二个参数也是字符串表示转换的目标。 |

#### 格式调整

| 内置函数                     | 含义                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| string.expandtabs(tabsize=8) | 把字符串 string 中的 tab 符号转为空格，tab 符号默认的空格数是 8。 |
| string.ljust(width)          | 返回一个原字符串左对齐,并使用空格填充至长度 width 的新字符串 |
| string.rjust(width)          | 返回一个原字符串右对齐,并使用空格填充至长度 width 的新字符串 |
| string.zfill(width)          | 返回长度为 width 的字符串，原字符串 string 右对齐，前面填充0 |

#### 其他

| 内置函数                                    | 含义                                                         |
| ------------------------------------------- | ------------------------------------------------------------ |
| string.split(str="", num=string.count(str)) | 以 str 为分隔符切片 string，如果 num 有指定值，则仅分隔 num+1 个子字符串 |
| string.splitlines([keepends])               | 按照行('\r', '\r\n', \n')分隔，返回一个包含各行作为元素的列表，如果参数 keepends 为 False，不包含换行符，如果为 True，则保留换行符。 |
| string.join(seq)                            | 以 string 作为分隔符，将 seq 中所有的元素(的字符串表示)合并为一个新的字符串 |
| string.partition(str)                       | 有点像 find()和 split()的结合体,从 str 出现的第一个位置起,把 字 符 串 string 分 成 一 个 3 元 素 的 元 组 (string_pre_str,str,string_post_str),如果 string 中不包含str 则 string_pre_str == string. |
| string.rpartition(str)                      | 类似于 partition()函数,不过是从右边开始查找                  |



## 3. 数值型

### 运算符

#### 算术运算符

| 运算符 | 含义     |
| ------ | -------- |
| +      | 相加     |
| -      | 相减     |
| *      | 相乘     |
| /      | 相除     |
| %      | 取余数   |
| **     | 幂运算   |
| //     | 整除取商 |

#### 比较运算符

| 运算符 | 含义         |
| ------ | ------------ |
| ==     | 是否相等     |
| !=     | 是否不相等   |
| >      | 是否大于     |
| <      | 是否小于     |
| >=     | 是否大于等于 |
| <=     | 是否小于等于 |

#### 赋值运算符

| 运算符             | 含义                             |
| ------------------ | -------------------------------- |
| =                  | 简单赋值                         |
| +=                 | 加法赋值，a += 1 等效于a = a + 1 |
| 其他算术运算符类似 |                                  |

#### 逻辑运算符

| 运算符 | 含义 |
| ------ | ---- |
| and    | 与   |
| or     | 或   |
| not    | 非   |

#### 成员运算符

- 对可迭代对象使用（包括字符串、列表或元祖）

| 运算符 | 含义       |
| ------ | ---------- |
| in     | 是否包含   |
| not in | 是否不包含 |

#### 身份运算符

- 判断两个对象的存储ID是否相同

| 运算符 | 含义       |
| ------ | ---------- |
| is     | 是否ID相同 |
| is not | 是否ID不同 |

- is 与 == 区别：is 用于判断两个变量引用对象是否为同一个(同一块内存空间)， == 用于判断引用变量的值是否相等

### 数值型内置函数

| 函数             | 含义                               |
| ---------------- | ---------------------------------- |
| abs(x)           | x 的绝对值                         |
| max(x1, x2, ...) | 返回给定参数的最大值               |
| min(x1, x2, ...) | 返回给定参数的最小值               |
| pow(x, y)        | x ** y                             |
| round(x, [n])    | x 的四舍五入值，n 表示保留小数位数 |



## 4. 序列

python中的序列包括字符串、列表、元组、集合和字典

### 4.1 序列通用操作

#### 索引

- sname[x]

- 索引从零开始

#### 切片

sname[start : end : step]

#### 相加和相乘

```python
a = [1, 2, 3]
b = [4, 5, 6]
print(a + b)
print(a * 2)
```

#### 是否包含

```python
print(1 in a)
print(1 not in a)
```

字典的 in 操作只能用来匹配 key

#### 内置函数

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



### 4.2 列表

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



### 4.3 元组

#### 创建元组

```python
s = (1, 2, 3)
t = tuple('abcd')
```

#### 和列表的区别

元组元素不可变！！！



### 4.4 字典

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



### 4.5 集合

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



# Part II 流程控制语句

## 1. 条件语句

### if 语句

```python
age = int(input("请输入你的年龄：") )
if age < 18 :
    print("你还未成年，建议在家人陪同下使用该软件！")
```

### if-else 语句

```python
b = False
if b:
    print('b是True')
else:
    print('b是False')
```

### if-elif-else 语句

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

### if 后面的表达式

当 if 后面的条件为如下元素时，也表示 False

- "" # 空字符串
- [] # 空列表
- ( ) # 空元组
- { } # 空字典
- None # 空值
- 0  # 零值
- 不带返回值的函数：对于没有 return 语句的函数，返回值为空，也即 None

### if 语句的嵌套

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

### pass 语句

表示忽略

### assert 语句

- 格式：assert 表达式, 注释
- 相当于以下条件语句

```python
if 表达式 == True:
    程序继续执行
else:
    程序报 AssertionError 错误
```



## 2. 循环语句

### while 语句

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

### for 语句

#### 语法格式

for 迭代变量 in 字符串|列表|元组|字典|集合：代码块

```python
letter = "xgg love xbb"
#for循环，遍历 add 字符串
for e in letter:
    print(e, end="")
```

#### 应用

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



### 循环嵌套

```python
i = 0
while i<10:
    for j in range(10):
        print("i=",i," j=",j)       
    i=i+1
```

### break 语句

跳出循环

```python
# 获取字符串逗号前的所有单词
letter = 'Hello, Mary!'
for i in letter:
		if i == ',':
				break
    print(i)
```

### continue 语句

跳出当前循环，继续下一次循环

```python
# 获取列表中不能被2整除的所有元素
list1 = [1, 4, 6, 3, 7, 8]
for i in list1:
    if i % 2 == 0:
				continue
		print(i)
```



# Part III 函数

## 1. 不带参数的函数

### 函数结构

```python
# 函数定义
def <函数名>(<形式参数>):
		<函数体>

# 函数调用
<函数名>（<实际参数>）
```

### 示例

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



## 2. 函数参数

### 示例

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

### 形参和实参

形式参数：在定义函数时，函数名后面括号中的参数就是形式参数

实际参数：在调用函数时，函数名后面括号中的参数称为实际参数

实参和形参的区别，就如同剧本选主角，剧本中的角色相当于形参，而演角色的演员就相当于实参



## 3. 函数返回值

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



## 4. 位置参数

实参和形参位置必须一致

```python
def area(height, width):
    return height * width / 2
print(area(4,3))
```



## 5. 关键字参数和默认参数

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



## 6. 可变参数

### *args

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

### **kwargs

函数调用时必须传递关键字参数

```python
def show_book(**kwargs):
		print(kwargs)  # {} 以字典形式输入

show_book()
show_book(bookname='西游记', author='吴承恩', number=5)
show_book(bookname='西游记')
show_book(bookname='西游记', author='吴承恩')
```



## 7. lambda 匿名函数

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



# Part IV 面向对象编程

## 1. 类和对象

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

## 2. 类的创建

### 创建格式

```python
class EasyClass():
    """创建一个xxx类"""
    def func(self, y):
        """这是类方法，可被类的实例对象调用"""
        print(y)
```

类名称首字母要大写，建议驼峰式命名

## 3. 类的基本组成

### 类属性与对象属性

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

- 类属性可以被类和实例化对象调用；对象属性只能被实例化对象调用。

### 类方法

- \__init__()方法为初始化方法，定义对象属性

- self 是所有方法的第一个“参数”

### 私有与共有

- 若要定义私有属性及私有方法，则在方法前加两个下划线，如\__private_attribute和__private_method()

- 但在实际开发中并不建议将属性设置为私有的，因为这会导致子类无法访问。所以通常以单下划线开头来表示属性是受保护的
- 此时若要修改属性值建议通过使用装饰器来包装getter和setter方法

```python
class Person(object):

    def __init__(self, name, age):
        self._name = name
        self._age = age

    # 访问器 - getter方法
    @property
    def name(self):
        return self._name

    # 访问器 - getter方法
    @property
    def age(self):
        return self._age

    # 修改器 - setter方法
    @age.setter
    def age(self, age):
        self._age = age
```

- 限定自定义类型的对象只能绑定某些属性，可以通过在类中定义\_\_slots\_\_变量来进行限定。需要注意的是\_\_slots\_\_的限定只对当前类的对象生效，对子类并不起任何作用。

```python
class Person(object):

    # 限定Person对象只能绑定_name, _age和_gender属性
    __slots__ = ('_name', '_age', '_gender')

    def __init__(self, name, age):
        self._name = name
        self._age = age
       
def main():
    person = Person('王大锤', 22)
    person._gender = '男'
```

### 静态方法

方法是类而并不属于对象

```python
from math import sqrt

class Triangle(object):
    def __init__(self, a, b, c):
        self._a = a
        self._b = b
        self._c = c

    @staticmethod
    def is_valid(a, b, c):
        return a + b > c and b + c > a and a + c > b

    def perimeter(self):
        return self._a + self._b + self._c

def main():
    a, b, c = 3, 4, 5
    # 静态方法和类方法都是通过给类发消息来调用的
    if Triangle.is_valid(a, b, c):
        t = Triangle(a, b, c)
        print(t.perimeter())
        # 也可以通过给类发消息来调用对象方法但是要传入接收消息的对象作为参数
        # print(Triangle.perimeter(t))
    else:
        print('无法构成三角形.')
```



## 4. 继承与多态

### 继承

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

- 方法重写——如果子类和父类定义了相同的属性或方法，则实例化子类对象执行子类方法时使用的是子类的属性及方法

### 多态

不同的子类对象会表现出不同的行为，这个就是多态（poly-morphism）



# Part V 异常处理

## 1. Python常见异常类型

### 常见异常

AssertionError：当 assert 关键字后的条件为假时，程序运行会停止并抛出 AssertionError 异常

AttributeError：当试图访问的对象属性不存在时抛出的异常

IndexError：索引超出序列范围会引发此异常

KeyError：字典中查找一个不存在的关键字时引发此异常

NameError：尝试访问一个未声明的变量时，引发此异常

TypeError：不同类型数据之间的无效操作

ZeroDivisionError：除法运算中除数为 0 引发此异常

### 查看异常种类

#### Exceptions

Python在`exceptions`模块内建了很多的异常类型

```python
import exceptions
print dir(exceptions)
```

具体信息可以查看：[6. Built-in Exceptions — Python 2.7.18 documentation](https://docs.python.org/2.7/library/exceptions.html#bltin-exceptions)

#### BaseException

`BaseException`是最基础的异常类，`Exception`继承了它。`BaseException`除了包含所有的`Exception`外还包含了`SystemExit`，`KeyboardInterrupt`和`GeneratorExit`三个异常。



## 2. 异常捕获

### try-except-else-finnally 语句

#### try-except

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

- `except`语句不是必须的，`finally`语句也不是必须的，但是二者必须要有一个
- `except`语句可以有多个，Python会按`except`语句的顺序依次匹配你指定的异常，如果异常已经处理就不会再进入后面的`except`语句
- `except`语句可以以元组形式同时指定多个异常
- `except`语句后面如果不指定异常类型，则默认捕获所有异常，你可以通过logging或者sys模块获取当前异常
- 如果要捕获异常后抛出异常，使用`raise`，后面不带任何参数或信息

```python
def f1():
    print(1/0)

def f2():
    try:
        f1()
    except Exception as e:
        raise  # don't raise e !!!
```

- 使用内置的语法范式代替try/except

如with语句或getattr()（用来获取一个不确定属性）

```python
name = getattr(test, 'name', 'default')
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

#### else-finally语句

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



## 3. 引发异常

### raise语句

```python
raise [exceptionName(reason)]
```

事实上，raise 语句引发的异常通常用 try except（else finally）异常处理结构来捕获并进行处理

```python
try:
    a = input("输入一个数：")
    #判断用户输入的是否为数字
    if(not a.isdigit()):
        raise ValueError("a 必须是数字")
except ValueError as e:
    print("引发异常：",repr(e))
```

### 自定义异常类型

从`Exception`类继承

```python
class SomeCustomException(Exception):
    pass
```



# Part VI 模块和包

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





## 03_文件操作

### 1. 打开及关闭文件

#### open()函数

用python读取文件（如txt、csv等），第一步要用open函数打开文件，它会返回一个文件对象，这个文件对象拥有read、readline、write、close等方法。

#### 语法：

open(file, mode)

- file：需要打开的文件路径
- mode（可选）：打开文件的模式，如只读、追加、写入等

#### **mode常用的模式：**

| 操作模式 | 具体含义                             |
| -------- | ------------------------------------ |
| `'r'`    | 只能读取 （默认）                    |
| `'w'`    | 只能写入（会先截断之前的内容）       |
| `'x'`    | 只能写入，如果文件已经存在会产生异常 |
| `'a'`    | 追加，将内容写入到已有文件的末尾     |
| `'b'`    | 二进制模式                           |
| `'t'`    | 文本模式（默认）                     |
| `'+'`    | 更新（既可以读又可以写）             |



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



