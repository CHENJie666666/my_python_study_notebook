# Python Grammar Book III 常用库



## os模块

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



## shutil 模块

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



## time 模块

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



## re模块

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

### 8. 附表

本符号进行的扼要总结。

| 符号                                                         | 解释                             | 示例                | 说明                                                         |
| ------------------------------------------------------------ | -------------------------------- | ------------------- | ------------------------------------------------------------ |
| .                                                            | 匹配任意字符                     | b.t                 | 可以匹配bat / but / b#t / b1t等                              |
| \\w                                                          | 匹配字母/数字/下划线             | b\\wt               | 可以匹配bat / b1t / b_t等<br>但不能匹配b#t                   |
| \\s                                                          | 匹配空白字符（包括\r、\n、\t等） | love\\syou          | 可以匹配love you                                             |
| \\d                                                          | 匹配数字                         | \\d\\d              | 可以匹配01 / 23 / 99等                                       |
| \\b                                                          | 匹配单词的边界                   | \\bThe\\b           |                                                              |
| ^                                                            | 匹配字符串的开始                 | ^The                | 可以匹配The开头的字符串                                      |
| $                  | 匹配字符串的结束                          | .exe$ | 可以匹配.exe结尾的字符串         |                     |                                                              |
| \\W                                                          | 匹配非字母/数字/下划线           | b\\Wt               | 可以匹配b#t / b@t等<br>但不能匹配but / b1t / b_t等           |
| \\S                                                          | 匹配非空白字符                   | love\\Syou          | 可以匹配love#you等<br>但不能匹配love you                     |
| \\D                                                          | 匹配非数字                       | \\d\\D              | 可以匹配9a / 3# / 0F等                                       |
| \\B                                                          | 匹配非单词边界                   | \\Bio\\B            |                                                              |
| []                                                           | 匹配来自字符集的任意单一字符     | [aeiou]             | 可以匹配任一元音字母字符                                     |
| [^]                                                          | 匹配不在字符集中的任意单一字符   | [^aeiou]            | 可以匹配任一非元音字母字符                                   |
| *                                                            | 匹配0次或多次                    | \\w*                |                                                              |
| +                                                            | 匹配1次或多次                    | \\w+                |                                                              |
| ?                                                            | 匹配0次或1次                     | \\w?                |                                                              |
| {N}                                                          | 匹配N次                          | \\w{3}              |                                                              |
| {M,}                                                         | 匹配至少M次                      | \\w{3,}             |                                                              |
| {M,N}                                                        | 匹配至少M次至多N次               | \\w{3,6}            |                                                              |
| \|                                                           | 分支                             | foo\|bar            | 可以匹配foo或者bar                                           |
| (?#)                                                         | 注释                             |                     |                                                              |
| (exp)                                                        | 匹配exp并捕获到自动命名的组中    |                     |                                                              |
| (?&lt;name&gt;exp)                                           | 匹配exp并捕获到名为name的组中    |                     |                                                              |
| (?:exp)                                                      | 匹配exp但是不捕获匹配的文本      |                     |                                                              |
| (?=exp)                                                      | 匹配exp前面的位置                | \\b\\w+(?=ing)      | 可以匹配I'm dancing中的danc                                  |
| (?<=exp)                                                     | 匹配exp后面的位置                | (?<=\\bdanc)\\w+\\b | 可以匹配I love dancing and reading中的第一个ing              |
| (?!exp)                                                      | 匹配后面不是exp的位置            |                     |                                                              |
| (?<!exp)                                                     | 匹配前面不是exp的位置            |                     |                                                              |
| *?                                                           | 重复任意次，但尽可能少重复       | a.\*b<br>a.\*?b     | 将正则表达式应用于aabab，前者会匹配整个字符串aabab，后者会匹配aab和ab两个字符串 |
| +?                                                           | 重复1次或多次，但尽可能少重复    |                     |                                                              |
| ??                                                           | 重复0次或1次，但尽可能少重复     |                     |                                                              |
| {M,N}?                                                       | 重复M到N次，但尽可能少重复       |                     |                                                              |
| {M,}?                                                        | 重复M次以上，但尽可能少重复      |                     |                                                              |

> **说明：** 如果需要匹配的字符是正则表达式中的特殊字符，那么可以使用\\进行转义处理。





## math模块

- 模块导入

```python
import math
import math as m  # 重命名
from math import floor, ceil  # 仅导入需要的函数
from math import * 
```

- 基本函数

| 函数     | 含义                                                         |
| -------- | ------------------------------------------------------------ |
| floor(x) | 返回数字的下整数                                             |
| ceil(x)  | 返回数字的上整数                                             |
| log(x)   | 返回以 e 为底的 x 的对数                                     |
| log10(x) | 返回以10为底的 x 的对数                                      |
| exp(x)   | 返回 e 的 x 次幂                                             |
| modf(x)  | 返回 x 的整数部分与小数部分，两部分的数值符号与x相同，整数部分以浮点型表示 |
| fabs(x)  | 返回数字的绝对值（浮点数）                                   |
| sqrt(x)  | 返回数字 x 的平方根                                          |

```python
import math
a = 10
b = math.log10(a)
print(b)
```

- 三角函数

| 函数       | 含义                  |
| ---------- | --------------------- |
| sin(x)     | 返回 x 弧度的正弦值   |
| cos(x)     | 返回 x 弧度的余弦值   |
| tan(x)     | 返回 x 弧度的正切值   |
| asin(x)    | 返回 x 的反正弦弧度值 |
| acos(x)    | 返回 x 的反余弦弧度值 |
| atan(x)    | 返回 x 的反正切弧度值 |
| degrees(x) | 将弧度转换为角度      |
| radians(x) | 将角度转换为弧度      |

- 常量

| 常量 | 含义     |
| ---- | -------- |
| pi   | 圆周率   |
| e    | 自然对数 |



## random 模块

### 产生随机数

- random.random() —— 返回(0, 1)范围内的随机数
- random.randint() —— 返回指定范围内的随机整数

> random.randint(a, b)，参数说明如下：
>
> a: 表示随机整数范围的起始值
>
> b: 表示随机整数范围的结束值，随机出现的整数包含 b 参数。

- random.uniform() —— 返回指定范围内的随机浮点数

> random.uniform(a, b)，参数说明如下:
>
> a: 表示指定随机范围的起始值，该值可能会出现在结果中。
>
> b: 表示指定随机范围的结束值，该值可能会出现在结果中。

### 抽样

- random.choice() —— 从非空序列中随机选择一个元素

> random.choice(seq) 参数说明如下: 
>
> seq: 表示需要随机抽取的序列。

- random.choices() —— 从序列中随机选择多个元素（放回式抽样）

> random.choices(population, weights=None, *, cum_weights=None, k=1) 参数说明如下: 
>
> population: 表示集群中随机抽取 K 个元素。
>
> weights: 表示相对权重列表。
>
> cum_weights: 表示累计权重，该参数与 weights 参数不能同时存在。

- random.sample() —— 返回序列中的随机N个元素（不放回式抽样）

> random.sample(population, k) 参数说明:
>
> population: 表示需要抽取的可变序列或不可变序列。
>
> k: 表示抽取元素的个数。

### 打乱顺序

- random.shuffle() —— 将序列随机打乱

> random.shuffle(x, random=None) 参数说明:
>
> x: 表示需要随机排列元素的序列。
>
> random: 可选参数，返回 [0.0,1.0)中的随机浮点数，默认情况下，该参数为 random() 方法

### 其他

- random.seed(a) —— 设定随机数种子

```python
import random
print(random.random)
print(random.randint(1, 9))
a = [1, 2, 3, 4, 5, 6, 7]
print(random.choice(a))
print(random.choices(a, k=2))
print(random.shuffle(a))
```



## JSON模块

JSON的数据类型和Python的数据类型是很容易找到对应关系的，如下面两张表所示。

| JSON                | Python       |
| ------------------- | ------------ |
| object              | dict         |
| array               | list         |
| string              | str          |
| number (int / real) | int / float  |
| true / false        | True / False |
| null                | None         |

| Python                                 | JSON         |
| -------------------------------------- | ------------ |
| dict                                   | object       |
| list, tuple                            | array        |
| str                                    | string       |
| int, float, int- & float-derived Enums | number       |
| True / False                           | true / false |
| None                                   | null         |

我们使用Python中的json模块就可以将字典或列表以JSON格式保存到文件中，代码如下所示。

```python
import json


def main():
    mydict = {
        'name': '骆昊',
        'age': 38,
        'qq': 957658,
        'friends': ['王大锤', '白元芳'],
        'cars': [
            {'brand': 'BYD', 'max_speed': 180},
            {'brand': 'Audi', 'max_speed': 280},
            {'brand': 'Benz', 'max_speed': 320}
        ]
    }
    try:
        with open('data.json', 'w', encoding='utf-8') as fs:
            json.dump(mydict, fs)
    except IOError as e:
        print(e)
    print('保存数据完成!')


if __name__ == '__main__':
    main()
```

json模块主要有四个比较重要的函数，分别是：

- `dump` - 将Python对象按照JSON格式序列化到文件中
- `dumps` - 将Python对象处理成JSON格式的字符串
- `load` - 将文件中的JSON数据反序列化成对象
- `loads` - 将字符串的内容反序列化成Python对象



序列化（serialization）在计算机科学的数据处理中，是指将数据结构或对象状态转换为可以存储或传输的形式，这样在需要的时候能够恢复到原先的状态，而且通过序列化的数据重新获取字节时，可以利用这些字节来产生原始对象的副本（拷贝）。与这个过程相反的动作，即从一系列字节中提取数据结构的操作，就是反序列化（deserialization）

大多数网络数据服务（或称之为网络API）都是基于HTTP协议提供JSON格式的数据

到该网站申请。

```Python
import requests
import json


def main():
    resp = requests.get('http://api.tianapi.com/guonei/?key=APIKey&num=10')
    data_model = json.loads(resp.text)
    for news in data_model['newslist']:
        print(news['title'])


if __name__ == '__main__':
    main()
```

在Python中要实现序列化和反序列化除了使用json模块之外，还可以使用pickle和shelve模块
