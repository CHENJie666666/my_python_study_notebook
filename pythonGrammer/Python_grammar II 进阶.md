# Python Grammar Book II 进阶



# Part I 高级函数

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



# Part III 进程&线程&协程

## 1. 进程与线程的定义

### 进程

- 进程就是操作系统中执行的一个程序，操作系统以进程为单位分配存储空间，每个进程都有自己的地址空间、数据栈以及其他用于跟踪进程执行的辅助数据，操作系统管理所有进程的执行，为它们合理的分配资源。

- 进程可以通过fork或spawn的方式来创建新的进程来执行其他的任务，进程之间通过进程间通信机制（IPC，Inter-Process Communication）来实现数据共享，具体的方式包括管道、信号、套接字、共享内存区等。

### 线程

- 一个进程还可以拥有多个并发的执行线索，简单的说就是拥有多个可以获得CPU调度的执行单元，这就是所谓的线程。

- 由于线程在同一个进程下，它们可以共享相同的上下文，信息共享和通信更加容易。

### 协程

单线程+异步I/O的编程模型称为协程，有了协程的支持，就可以基于事件驱动编写高效的多任务程序。协程最大的优势就是极高的执行效率，因为子程序切换不是线程切换，而是由程序自身控制，因此，没有线程切换的开销。协程的第二个优势就是不需要多线程的锁机制，因为只有一个线程，也不存在同时写变量冲突，在协程中控制共享资源不用加锁，只需要判断状态就好了，所以执行效率比多线程高很多。如果想要充分利用CPU的多核特性，最简单的方法是多进程+协程，既充分利用多核，又充分发挥协程的高效率，可获得极高的性能。

## 2. 多进程与多线程

### 多进程

多进程相当于多核处理，可以把任务平均分配给每一个核，并且让它们同时进行

### 多线程



### 并发与并行

在单核CPU系统中，真正的并发是不可能的，因为在某个时刻能够获得CPU的只有唯一的一个线程，多个线程共享了CPU的执行时间。

### GIL（global interpreter lock，全局解释器锁）

python代码执行由python虚拟机（解释器主循环）来控制。对python虚拟机的访问由GIL控制，GIL保证同一时刻只有一个线程在执行。由于GIL的限制，python多线程实际只能运行在单核CPU。如要实现多核CPU并行，只能通过多进程的方式实现。

### fork()函数

Unix和Linux操作系统上提供了 `fork()`系统调用来创建进程，调用 `fork()`函数的是父进程，创建出的是子进程，子进程是父进程的一个拷贝，但是子进程拥有自己的PID。`fork()`函数非常特殊它会返回两次，父进程中可以通过 `fork()`函数的返回值得到子进程的PID，而子进程中的返回值永远都是0。Python的os模块提供了 `fork()`函数。

### 并行的好处

任务分为计算密集型和I/O密集型。计算密集型任务的特点是要进行大量的计算，消耗CPU资源，比如对视频进行编码解码或者格式转换等等，这种任务全靠CPU的运算能力，虽然也可以用多任务完成，但是任务越多，花在任务切换的时间就越多，CPU执行任务的效率就越低。计算密集型任务由于主要消耗CPU资源，这类任务用Python这样的脚本语言去执行效率通常很低，最能胜任这类任务的是C语言，我们之前提到过Python中有嵌入C/C++代码的机制。

除了计算密集型任务，其他的涉及到网络、存储介质I/O的任务都可以视为I/O密集型任务，这类任务的特点是CPU消耗很少，任务的大部分时间都在等待I/O操作完成（因为I/O的速度远远低于CPU和内存的速度）。对于I/O密集型任务，如果启动多任务，就可以减少I/O等待时间从而让CPU高效率的运转。有一大类的任务都属于I/O密集型任务，这其中包括了我们很快会涉及到的网络应用和Web应用。



## 3. Multiprocessing模块

由于Windows系统没有 `fork()`调用，因此要实现跨平台的多进程编程，可以使用multiprocessing模块的 `Process`类来创建子进程，而且该模块还提供了更高级的封装，例如批量启动进程的进程池（`Pool`）、用于进程间通信的队列（`Queue`）和管道（`Pipe`）等。

##### Process类

```python
multiprocessing.Process(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None)
```

- 参数：

group为预留参数。

target为可调用对象（函数对象），为子进程对应的活动；相当于multiprocessing.Process子类化中重写的run()方法。

name为线程的名称，默认（None）为"Process-N"。

args、kwargs为进程活动（target）的非关键字参数、关键字参数。

deamon为bool值，表示是否为守护进程。

- 使用

```python
from multiprocessing import Process
from os import getpid

def _func(params):
    print('启动下载进程，进程号[%d].' % getpid())
    pass

def main():
    p1 = Process(target=_func, args=(params1, ))	# 创建进程
    p1.start()		# 启动进程
    p2 = Process(target=_func, args=(params2, ))
    p2.start()
    p1.join()	# 等待进程执行结束后再执行后面代码
    p2.join()

if __name__ == '__main__':
    main()
```

- 每个子进程有自己独立的内存空间，若要同时修改一个变量，则需要使用Queue

##### Pool类

创建进程池

- 语法

```python
multiprocessing.Pool(processes=None, initializer=None, initargs=(), maxtasksperchild=None)
```

> processes ：使用的工作进程的数量，如果processes是None那么使用 os.cpu_count()返回的数量。
>
> initializer： 如果initializer不是None，那么每一个工作进程在开始的时候会调用initializer(*initargs)。
>
> maxtasksperchild：工作进程退出之前可以完成的任务数，完成后用一个新的工作进程来替代原进程，来让闲置的资源被释放。maxtasksperchild默认是None，意味着只要Pool存在工作进程就会一直存活。
>
> context: 用在制定工作进程启动时的上下文，一般使用 multiprocessing.Pool() 或者一个context对象的Pool()方法来创建一个池，两种方法都适当的设置了context。

- 示例

没有返回值的情况：

```python
def f(a, b = value):
    pass

pool = multiprocessing.Pool() 
pool.apply_async(f, args = (a,), kwds = {b : value})
pool.close()
pool.join()
```

> apply_async(func, args=(), kwds={}) 表示异步调用
>
> apply(func, args=(), kwds={}, callback=None, error_callback=None) 则表示排队执行（阻塞的）
>
> callback为回调函数（在func执行完毕后执行），其应具有一个参数，该参数为func的返回值（也即func应有一个返回值）。

返回值的情况：建议采用map方式（子进程活动只允许1个参数）

```python
def f(a): #map方法只允许1个参数
    pass

pool = multiprocessing.Pool() 
result = pool.map_async(f, (a0, a1, ...)).get()
pool.close()
pool.join()
```

> map(func, iterable, chunksize=None) 阻塞
>
> map_async(func, iterable, chunksize=None, callback=None, error_callback=None) 并行

如果内存不够用，也可采用imap和imap_unordered迭代器方式

如果子进程活动具有多个参数，则采用starmap和starmap_async

##### 进程间通信——数据共享

- 共享值（共享内存）

```python
multiprocessing.Value(typecode_or_type, *args, lock=True)
multiprocessing.RawValue(typecode_or_type, *args) #也有简化的共享值，其不具备锁功能。
```

> typecode_or_type：数组中的数据类型，为代表数据类型的类或者str。比如，'i'表示int，'f'表示float。
>
> args：可以设置初始值。比如：multiprocessing.Value('d',6)生成值为6.0的数据。
>
> lock：bool，是否加锁。

共享单个数据，其值通过value属性访问

- 共享数组（共享内存）

```python
multiprocessing.Array(typecode_or_type, size_or_initializer, *, lock=True)
multiprocessing.RawArray(typecode_or_type, size_or_initializer)
```

> size_or_initializer：数组大小，int；或初始数组内容（序列）。比如：multiprocessing.Array('i', 10)生成的数组转为list为[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]； multiprocessing.Array('i', range(10))生成的数组转为list为[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]。

其返回的数组实例可通过索引访问

- 更复杂情况（共享进程）

```python
m = multiprocessing.Manager()
dic = m.dict() #可采用一般dict的方法访问
```

此进程包含的python对象可以被其他的进程通过proxies来访问。其具有'address', 'connect', 'dict', 'get_server', 'join', 'list', 'register', 'shutdown', 'start'等方法，'Array', 'Barrier', 'BoundedSemaphore', 'Condition', 'Event', 'JoinableQueue', 'Lock', 'Namespace', 'Pool', 'Queue', 'RLock', 'Semaphore', 'Value'等类。

在操作共享对象元素时，除了赋值操作，其他的方法都作用在共享对象的拷贝上，并不会对共享对象生效。

manager不仅可以在本地进程间共享，甚至可以在多客户端实现网络共享。不过manager占用资源较大

##### 进程间通信——数据传递

- 队列Queue

```python
multiprocessing.Queue(maxsize=0)
# maxsize：表示队列允许的最多元素个数，缺省为0，表示不限数量。
```

通过put()方法增加元素，通过get()方法获取元素

```python
put(item, block=True, timeout=None)
```

如果block为True，timeout为None，则将阻塞，直到有一个位置可以加入元素（只有size有限的队列才能阻塞）；如果timeout为非负数值（秒数），则最多阻塞这么长时间。

如果block为False，则直接加入元素，且在无空位可放入元素时直接报Full异常。

```python
get(block=True, timeout=None)
```

如果block为True，timeout为None，则将阻塞，直到有一个元素可以返回；如果timeout为非负数值（秒数），则最多阻塞这么长时间。

如果block为False，则立即返回元素，且在无元素可返回时直接报Empty异常。

```python
#multiprocessing.Queue 与 queue.Queue 用法一致,
msg_format = 'pid:{} {} {}'
 
def getter(q:multiprocessing.Queue):
    while 1:
        value = q.get()                 #阻塞,直到有数据
        print('\t\t'+msg_format.format(multiprocessing.current_process().pid,'get',value))
def setter(q:multiprocessing.Queue):
    pid = multiprocessing.current_process().pid
    for i in range(3):
        q.put(i)
        print(msg_format.format(pid,'set',i))
    q.close()                           #close() 指示当前进程不会在此队列上放置更多数据
 
 
if __name__ == '__main__':
    q = multiprocessing.Queue()
    get_process = multiprocessing.Process(target=getter,args=(q,))
    set_process = multiprocessing.Process(target=setter,args=(q,))
    get_process.start()
    set_process.start()
    pid = os.getpid()
    while 1:
        print('main thread {} . getprocess alive : {} , setprocess alive : {}'.format(
            pid,get_process.is_alive(),set_process.is_alive()
        ))
        time.sleep(5)
```

- 其他Queue

JoinableQueue 可阻塞的队列

```python
multiprocessing.JoinableQueue(maxsize=0)
```

可以通过join()阻塞队列（Queue和JoinableQueue均可通过join_thread()阻塞）

multiprocessing.SimpleQueue() # 简化队列，其只具有empty、get、put3个方法。

- 管道Pipe

```python
multiprocessing.Pipe(duplex=True) 
```

如果duplex为True，则可以双向通信。如果duplex为False，则只能从conn_parent向conn_child单向通信。

方法：

XXX.send(data) #发送数据data。XXX为管道对象。注意，管道只能发送可pickle的数据（自定义类的实例不能pickle，其他一般可以，具体的需要单独文章再讲）。

XXX.recv() #读取管道中接收到的数据。XXX为管道对象。

XXX.poll() #判断管道对象是否有收到数据待读取，返回bool值，通常用来判断是否需要recv()。

```python
import sys,multiprocessing,os,time
 
"""
    使用Pipe 2个进程互相发送数据.
    recv 阻塞 , 直到有数据到来. 
    close 关闭管道
"""
def sender(pipe:multiprocessing.Pipe):
    pipe.send({'1':1,'2':2})
    print('sender 发送完成!')
    pipe.close()
 
def recver(pipe:multiprocessing.Pipe):
    time.sleep(3)
    print('recver ready!')
    obj =  pipe.recv()
    print('recver recv:',obj)
    pipe.close()
 
 
if __name__ =='__main__':
    con1 , con2 = multiprocessing.Pipe()
    p1 = multiprocessing.Process(target=sender,args=(con1,),name='sender')
    p2 = multiprocessing.Process(target=recver,args=(con2,),name='recver')
    p2.start()
    p1.start()
 
    while 1:
        time.sleep(3)
        print( ' active process :' , multiprocessing.active_children())
```

### 2. 多线程

threading模块



```
def download_all_images(list_page_urls):
    # 获取每一个详情妹纸
    works = len(list_page_urls)
    with concurrent.futures.ThreadPoolExecutor(works) as exector:
        for url in list_page_urls:
            exector.submit(download,url)
```


