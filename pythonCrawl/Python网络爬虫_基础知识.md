# 基础知识及技巧

## 0. 谷歌搜索小技巧

- 完全匹配：双引号（“爬虫”）
- 模糊匹配：星号（我真的*在那一个雨季）
- 排除搜索：减号（爬虫 -知乎）
- 特定的网站内搜索：通过「site:」来指定（爬虫 site:zhihu.com）
- 特定文件类型搜索：文件的特定搜索用「filetype:」+ 文件的类型（爬虫 filetype:xls）



## 1. 实用工具

### 谷歌web_scraper自动爬虫插件使用

- 安装及使用

[那个...我不会写代码，可以爬取数据吗？ (qq.com)](https://mp.weixin.qq.com/s?__biz=MzkyNTExNzY4NA==&mid=2247484935&idx=1&sn=ad9f68845455ca35c08c0e11f92aa4a6&chksm=c1ca3b9cf6bdb28a8647bc911079221b790780611e019e628613657ebfbc38e1e317f53ab00f&token=1453775207&lang=zh_CN#rd)

### 手机抓包工具Fiddler

https://www.telerik.com/download/fiddler

Chrome发送请求给服务器的时候会被 Fiddler 拦截下来，修改请求参数后 Fiddler 假装自己是浏览器再发送数据给服务器。服务器接收到 Fiddler 的请求返回数据，返回的数据又被 Fiddler 拦截下来了，Fiddler 对数据进行修改返回给 Chrome

<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220522232856304.png" alt="image-20220522232856304" style="zoom:50%;" />

使用及配置：[python爬虫入门02：教你通过 Fiddler 进行手机抓包 (qq.com)](https://mp.weixin.qq.com/s?__biz=Mzg2NzYyNjg2Nw==&mid=2247489894&idx=1&sn=d620c16bf3fcb4657c8c44152d936fc7&chksm=ceb9e37af9ce6a6c3017158256b06afd5fb1945a4cd05f9db7e27c31606626ee73d0cc44a074&scene=27#wechat_redirect)



## 2. 爬虫基础

- 爬虫：模拟浏览器发送请求
- URL：

协议：http、https、ftp等（超文本传输协议，定义了文本传输的规则）

资源名称：完整的地址，通常包括主机名（域名，即服务器的IP地址）——www.google.com

文件名：访问服务器某个位置的文件路径

端口号：默认访问不需要输入端口（80），如http://www.google.com:80

请求参数：key-value对，如http://www.google.com/?start=1&end=10

- 网页请求过程（如访问百度）

浏览器通过DNS查找baidu.com对应的IP地址

然后向其服务器发送HTTP请求（附带header信息）

baidu服务器301重定向响应（重定向到www.baidu.com）

浏览器请求重定向地址

baidu服务器处理请求，返回HTML响应

- HTTP 的请求方式：GET, POST, PUT, DELETE, HEAD, OPTIONS, TRACE

POST：以 Form 表单的形式将数据提交给服务器

GET请求把请求参数都暴露在URL上，而POST请求的参数放在request body 里面，POST请求方式还对密码参数加了密

- 请求头 Request Header

在做 HTTP 请求的时候除了提交一些参数之外，还定义一些 HTTP 请求的头部信息，比如 Accept、Host、cookie、User-Agent等等

User-Agent是向 baidu 服务器提供浏览器的类型，操作系统版本，浏览插件，浏览器语言等信息。

Accept是告诉服务器说我们需要接收的类型是什么样子的。

Connection:keep-alive 是为了后边请求不要关闭TCP连接。

Cookie 是以文本形式存储，每次请求的时候就会发送给服务器，它可以存储用户的状态，用户名等信息。

- 服务器响应

响应码：200 表示成功请求

响应头：告诉我们数据以什么样的形式展现以及cookie的设置

响应体：服务器返回给我们的数据



## 3. 爬虫库使用

### 常用库

- urllib、requests、beautifulsoup、xpath

- selenium、appium

- scrapy

### json库

网站需要动态更新数据常使用轻量级的json

```python
# 将Python对象转化为json对象
json_data = json.dumps()
# 将json对象转化为Python对象
python_data = json.loads()

# 使用
python_data.get('key')
```

### threading库

- 进程、线程、协程（微线程）

线程：threading.Thread、multiprocessing.dummy

协程：genvent、monkey.patch_all

线程和进程是通过系统调度的，微线程自己调度（函数之间的切换）

- 线程池

```python
import threading
import time
from queue import Queue

class CustomThread(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.__queue = queue

    def run(self):
        while True:
            q_method = self.__queue.get()
            q_method()
            self.__queue.task_done()

def moyu():
    print(" 开始摸鱼 %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

def queue_pool():
    queue = Queue(5)
    for i in range(queue.maxsize):
        t = CustomThread(queue)
        t.setDaemon(True)
        t.start()

    for i in range(20):
        queue.put(moyu)
    queue.join()

if __name__ == '__main__':
    queue_pool()
```

### 数据分析相关库

- 存储：csv、pandas、pymysql、pymongo

- 分析：matplotlib、seaborn、pyechart


