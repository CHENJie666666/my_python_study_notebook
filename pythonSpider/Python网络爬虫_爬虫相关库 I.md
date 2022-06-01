# 爬虫相关库 I

## urllib库

### 1. request模块

#### 简单实例

```python
import urllib.request
response = urllib.request.urlopen('https://www.python.org')  # 发送请求
print(response.read().decode('utf-8'))  # 输出网页源代码
print(type(response))   # 输出响应类型
```

#### 方法及属性

response中包含了read()、readinto()、getheader(name)、getheaders()、fileno()等方法，
以及msg、version、status、reason、debuglevel、closed等属性

```
print(response.status)  # 输出响应的状态码
print(response.getheaders())    # 输出响应的头信息
print(response.getheader('Server')) # 获取了响应头中的Server值
```

#### urlopen()方法

```python
urllib.request.urlopen(url,data=None,[timeout,]*,cafile=None,capath=None,context=None)
```

- data参数：如果传递了这个参数，则它的请求方式就不再是GET方式，而是POST方式

- timeout参数：用于设置超时时间，单位为秒，意思就是如果请求超出了设置的这个时间，还没有得到响应，就会抛出异常

```
import socket
import urllib.request
import urllib.error
try:
	response = urllib.request.urlopen('http://httpbin.org/get', timeout=0.1)
except urllib.error.URLError as e:
    if isinstance(e.reason, socket.timeout):
    	print('TIME OUT')
```

- context参数：它必须是ssl.SSLContext类型，用来指定SSL设置
- cafile和capath：分别指定CA证书和它的路径，这个在请求HTTPS链接时会有用

### 2. Request类

传递header信息及修改请求方式

```python
class urllib.request.Request(url,data=None,headers={},origin_req_host=None,unverifiable=False,method=None)
```

- data参数：必须传bytes（字节流）类型，若为字典类型，则通过parse转化

```python
data = bytes(urllib.parse.urlencode({'word':'hello'}), encoding＝'utf-8')
```

- headers：请求头，也可通过add_header()方法来添加
- origin_req_host：指的是请求方的host名称或者IP地址
- unverifiable：表示这个请求是否是无法验证的，默认是False（用户没有足够权限来选择接收这个请求的结果）
- method：是一个字符串，用来指示请求使用的方法，比如GET、POST和PUT等

```python
import urllib .request
request = urllib.request.Request('https://python.org')
response = urllib.request.urlopen(request)
```

- 模拟浏览器登录

```python
from urllib import request, parse
import ssl

context = ssl._create_unverified_context() # https协议请求，使用ssl未经验证的上下文
url = 'https://biihu.cc//account/ajax/login_process/'
# 请求头，模拟浏览器
headers = {
    'User-Agent':' Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
}
# 请求参数
dict = {
    'return_url':'https://biihu.cc/',
    'user_name':'abc@gmail.com',
    'password':'123456789',
    '_post_type':'ajax',
}
data = bytes(parse.urlencode(dict),'utf-8')
# 发送请求
req = request.Request(url,data=data,headers=headers,method='POST')
response = request.urlopen(req,context=context)
print(response.read().decode('utf-8'))
```

### 3. Handler类

- 主要类

HITPDefaultErrorHandler：用于处理HTTP响应错误，错误都会抛出HTTPError类型的异常。

HTTPRedirectHandler：用于处理重定向。

HTTPCookieProcessor：用于处理Cookies。

ProxyHandler：用于设置代理，默认代理为空。

HπPPasswordMgr：用于管理密码，它维护了用户名和密码的表。

HTTPBasicAuthHandler：用于管理认证，如果一个链接打开时需要认证，那么可以用它来解决认证问题。

#### 验证

请求需要验证用户名和密码才能访问的网页数据

示例：Handler验证码模拟登陆QQ邮箱.py

```python
from urllib.request import HTTPPasswordMgrWithDefaultRealm, HTTPBasicAuthHandler, build_opener
from urllib.error import URLError

username = 'andrew'
password = 'abcd'
url = 'https://mail.qq.com/'

#实例化HTTPBasicAuthHandler对象，其参数是HTTPPasswordMgrWithDefaultRealm对象
p = HTTPPasswordMgrWithDefaultRealm()
# 添加用户名和密码
p.add_password(None, url, username, password)
# 建立用于处理验证的handler
auth_handler = HTTPBasicAuthHandler(p)

# 创建opener
opener = build_opener(auth_handler)
try:
    # 打开url
    result = opener.open(url)
    html = result.read().decode('gbk')
    print(html)
except URLError as e:
    print(e.reason)
```

#### 代理

示例：Handler_代理.py

```python
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

# 在本地搭建代理，运行在9743端口上
proxy_handler = ProxyHandler({
    'http':'http://127.0.0.1:9743',
    'https':'https://127.0.0.1:9743'
    })  # ProxyHandler，其参数是一个字典，键名是协议类型（比如HTTP或者HTTPS等），键值是代理链接，可以添加多个代理
opener = build_opener(proxy_handler)
try:
    response = opener.open('https://www.baidu.com')
    print('访问成功')
    print(response.read().decode('utf-8'))
except URLError as e:
    print(e.reason)
```

#### Cookies

示例：Handler_cookies.py

```python
import http.cookiejar, urllib.request

"""获取网址的Cookies"""
# # 声明一个CookieJar对象
# cookie = http.cookiejar.CookieJar()
# # 利用HTTPCookieProcessor来构建一个Handler
# handler = urllib.request.HTTPCookieProcessor(cookie)
# # 构建opener
# opener = urllib.request.build_opener(handler)
# response = opener.open('http://www.baidu.com')
# for item in cookie:
#     print(item.name+'='+item.value)

"""将网站Cookies保存到本地"""
# filename = 'cookies.txt'
# cookie = http.cookiejar.MozillaCookieJar(filename)
# # MozillaCookieJar是CookieJar的子类，可以用来处理Cookies和文件相关的事件，比如读取和保存Cookies，可以将Cookies保存成Mozilla型浏览器的Cookies格式
# # cookie = http.cookiejar.LWPCookieJar(filename)  # 保存为成LWP格式的Cookies文件
# handler = urllib.request.HTTPCookieProcessor(cookie)
# opener = urllib.request.build_opener(handler)
# response = opener.open('http://www.baidu.com')
# cookie.save(ignore_discard=True, ignore_expires=True)

"""导入本地Cookies访问网址"""
cookie = http.cookiejar.MozillaCookieJar()
cookie.load('cookies.txt', ignore_discard=True, ignore_expires=True)
handler = urllib.request.HTTPCookieProcessor(cookie)
opener = urllib.request.build_opener(handler)
response = opener.open('http://www.baidu.com')
print(response.read().decode('utf-8'))
```

### 4. 处理异常模块

#### URLError

继承自OSError类，由request模块生的异常都可以通过捕获这个类来处理

```python
from urllib import request, error
try:
    response = request.urlopen('https://cuiqingcai.com/index.htm')
except error.URLError as e:
    print(e.reason)
```

#### HTIPError

URLError的子类，专门用来处理HTTP请求错误，比如认证请求失败等

三个属性：

code：返回HTTP状态码，比如404表示网页不存在，500表示服务器内部错误等。

reason：同父类一样，用于返回错误的原因。

headers：返回请求头。

#### 实例

```python
"""
常见异常处理方法
1. 先捕获HTTPError，获取它的错误状态码、原因、headers等信息。
2. 如果不是HTTPError异常，就会捕获URLError异常，输出错误原因。
3. 最后，用else来处理正常的逻辑。
"""

from urllib import request, error
try:
    response = request.urlopen('https://cuiqingcai.com/index.htm')
except error.HTTPError as e:
    print(e.reason, e.code, e.headers, sep='\n')
except error.URLError as e:
    print(e.reason)
else:
    print('Request Successfully')
```

### 5. parse解析模块

#### urlparse()

（1）实现URL的识别和分段

scheme://netloc/path;params?query#fragment

scheme，代表协议

第一个/符号前面便是netloc，即域名，后面是path，即访问路径

分号;前面是params，代表参数

问号?后面是查询条件query，一般用作GET类型的URL

井号#后面是锚点，用于直接定位页面内部的下拉位置。

（2）API用法

```python
urllib.parse.urlparse(urlstring, scheme='', allow_fragments=True)
```

urlstring：这是必填项，即待解析的URL。

scheme：协议（如http或https等）。假如这个链接无协议信息，会将这个作为默认的协议。如果Url中有scheme信息，就会返回解析出的scheme。

allow_fragments：即是否忽略fragment。如果它被设置为False，fragment部分就会被忽略（当URL中不包含params和query时，fragment便会被解析为path的一部分。）

#### 其他

- urlunparse()

实现了URL的构造

接受的参数是一个可迭代对象，但是它的长度必须是6，否则会抛出参数数量不足或者过多的问题。

- urlsplit()：和urlparse()方法非常相似，只不过它不再单独解析params这一部分，params会合并到path中。
- urlunsplit()：和urlunparse()类似，接收长度为5的参数
- urljoin()：提供一个base_url作为第一个参数，将新的链接作为第二个参数，该方法会分析base_url的scheme、netloc和path这3个内容并对新链接缺失的部分进行补充，最后返回结果
- urlencode()：首先声明了一个字典来将参数表示出来，然后调用urlencode()方法将其序列化为GET请求参数。
- parse_qs()：将一串GET请求参数转化为字典
- parse_qsl()：将参数转化为元组组成的列表
- quote()：将内容转化为URL编码的格式(用这个方法可以将巾文字符转化为URL编码)
- unquote()：进行URL解码

#### 解析实例

```python
# urlparse——URL解析
from urllib.parse import urlparse
result = urlparse('http://www.baidu.com/index.html;user?id=5#comment')
print(type(result), result)

# urlunparse——URL构造
from urllib.parse import urlunparse
data =['http','www.baidu.com','index.html','user','a=6','comment']
print(urlunparse(data))

# urlsplit——URL解析
from urllib.parse import urlsplit
result = urlsplit('http://www.baidu.com/index.html;user?id=S#comment')
print(result)

# urlunsplit——URL构造
from urllib.parse import urlunsplit
data =['http','www.baidu.com','index.html','a=6','comment']
print(urlunsplit(data))

# urljoin——URL构造
from urllib.parse import urljoin
print(urljoin('www.baidu.com','?category=2#comment'))

# urlencode()——URL构造
from urllib.parse import urlencode
params = {
    'name':'germey',
    'age': 22
    }
base_url = 'http://www.baidu.com?'
url = base_url + urlencode(params)
print(url)

# parse_qs
from urllib.parse import parse_qs
query= 'name=germey&age=22'
print(parse_qs(query))

# parse_qsl
from urllib.parse import parse_qsl
query = 'name=germey&age=22'
print(parse_qsl(query))

# quote
from urllib.parse import quote
keyword = '壁纸'
url = 'https://www.baidu.com/s?wd=' + quote(keyword)
print(url)

from urllib.parse import unquote
url = 'https://www.baidu.com/s?wd=%E5%A3%81%E7%BA%B8'
print(unquote(url))
```

### 6. robotparser模块

#### Robots协议

网络爬虫排除标准（Robots Exclusion Protocol），用来告诉爬虫和搜索引擎哪些页面可以抓取，哪些不可以抓取。它通常是一个叫作robots.txt的文本文件，一般放在网站的根目录下。

http://你的网址/robots.txt

User-agent：描述了搜索爬虫的名称，这里将其设置为＊则代表该协议对任何爬取爬虫有效。

Disallow：指定了不允许抓取的目录，设置为／则代表不允许抓取所有页面。

Allow：一般和Disallow一起使用，一般不会单独使用，用来排除某些限制。

#### robotparser

该模块提供了一个类RobotFileParser，它可以根据某网站的robots.txt文件来判断一个爬取爬虫是否有权限来爬取这个网页。

```python
urllib.robotparser.RobotFileParser(url='')
```

set_url()：用来设置robots.txt文件的链接。

read()：读取robots.txt文件并进行分析（务必调用）

parse()：用来解析robots.txt文件

can_fetch()：该方法传人两个参数，第一个是User-agent，第二个是要抓取的URL。返回的内容是该搜索引擎是否可以抓取这个URL

mtime()： 返回的是上次抓取和分析robots.txt的时间

modified()：将当前时间设置为上次抓取和分析robots.txt的时间。

#### 实例

```python
from urllib.robotparser import RobotFileParser
rp = RobotFileParser()  # 创建RobotFileParser对象
rp.set_url('http://www.jianshu.com/robots.txt') # 设置robots.txt的链接
rp.read()
print(rp.can_fetch('*','http://www.jianshu.com/'))
print(rp.can_fetch('*','http://www.jianshu.com/search?q=python&page=l&type=collections'))

# 也可以使用parse()方法执行读取和分析
from urllib.robotparser import RobotFileParser
from urllib.request import urlopen
rp = RobotFileParser()
rp.parse(urlopen('http://www.jianshu.com/robots.txt').read().decode('utf-8').split('\n'))
print(rp.can_fetch('*','http://www.jianshu.com/'))
print(rp.can_fetch('*','http://www.jianshu.com/search?q=python&page=l&type=collections'))
```



## requests库

### 1. 基本用法

- 示例

```python
import requests

r = requests.get('https://www.baidu.com/')
print(type(r))
print(r.status_code)
print(type(r.text))
print(r.text)
print(r.cookies)
```

### 2. 语法

- 各种请求

```python
r = requests.get('https://www.baidu.com/')
r = requests.post('http://httpbin.org/post')
r = requests.put('http://httpbin.org/put')
r = requests.delete('http://httpbin.org/delete')
r = requests.head('http://httpbin.org/get')
r = requests.options('http://httpbin.org/get')
```

- 添加请求参数

get请求

```python
data = {
    'name': 'germey',
    'age': 22,
}
r = requests.get('http://httpbin.org/get', params=data)
```

post请求

```python
r = requests.post('http://httpbin.org/post', data=data)		# 使用元组列表或字典作为参数
r = requests.post('http://httpbin.org/post', json=json)		# 使用json作为参数
```

模拟浏览器

```python
url = 'https://api.github.com/some/endpoint'
headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36'
    }
r = requests.get(url, headers=headers)
```

设置超时

```python
requests.get('https://github.com/', timeout=0.001)
```

### 3. 响应

使用text和content分别获取文本及二进制响应内容

```python
# 获取文本
r = requests.get('https://api.github.com/events')
print(r.text)

# 下载图片
r = requests.get('https://github.com/favicon.ico')
with open('favicon.ico','wb') as f:
    f.write(r.content)
```

响应参数

```python
print(r.encoding)	# 编码方式
print(type(r.status_code), r.status_code)   # 状态码
print(type(r.headers), r.headers)   # 响应头
print(type(r.cookies), r.cookies)  # Cookies
print(type(r.url), r.url)   # Url
print(type(r.history), r.history)   # 请求历史
print(r.json())		# 获取json响应内容
print(r.raw, r.raw.read(10))		# 获取socket流响应内容
```

### 4. 高级用法

#### 文件上传

上传文件至网站

```python
import requests

# 文件上传
files = {'file': open('favicon.ico','rb')}
r = requests.post('http://httpbin.org/post', files=files)
print(r.text)
```

#### 获取和设置Cookies

```python
# 获取cookies
url = 'http://example.com/some/cookie/setting/url'
r = requests.get(url)
r.cookies['example_cookie_name']
```

```python
# 设置cookies
# 在知乎登录界面导入Cookies，复制进请求头信息
headers = {
    'Cookie':'q_c1=25a8d964afc749fc849e6db54c290b53|1612616789000|1612616789000;'
             '_zap=c516b12e-14a7-4d04-b59e-a7c642e68910; '
             'd_c0="ALAReuUF8BGPTvznNc0n34BF7jU0y4XKPqU=|1600945449"; '
             '_ga=GA1.2.1782116301.1600945453; _xsrf=KL97xWawxJByKoakCHxvWURbls2UmVQY; '
             'tst=r; Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1613802114,1613802181,1614088552,1614088642; '
             'SESSIONID=puSAj4L0V6uvMmrdxqF2LGxtjF1i7Y40zL9aCL3CwT3; '
             'JOID=UFATBE2R5yhPNUe3eZT5-kHCR0ls65x6D1t59UjD21M4eS7PE1fhHSo7Q7h5h-nUt5mfjb0PlbvPRKAYbwwT9Wg=; '
             'osd=UVAXBkyQ5yxNNEa3fZb4-0HGRUht65h4Dlp58UrC2lM8ey_OE1PjHCs7R7p4hunQtZiejbkNlLrPQKIZbgwX92k=; '
             'captcha_session_v2="2|1:0|10:1614088649|18:captcha_session_v2|88:bDdRc2x4ZkpOMm5VZWI1eDZ1VUxhak5Mb3JBampIWlRzbmxuMjNBckpJa3VqVlpWY1Z1TmREMXFWQXIvMnZjMg==|66827a8c9fccfe8703ed8685f40de810161feb65d28c791d0baebac3c9ce61bc"; '
             'captcha_ticket_v2="2|1:0|10:1614088663|17:captcha_ticket_v2|244:eyJhcHBpZCI6IjIwMTIwMzEzMTQiLCJyZXQiOjAsInRpY2tldCI6InQwM3VoYU5ER0RGa2xFcGt5dVZvZFdBakxYa0R6VWVzTXhpQm9SR0Y5UlpzdWZsb2JuUEd6LTE1MzUxU0lyUVRWYlduZEJua0liVE1uWWNPWW1IMWpvcFB6TVBsQl96SXFOSzd2WGdLMFNGZU5IaENTLTBJOTdFSmcqKiIsInJhbmRzdHIiOiJASFo3In0=|cbdaba63ae74ec758f4df75aece7a7edb6fd567eaa2ad93f262cb40afdb986b1"; '
             'z_c0="2|1:0|10:1614088663|4:z_c0|92:Mi4xQ3BSYURnQUFBQUFBc0JGNjVRWHdFU1lBQUFCZ0FsVk4xMWNpWVFBSmh5LXZzSFUtTTF5ODdfSXVVMGdGZDdLSVZR|7ba409d69cab802f9dd03bf0838d441791ab8ef3504cf40f050d00cb2c3d4bd0"; Hm_lpvt_98beee57fd2ef70ccdd5ca52b9740c49=1614088667; KLBRSID=c450def82e5863a200934bb67541d696|1614088694|1614088548',
    'Host':'www.zhihu.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36',
}

# 对知乎发送请求
r = requests.get('https://www.zhihu.com',headers=headers)
print(r.text)
```

#### 会话维持

通常用于模拟登录成功之后再进行下一步的操作

```python
# 无Sessions时
requests.get('http://httpbin.org/cookies/set/number/123456789 ')
r = requests.get('http://httpbin.org/cookies')
print(r.text)

# 设置Sessions
s = requests.Session()
s.get('http://httpbin.org/cookies/set/number/123456789')
r = s.get('http://httpbin.org/cookies')
print(r.text)
```

#### SSL证书验证

当发送HTTPS请求的时候，它会检查SSL证书，可以使用verify参数控制是否检查此证书。如果不加verify参数的话，默认是True，会自动验证。

```python
# 验证证书
response = requests.get('https://www.12306.cn')
print(response.status_code)
# 不验证证书
response = requests.get('https://www.12306.cn', verify=False)
print(response.status_code)
```

#### 代理

若代理需要使用HTTP Basic Auth，可以使用类似http://user:password@host:port这样的语法来设置代理('http://user:password@10.10.1.10:3128/')

```python
proxies = {
    'http':'http://110.10.1.10:3128',
    'https':'http://10.10.1.10:1080',
}
requests.get('https://www.taobao.com',proxies=proxies)

# SOCKS协议的代理
proxies = {
    'http':'socks5://user:password@ip:port',
    'https':'socks5://user:password@ip:port'
}
r = requests.get('https://www.taobao.com',proxies=proxies)
```

#### 身份认证

```python
# 传入HTTPBasicAuth类
from requests.auth import HTTPBasicAuth
r = requests.get('http://localhost:5000',auth=HTTPBasicAuth('username','password'))
print(r.status_code)

# 直接传入元祖
r = requests.get('http://localhost:5000', auth=('username','password'))
print(r.status_code)

# OAuth1认证
from requests_oauthlib import OAuth1

url = 'https://api.twitter.com/1.1/account/verify_credentials.json'
auth = OAuth1('YOUR_APP_KEY’,'YOUR_APP_SECRET','USER_OAUTH_TOKEN','USER_OAUTH_TOKEN_SECRET’)
requests.get(url,auth=auth)
```



## BeautifulSoup库

用于解析Html数据

### 1. bs4对象实例化

（1）加载本地对象

```python
fp = open('test.html', 'r', encoding='utf-8')
soup = BeautifulSoup(fp, 'lxml')
```

（2）加载爬取源码

```python
page_text = response.text
soup = BeautifulSoup(page_text, 'lxml')
```

### 2. 常用解析方法和属性

- 获取标签

```python
print(soup.title.string)  # 获取标签内容
print(soup.title.parent.name)	# 获取标签的父级标签名

soup.find(tagName)
soup.find('tagName', class_='atr')   #利用属性查找
soup.find_all('div')    #返回所有div标签的列表

# 根据css选择
soup.select('.atr')  #返回包含id/class/标签的内容列表
soup.select('.tang > ul > a')   #层级选择器
```

- 获取文本内容

```python
soup.a.text		#获取a标签下的全部文本内容
soup.a.get_text()	
soup.a.string   #获取标签下的直系文本内容
```

- 获取标签的属性值

```python
soup.a['herf']
```



## xpath库

用于解析Html数据

### 1. xpath解析原理

实例化etree对象，将需要被解析的源码加载到对象中

调用xpath方法，结合xpath表达式实现标签定位及内容捕获

```python
# 实例化etree对象
from lxml import etree
etree.parse(filePath)   #本地html文档
etree.HTML('page_text')
```

### 2. xpath表达式

- 定位

/html/head/title    从根层级开始定位

//  表示多个层级，或从任意层级开始定位

//div[@class='']    属性定位

//div[@class='']/p[3]   索引定位

- 取文本

/text()  取直属文本

//text()    取所有文本

- 取属性

/@attrName

### 3. 实例

```python
from lxml import etree

parser = etree.HTMLParser(encoding="utf-8")
tree = etree.parse('QQmail.html',parser=parser)
r = tree.xpath('/html/head/title')
print(r)
```

### 4. 注意事项

- xpath表达式中不能有tbody，直接省略即可
