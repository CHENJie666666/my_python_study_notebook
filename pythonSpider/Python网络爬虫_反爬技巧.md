# 反爬手段

## 1. IP伪装

```python
if __name__ == '__main__':

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'
    }
    url = 'http://127.0.0.1:5000/getInfo'
    response = requests.get(url,headers=headers)
    print(response.text)
```

## 2. 代理

### requests库

```python
proxie = { 
        'http' : 'http://xx.xxx.xxx.xxx:xxxx',
        'http' : 'http://xxx.xx.xx.xxx:xxx',
        ....
    } 
response = requests.get(url,proxies=proxies)	# 使用代理
```

### Github项目：ProxyPool

docker拉起服务：docker-compose -f build.yaml up

返回随机可用代理

```
import requests

PROXY_POOL_URL = 'http://localhost:5555/random'

def get_proxy():
    try:
        response = requests.get(PROXY_POOL_URL)
        if response.status_code == 200:
            return response.text
    except ConnectionError:
        return None
```

## 3. Cookies设置

### 登录获取Cookies

登录后在Request Headers 查找 Cookie

```python
import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/73.0.3683.75 Chrome/73.0.3683.75 Safari/537.36',
    # 把你刚刚拿到的Cookie塞进来
    'Cookie': 'eda38d470a662ef3606390ac3b84b86f9; Hm_lvt_f1d3b035c559e31c390733e79e080736=1553503899; biihu__user_login=omvZVatKKSlcXbJGmXXew9BmqediJ4lzNoYGzLQjTR%2Fjw1wOz3o4lIacanmcNncX1PsRne5tXpE9r1sqrkdhAYQrugGVfaBIicq7pZgQ2pg38ZzFyEZVUvOvFHYj3cChZFEWqQ%3D%3D; Hm_lpvt_f1d3b035c559e31c390733e79e080736=1553505597',
}

session = requests.Session()	# 维持会话
response = session.get('https://biihu.cc/people/wistbean%E7%9C%9F%E7%89%B9%E4%B9%88%E5%B8%85', headers=headers)
print(response.text)
```

### 表单请求

```python
dict = {
    'return_url':'https://biihu.cc/',
    'user_name':'xiaoshuaib@gmail.com',
    'password':'123456789',
    '_post_type':'ajax',
}
data = bytes(parse.urlencode(dict),'utf-8')
req = request.Request(url,data=data,headers=headers,method='POST')
```

### Selenium 自动登录

```python
username = WAIT.until(EC.presence_of_element_located((By.CSS_SELECTOR, "帐号的selector")))
password = WAIT.until(EC.presence_of_element_located((By.CSS_SELECTOR, "密码的selector")))
submit = WAIT.until(EC.element_to_be_clickable((By.XPATH, '按钮的xpath')))

username.send_keys('你的帐号')
password.send_keys('你的密码')submit.click()
cookies = webdriver.get_cookies()
```



## 4. 验证码

### 图像验证码

OCR

灰度处理+二值化

```python
def convert_img(img,threshold):
    img = img.convert("L")  # 处理灰度
    pixels = img.load()
    for x in range(img.width):
        for y in range(img.height):
            if pixels[x, y] > threshold:
                pixels[x, y] = 255
            else:
                pixels[x, y] = 0
    return img
```

### 滑动验证码

selenium模拟滑块滑动

```python
knob =  WAIT.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#gc-box > div > div.gt_slider > div.gt_slider_knob.gt_show")))
ActionChains(driver).drag_and_drop_by_offset(knob, distance, 0).perform()
```

非匀速滑动

```python
def get_path(distance):
    result = []
    current = 0
    mid = distance * 4 / 5
    t = 0.2
    v = 0
    while current < (distance - 10):
    	if current < mid:
        	a = 2
        else:
        	a = -3
        v0 = v
        v = v0 + a * t
        s = v0 * t + 0.5 * a * t * t
        current += s
        result.append(round(s))
    return result
        
knob = WAIT.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#gc-box > div > div.gt_slider > div.gt_slider_knob.gt_show")))
result = get_path(distance)
ActionChains(driver).click_and_hold(knob).perform()

for x in result:
	ActionChains(driver).move_by_offset(xoffset=x, yoffset=0).perform()

time.sleep(0.5)
ActionChains(driver).release(knob).perform() 
```

### 模拟登录项目

[Kr1s77/awesome-python-login-model: 😮python模拟登陆一些大型网站，还有一些简单的爬虫，希望对你们有所帮助❤️，如果喜欢记得给个star哦🌟 (github.com)](https://github.com/Kr1s77/awesome-python-login-model)

虾米音乐、Facebook、微博、知乎、QQ空间、CSDN、淘宝、百度、果壳、京东、163邮箱、拉钩、B站、豆瓣、猎聘网、微信网页版、Github、网易云音乐、糗事百科、百度贴吧、百度翻译



## 5. CSS加密

[python爬虫反反爬 | 看完这篇，你几乎可以横扫大部分 css 字体加密的网站！ | 通往Python高手之路 (fxxkpython.com)](https://vip.fxxkpython.com/?p=3733)



## 6. JS加密

[python爬虫反反爬 | 像有道词典这样的 JS 混淆加密应该怎么破 (qq.com)](https://mp.weixin.qq.com/s?__biz=Mzg2NzYyNjg2Nw==&mid=2247490003&idx=1&sn=81080d79b623871033c4ebd3b9f1690a&source=41#wechat_redirect)

通过断点查找加密算法，模拟算法实现爬虫

### 常见加密算法

#### MD5加密

- MD5

线性散列算法，产生一个128位（16字节）的散列值，即固定长度的数据

解密方法：暴力破解，使用不同数据进行加密并与密文比对

可以通过多次MD5增加破解成本

账号登录时密码通常采用MD5加密

#### DES/AES加密

- DES/AES

加密和解密过程使用同样的密钥

用AES替代DES

DES加密后密文长度是8的整数倍，AES加密后密文长度是16的整数倍

企业级使用DES足够安全

解密方法：密钥解密，暴力破解（$x$位密钥则需要尝试$2^x$）

- DES算法入口参数

key：7字节，工作密钥

data: 8字节，被加密数据

mode：DES工作方式（padding填充模式）

encrypt/decrypt

#### RSA加密

- RAS

非对称加密算法，公开密钥（publickey）和私有密钥（privatekey）

公钥进行加密，私钥进行解密，私钥是通过公钥计算生成的

js中的关键字：setPublicKey/setPrivateKey

#### base64伪加密

用64个字符表示任意二进制数据，实际上是一种编码

使用 A-Z、a-z、0-9、+、/ 字符组成

关键字：Base64.encode/Base64.decode

### HTTPS加密

http协议数据传输中都是明文，存在数据泄露和篡改风险

使用非对称密钥和证书进行数据加密

### JS加密破解

#### 主要流程

- 网页请求分析关键词
- 断点测试定位关键JS代码

- JS调试工具测试

若有相关变量确实，则定义为空字典

- python实现——PyExecJS

#### PyExecJs

- 安装

安装nodejs开发环境

`pip install PyExecJs`

- 使用

```python
import execjs

# 实例化node对象
node = execjs.get()

# js源文件编译
ctx = node.compile(open('./wechat.js', encoding='utf-8').read())

# 执行js函数
funcName = 'getPwd("{0}")'.format('123456')
pwd = ctx.eval(funcName)
print(pwd)
```

#### JS相关知识

- JS内部定义的变量可以指向this（如navigator）

- serializeArray()：将表单数据序列化
- JS混淆（变相加密）——JS反混淆

浏览器自带的反混淆工具

开发者工具-Source-Settings-Sources-Search in anonymous and content scripts 打钩

关键词全局搜索-VMxx（反混淆后的代码）

- sessionID是保存在cookies中的



## 7. iframe技术

使用selenium中的switch_to.frame

```
browser.get('xxx')
browser.switch_to.frame('id') # 根据iframe的id进行切换作用域
```

