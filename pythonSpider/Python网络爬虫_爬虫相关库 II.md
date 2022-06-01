# 爬虫相关库 II

## selenium库

自动访问浏览器

### 1. 安装

需要安装selenium包和浏览器驱动

| 浏览器       | 驱动下载链接                                                 |
| ------------ | ------------------------------------------------------------ |
| **Chrome**:  | https://sites.google.com/a/chromium.org/chromedriver/downloads<br />https://npm.taobao.org/mirrors/chromedriver/ |
| **Edge**:    | https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/ |
| **Firefox**: | https://github.com/mozilla/geckodriver/releases              |
| **Safari**:  | https://webkit.org/blog/6900/webdriver-support-in-safari-10/ |

之后将驱动安装路径加入环境变量或将解压后的chromedriver.exe文件拖到Python的Scripts目录下

### 2. 官方文档

https://selenium-python.readthedocs.io/

### 3. 使用

- 浏览器声明

Selenium支持Chrome、Firefox、Edge等，还有Android、Black.Berry等手机端的浏览器。另外，也支持无界面浏览器PhantomJS。

```python
from selenium import webdriver

browser = webdriver.Chrome()
browser = webdriver.Firefox()
browser = webdriver.Edge()
browser = webdriver.PhantomJS()
browser = webdriver.Safari()
```

- 访问页面

```python
browser.get('https://www.taobao.com')
```

- 查找节点

查找到节点后可以驱动浏览器完成各种操作

```python
input_first = browser.find_element_by_id('q')   # 利用属性查找
input_second = browser.find_element_by_css_selector('#q')   # 利用CSS选择器查找
input_third = browser.find_element_by_xpath('//*[@id="q"]') # 利用Xpath查找
```

其他获取节点方法：（只匹配第一个节点）

find_element_by_name

find_element_by_link_text

find_element_by_partial_link_text

find_element_by_tag_name

find_element_by_class_name

find_element(By.ID, id) # 通用方法

find_element(By.NAME, name)

查找多个节点：find_elements_by_XXX

- 节点交互

驱动浏览器来执行一些操作

输入文字时用send_keys()方法，清空文字时用clear()方法，点击按钮时用click()方法

- 获取请求信息

```python
browser.current_url		# 获取请求链接
browser.get_cookies()		# 获取cookies
browser.page_source		# 源代码
```

- 动作链

```
from selenium.webdriver import ActionChains
button = browser.find_element_by_id('id')
# 定义动作链
action = ActionChain(browser)
action.click_and_hold(button)
# 拖动
for i in range(5):
	action.move_by_offset(17, 0).perform()  # 水平方向17像素移动
	time.sleep(1)
action.release()  # 释放动作链
```



## Appium库

爬取手机APP数据

### 1. 安装

- 安装node和npm（添加到PATH）：https://nodejs.org/zh-cn/download/
- 安装JDK（配置环境变量）：https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html

- 安装Android SDK：https://developer.android.com/studio#downloads或http://www.android-studio.org/index.php/download/hisversion
- 安装 android 虚拟机：https://www.genymotion.com/download
- 安装 Appium
- 安装 Appium-Python-Client

### 2.  使用

[python爬虫24 | 搞事情了，用 Appium 爬取你的微信朋友圈。 | 通往Python高手之路 (fxxkpython.com)](https://vip.fxxkpython.com/?p=4950)

